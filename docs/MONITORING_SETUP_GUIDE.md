# 📊 Xorb PTaaS Monitoring Dashboard Setup Guide

## Overview

This guide provides comprehensive instructions for setting up monitoring and observability for your Xorb PTaaS deployment on AMD EPYC hardware.

## 🚀 Quick Start

### 1. Deploy Observability Stack

```bash
# Start monitoring services
docker-compose -f compose/observability/docker-compose.observability.yml up -d

# Verify services are running
docker-compose -f compose/observability/docker-compose.observability.yml ps
```

### 2. Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3001 | admin / xorb_grafana_admin_2024 |
| **Prometheus** | http://localhost:9090 | No auth required |

## 📊 Available Dashboards

### 1. Performance Budgets Dashboard
- **Location**: `compose/observability/grafana/dashboards/performance-budgets.json`
- **Features**:
  - API latency percentiles (p50, p95, p99)
  - Scan lag monitoring
  - Payout processing delays
  - Error budget tracking
  - SLI/SLO compliance

### 2. Cost Monitoring Dashboard
- **Location**: `compose/observability/grafana/dashboards/cost-monitoring.json`
- **Features**:
  - Total monthly costs
  - Cost breakdown by service/category
  - GPT token usage tracking
  - Stripe fee monitoring
  - S3 storage costs
  - Budget vs actual comparisons

## 🔧 Configuration

### Prometheus Configuration

**File**: `compose/observability/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'xorb-api'
    static_configs:
      - targets: ['api:8000']
    
  - job_name: 'xorb-orchestrator'
    static_configs:
      - targets: ['orchestrator:8001']
    
  - job_name: 'xorb-postgres'
    static_configs:
      - targets: ['postgres:5432']
```

### Alert Rules

**File**: `compose/observability/prometheus/rules/performance-budgets.yml`

Key alerts configured:
- API latency > 2s (P95)
- Scan processing lag > 5min
- Error rate > 5%
- Memory usage > 80%
- CPU usage > 70%

## 📈 Key Metrics

### Application Metrics

| Metric | Description | Type |
|--------|-------------|------|
| `xorb_api_requests_total` | API request counter | Counter |
| `xorb_api_request_duration_seconds` | Request duration | Histogram |
| `xorb_scan_operations_total` | Scan operations | Counter |
| `xorb_agent_executions_total` | Agent executions | Counter |
| `xorb_campaign_operations_total` | Campaign operations | Counter |

### Performance Budget Metrics

| Metric | Description | SLO Target |
|--------|-------------|------------|
| `xorb_api_latency_p95` | 95th percentile API latency | < 2s |
| `xorb_scan_lag_seconds` | Scan processing delay | < 300s |
| `xorb_payout_delay_seconds` | Payout processing delay | < 1800s |
| `xorb_error_rate` | Application error rate | < 5% |

### Cost Tracking Metrics

| Metric | Description |
|--------|-------------|
| `xorb_monthly_costs_dollars` | Monthly costs by service |
| `xorb_gpt_token_usage_total` | GPT token consumption |
| `xorb_stripe_fees_total` | Stripe processing fees |
| `xorb_s3_storage_bytes` | S3 storage usage |

## 🎯 Performance Budgets

### Error Budget Configuration

```yaml
# 99.9% availability target = 43.2 minutes downtime/month
availability_slo: 0.999
error_budget_minutes: 43.2

# Latency budget
latency_p95_slo: 2000  # milliseconds
latency_p99_slo: 5000  # milliseconds

# Throughput budget
min_requests_per_second: 100
max_requests_per_second: 1000
```

### SLI/SLO Definitions

| Service Component | SLI | SLO Target |
|-------------------|-----|------------|
| **API Availability** | HTTP 2xx responses / total requests | 99.9% |
| **API Latency** | P95 response time | < 2000ms |
| **Scanner Throughput** | Scans completed / hour | > 100/hour |
| **Data Freshness** | Time from scan to result | < 5 minutes |

## 🔍 Monitoring Best Practices

### 1. Dashboard Organization

```
grafana/dashboards/
├── performance-budgets.json    # SLI/SLO tracking
├── cost-monitoring.json        # Financial metrics
├── infrastructure.json         # System resources
└── security.json              # Security metrics
```

### 2. Alert Hierarchy

**P0 - Critical (Page immediately)**
- API down > 1 minute
- Database connection failed
- Error rate > 10%

**P1 - High (Page during business hours)**
- High latency (P95 > 5s)
- Memory usage > 90%
- Disk space > 85%

**P2 - Medium (Ticket only)**
- Cost budget exceeded
- Performance degradation
- Non-critical service issues

### 3. Retention Policies

| Data Type | Retention Period | Resolution |
|-----------|------------------|------------|
| **Raw metrics** | 15 days | 15s |
| **Aggregated 5m** | 90 days | 5m |
| **Aggregated 1h** | 1 year | 1h |
| **Aggregated 1d** | 5 years | 1d |

## 🔧 Custom Metrics Setup

### Adding New Metrics

1. **Instrument your code**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Counter for operations
operation_counter = Counter('xorb_operations_total', 'Operations', ['type', 'status'])

# Histogram for durations
duration_histogram = Histogram('xorb_operation_duration_seconds', 'Duration', ['operation'])

# Gauge for current values
active_connections = Gauge('xorb_active_connections', 'Active connections')
```

2. **Update Prometheus config**:
```yaml
scrape_configs:
  - job_name: 'your-service'
    static_configs:
      - targets: ['your-service:port']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

3. **Create Grafana panels**:
- Use PromQL queries
- Set appropriate thresholds
- Configure alerting rules

## 🚨 Alerting Setup

### Webhook Configuration

```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'xorb-webhook'

receivers:
- name: 'xorb-webhook'
  webhook_configs:
  - url: 'http://api:8000/api/v1/alerts/webhook'
```

### Slack Integration

```yaml
receivers:
- name: 'slack-alerts'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#xorb-alerts'
    title: 'Xorb PTaaS Alert'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

## 💾 Backup Monitoring

### Backup Success Metrics

```python
backup_status = Gauge('xorb_backup_last_success_timestamp', 'Last successful backup')
backup_duration = Histogram('xorb_backup_duration_seconds', 'Backup duration')
backup_size = Gauge('xorb_backup_size_bytes', 'Backup size')
```

### B2 Lifecycle Monitoring

```bash
# Monitor B2 storage costs
python3 scripts/b2_lifecycle_manager.py --report

# Optimize lifecycle policies
python3 scripts/b2_lifecycle_manager.py --optimize
```

## 🔍 Troubleshooting

### Common Issues

1. **Metrics not appearing**:
   - Check service `/metrics` endpoints
   - Verify Prometheus scrape config
   - Check service discovery

2. **High memory usage**:
   - Adjust retention policies
   - Increase storage
   - Optimize queries

3. **Dashboard not loading**:
   - Check Grafana logs
   - Verify data source configuration
   - Test PromQL queries

### Health Checks

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana health
curl http://localhost:3001/api/health

# Verify metrics collection
curl http://localhost:8000/metrics
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
- [SRE Workbook](https://sre.google/workbook/table-of-contents/)

## 🎯 Next Steps

1. **Set up alerting**: Configure notifications for critical metrics
2. **Create custom dashboards**: Build team-specific monitoring views
3. **Implement SLI/SLO tracking**: Monitor service reliability
4. **Set up log aggregation**: Centralize application logs
5. **Performance testing**: Validate monitoring under load

---

**Need help?** Check the troubleshooting section or review the deployment verification results in your latest verification report.