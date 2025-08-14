# XORB Monitoring Infrastructure

This directory contains monitoring and observability configuration for the XORB platform.

## Components

### Prometheus Configuration
- **prometheus.yml** - Main Prometheus server configuration
- **prometheus/prometheus-rules.yml** - Alert rules including post-release v2025.08-rc1 monitoring

### Grafana Dashboards
- **grafana/dashboards/xorb-release-slo-dashboard.json** - Release SLO monitoring with burn-down panels and CI/CD visibility
- **grafana/dashboards/xorb-slo-error-budgets.json** - Error budget monitoring and SLO compliance tracking

### AlertManager
- **alertmanager.yml** - Alert routing and notification configuration

## Usage

### Import Grafana Dashboard
```bash
# Import the release SLO dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @infra/monitoring/grafana/dashboards/xorb-release-slo-dashboard.json
```

### Validate Prometheus Rules
```bash
# Using Docker
docker run --rm -v $(pwd):/workspace --entrypoint=promtool prom/prometheus:latest check rules /workspace/infra/monitoring/prometheus/prometheus-rules.yml

# Direct (if promtool installed)
promtool check rules infra/monitoring/prometheus/prometheus-rules.yml
```

### Deploy Monitoring Stack
```bash
# Production monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Development stack
docker-compose -f docker-compose.development.yml up -d
```

## Access Points

- **Prometheus**: http://localhost:9092
- **Grafana**: http://localhost:3010 (admin / SecureAdminPass123!)
- **AlertManager**: http://localhost:9093

## Release Monitoring (v2025.08-rc1)

The monitoring stack includes specialized post-release monitoring:

### SLO Dashboards
- Release burn-down panels
- Error budget consumption tracking
- Operational readiness scoring
- CI/CD failure visibility

### Post-Release Alerts
- API latency spikes following deployment
- Failure rate increases post-release
- Tenant fairness imbalance detection
- Evidence verification system monitoring
- Control plane quota enforcement alerts

See [prometheus-rules.yml](prometheus/prometheus-rules.yml) for complete alert definitions.
