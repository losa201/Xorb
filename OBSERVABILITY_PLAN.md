# XORB Phase G5 Observability & SLO Plan

## Overview

This document defines the comprehensive observability strategy for XORB Phase G5, implementing customer-facing Service Level Indicators (SLIs) and Service Level Objectives (SLOs) with error budget monitoring and burn rate alerting.

## Service Level Indicators (SLIs)

### Core Customer-Facing SLIs

#### 1. Bus Publish-to-Deliver Latency
- **Metric**: `bus_publish_to_deliver_p95_ms`
- **Description**: Time from message publish to consumer delivery
- **Target**: P95 < 100ms
- **SLO**: 99% (1% error budget)
- **Measurement**: Histogram tracking end-to-end latency per tenant/subject

#### 2. Evidence Ingest Latency  
- **Metric**: `evidence_ingest_p95_ms`
- **Description**: Evidence processing end-to-end latency
- **Target**: P95 < 500ms
- **SLO**: 99.5% (0.5% error budget - higher bar for evidence)
- **Measurement**: Histogram tracking from ingest start to completion

#### 3. Authentication Error Rate
- **Metric**: `auth_error_rate`
- **Description**: Authentication request failure rate
- **Target**: < 1% error rate
- **SLO**: 99.9% (0.1% error budget)
- **Measurement**: Counter ratio of failed/total auth requests

#### 4. mTLS Handshake Failure Rate
- **Metric**: `mtls_handshake_fail_rate` 
- **Description**: mTLS connection establishment failure rate
- **Target**: < 0.5% failure rate
- **SLO**: 99.9% (0.1% error budget)
- **Measurement**: Counter ratio of failed/total handshakes

#### 5. Replay Backlog Depth (Read-Only)
- **Metric**: `replay_backlog_depth`
- **Description**: Current replay stream backlog in messages
- **Target**: < 10,000 messages (monitoring only)
- **Measurement**: Gauge from existing G4 replay system

## Error Budget Framework

### Error Budget Calculation
- **Error Budget = (1 - SLO) × Time Window**
- **Burn Rate = Actual Error Rate ÷ Error Budget**
- **Time to Exhaustion = Remaining Budget ÷ Current Burn Rate**

### Alert Thresholds (Google SRE Standards)
- **Fast Burn**: 14.4x burn rate (2% budget in 1 hour) → Critical
- **Medium Burn**: 6x burn rate (5% budget in 6 hours) → Warning  
- **Slow Burn**: 1x burn rate (10% budget in 3 days) → Ticket
- **Low Budget**: < 10% remaining → Warning
- **Critical Budget**: < 5% remaining → Critical

## Technical Implementation

### OpenTelemetry Integration
- **Location**: `/src/api/app/observability/`
- **Components**:
  - `instrumentation.py`: Core OTel setup with Prometheus integration
  - `sli_metrics.py`: SLI metric collection classes  
  - `error_budgets.py`: Error budget tracking and alerting

### Prometheus Metrics
- **Endpoint**: `:8080/metrics` (Prometheus scrape target)
- **Export**: Both pull-based (Prometheus) and push-based (OTLP)
- **Retention**: 15 days for error budget calculations

### Grafana Dashboards
- **Location**: `/infra/monitoring/grafana/dashboards/`
- **Dashboard**: `xorb-slo-error-budgets.json`
- **Features**:
  - Real-time SLI visualization
  - Error budget remaining gauges
  - Multi-window burn rate tracking
  - Tenant-scoped filtering

### Alert Rules
- **Location**: `/infra/monitoring/xorb-slo-alert-rules.yml`
- **Integration**: Prometheus Alertmanager
- **Routing**: PagerDuty for critical, Slack for warnings

## Operational Runbooks

### Bus Latency SLO Breach
**Runbook**: https://wiki.xorb.io/runbooks/slo-bus-latency-fast-burn

#### Symptoms
- P95 bus latency > 100ms sustained
- Error budget burning > 14.4x rate
- Customer impact on real-time operations

#### Investigation Steps
1. Check NATS JetStream metrics for backlog/congestion
2. Verify tenant isolation - check if single tenant causing issues
3. Examine bus publish patterns for traffic spikes
4. Check downstream consumer processing times

#### Mitigation Actions
1. **Immediate**: Enable circuit breakers to shed load
2. **Short-term**: Scale NATS cluster horizontally  
3. **Medium-term**: Implement tenant-level rate limiting
4. **Long-term**: Review bus architecture for bottlenecks

### Evidence Ingest SLO Breach
**Runbook**: https://wiki.xorb.io/runbooks/slo-evidence-ingest-fast-burn

#### Symptoms
- P95 evidence ingest > 500ms
- Higher SLO bar (99.5%) due to legal/compliance requirements
- Error budget burning > 14.4x rate

#### Investigation Steps
1. Check evidence storage backend (S3/IPFS) latencies
2. Examine evidence size distribution for large artifacts
3. Verify encryption/signing overhead in evidence pipeline
4. Check Ed25519 signing performance (Phase G7)

#### Mitigation Actions
1. **Immediate**: Enable evidence size limits temporarily
2. **Short-term**: Scale evidence processing workers
3. **Medium-term**: Implement evidence compression
4. **Long-term**: Optimize signing/verification pipeline

### Authentication SLO Breach  
**Runbook**: https://wiki.xorb.io/runbooks/slo-auth-error-fast-burn

#### Symptoms
- Auth error rate > 1%
- JWT validation failures or timeout issues
- Customer login/API access degraded

#### Investigation Steps
1. Check JWT token expiration and refresh logic
2. Examine auth backend connectivity (database/cache)
3. Verify certificate validity for mTLS
4. Check for brute force attack patterns

#### Mitigation Actions
1. **Immediate**: Enable temporary fallback auth method
2. **Short-term**: Scale auth service horizontally
3. **Medium-term**: Implement auth rate limiting per tenant
4. **Long-term**: Review auth architecture resilience

### Platform Availability Critical
**Runbook**: https://wiki.xorb.io/runbooks/platform-availability-critical

#### Symptoms
- Multiple SLIs breaching simultaneously
- Customer-facing service degradation
- Error budgets exhausting rapidly

#### Investigation Steps
1. Check infrastructure health (k8s, databases, message bus)
2. Examine inter-service dependencies for cascading failures
3. Verify network connectivity and DNS resolution
4. Check for resource exhaustion (CPU/memory/disk)

#### Mitigation Actions
1. **Immediate**: Declare incident, page on-call
2. **Short-term**: Enable service degradation mode
3. **Medium-term**: Scale critical services
4. **Long-term**: Implement chaos engineering to prevent

## Compliance & Reporting

### SLO Reporting
- **Weekly**: SLO compliance summary to engineering teams
- **Monthly**: Error budget utilization report to leadership
- **Quarterly**: SLO review and target adjustment

### Audit Requirements
- **Metric Retention**: 15 days minimum for error budget calculations
- **Alert History**: All SLO alerts stored in audit log
- **Change Tracking**: SLO target changes require approval

### Customer Communication
- **Status Page**: Real-time SLO status at status.xorb.io
- **Incident Reports**: Post-mortem for SLO breaches with customer impact
- **SLA Credits**: Automatic compensation for extended SLO violations

## Make Targets

### Available Commands

```bash
# Initialize observability instrumentation
make obs-instrument

# Deploy Grafana dashboards  
make obs-dashboards

# Validate SLO configuration
make obs-validate

# Generate SLO report
make obs-report

# Test error budget alerting
make obs-test-alerts
```

### Implementation

```makefile
.PHONY: obs-instrument obs-dashboards obs-validate obs-report obs-test-alerts

obs-instrument:
	@echo "🔧 Setting up G5 Observability instrumentation..."
	cd src/api && python -c "from app.observability import setup_instrumentation; setup_instrumentation()"
	@echo "✅ Instrumentation initialized"

obs-dashboards:
	@echo "📊 Provisioning Grafana SLO dashboards..."
	curl -X POST \
		-H "Authorization: Bearer $${GRAFANA_API_KEY}" \
		-H "Content-Type: application/json" \
		-d @infra/monitoring/grafana/dashboards/xorb-slo-error-budgets.json \
		http://localhost:3000/api/dashboards/db
	@echo "✅ Dashboards provisioned"

obs-validate:
	@echo "🔍 Validating SLO configuration..."
	prometheus/promtool check rules infra/monitoring/xorb-slo-alert-rules.yml
	cd src/api && python -c "from app.observability.sli_metrics import get_sli_metrics; m = get_sli_metrics(); print(f'✅ {len(m.get_sli_targets())} SLI targets configured')"

obs-report:
	@echo "📋 Generating SLO compliance report..."
	cd src/api && python -c "from app.observability.error_budgets import generate_slo_report; import json; print(json.dumps(generate_slo_report(), indent=2))"

obs-test-alerts:
	@echo "🚨 Testing error budget alert rules..."
	curl -X POST http://localhost:9093/api/v1/alerts \
		-H "Content-Type: application/json" \
		-d '[{"labels":{"alertname":"XORBSLOBusLatencyFastBurn","severity":"critical","tenant_id":"test"}}]'
	@echo "✅ Test alert sent"
```

## Monitoring Stack Integration

### Prometheus Configuration
```yaml
# Additional scrape target for XORB API SLI metrics
- job_name: 'xorb-api-sli'
  static_configs:
    - targets: ['xorb-api:8080']
  scrape_interval: 15s
  metrics_path: /metrics
```

### Alertmanager Routes
```yaml
route:
  routes:
    - match:
        service: xorb-backplane
      receiver: xorb-platform-team
      group_wait: 5s
      group_interval: 5m
      repeat_interval: 12h
      
    - match:
        alert_type: fast_burn
      receiver: xorb-oncall-pagerduty
      group_wait: 0s
      group_interval: 1m
      repeat_interval: 5m
```

### Dashboard URLs
- **SLO Overview**: http://localhost:3000/d/xorb-slo-error-budgets
- **Bus Metrics**: http://localhost:3000/d/xorb-backplane  
- **Evidence Pipeline**: http://localhost:3000/d/xorb-evidence
- **Authentication**: http://localhost:3000/d/xorb-auth

## Future Enhancements

### Phase G6 Integration
- Tenant-isolated SLI metrics per NATS account
- Per-tenant error budget tracking
- Tenant-specific SLO targets

### Phase G7 Integration  
- Evidence signing latency SLI
- Merkle roll-up processing SLI
- Cryptographic verification performance

### Phase G8 Integration
- WFQ scheduler fairness SLI
- Quota enforcement effectiveness SLI
- Control plane response time SLI

---

*This plan implements Google SRE error budget methodology with XORB-specific customer reliability requirements. All SLIs are designed to reflect actual customer experience rather than internal system metrics.*