# Xorb 2.0 Observability Runbook

## Overview
This runbook covers the observability features implemented in Phase 1 of the Xorb 2.0 roadmap, including Prometheus metrics, structured logging, and Grafana dashboards.

## Prometheus Metrics

### API Metrics
The Xorb API exposes Prometheus metrics at `http://localhost:8000/metrics` when running.

Key metrics include:
- `http_requests_total` - Total HTTP requests by method, handler, and status
- `http_request_duration_seconds` - Request duration histogram
- `xorb_api_requests_inprogress` - Current in-progress requests

### Worker Metrics  
The Xorb Worker exposes Prometheus metrics at `http://localhost:9000/metrics` when running.

Key metrics include:
- `xorb_workflow_executions_total` - Total workflow executions by type and status
- `xorb_activity_executions_total` - Total activity executions by name and status
- `xorb_workflow_duration_seconds` - Workflow execution duration histogram
- `xorb_active_workflows` - Current number of active workflows
- `xorb_worker_health` - Worker health status (1=healthy, 0=unhealthy)

### Accessing Metrics
```bash
# Check API metrics
make metrics

# Or manually:
curl -s http://localhost:8000/metrics | grep http_requests_total

# Check worker metrics  
curl -s http://localhost:9000/metrics | grep xorb_
```

## Structured Logging

All Xorb services use structured JSON logging for consistent log aggregation and analysis.

### Log Format
```json
{
  "event": "API request completed",
  "service": "xorb-api", 
  "level": "info",
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "method": "GET",
  "path": "/health",
  "status_code": 200,
  "duration_ms": 12.5
}
```

### Configuration
Logging is configured in `packages/xorb_core/xorb_core/logging.py` and can be customized per service:

```python
from xorb_core.logging import configure_logging, get_logger

configure_logging(level="INFO", service_name="my-service")
logger = get_logger(__name__)
```

## Grafana Dashboard

### Installation
1. Import the dashboard JSON from `grafana/xorb_api_latency.json`
2. Configure Prometheus as a data source pointing to your metrics endpoints
3. Set the Prometheus target name to match your deployment (default: `xorb-api`)

### Dashboard Panels
- **API Request Count**: Real-time request rate in requests/second
- **API P95 Response Time**: 95th percentile latency in seconds  
- **Request Rate by Endpoint**: Breakdown of requests by API endpoint
- **Response Status Codes**: Distribution of HTTP response codes
- **Worker Health Status**: Current health status of worker services

### Prometheus Configuration
Add these scrape configs to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'xorb-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
    
  - job_name: 'xorb-worker'  
    static_configs:
      - targets: ['localhost:9000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## Troubleshooting

### Metrics Endpoint Not Available
1. Verify the service is running: `curl -f http://localhost:8000/health`
2. Check if metrics are enabled: `curl -s http://localhost:8000/metrics | head`
3. Ensure `ENABLE_METRICS=true` environment variable is set
4. Check service logs for startup errors

### No Metrics Data in Grafana
1. Verify Prometheus is scraping the endpoints successfully
2. Check Prometheus targets status at `http://localhost:9090/targets`
3. Confirm the `job` label matches the dashboard queries (should be `xorb-api`)
4. Generate some API traffic to populate metrics: `for i in {1..10}; do curl -s http://localhost:8000/health; done`

### Structured Logging Issues
1. Check log output format - should be JSON, not plain text
2. Verify logging configuration is called during service startup
3. Test with different log levels: `configure_logging(level="DEBUG")`
4. Check for import errors in the logging module

### Worker Metrics Missing
1. Ensure worker service is running and healthy
2. Check worker logs for startup errors or missing dependencies
3. Verify Temporal connection is successful
4. Test worker metrics endpoint: `curl -f http://localhost:9000/metrics`

## Development Testing

### Local Testing
```bash
# Run all services with docker-compose
make up

# Wait for services to start (60 seconds)
sleep 60

# Test metrics endpoints
make metrics

# Check Grafana dashboard (if running)
open http://localhost:3000
```

### CI Pipeline Testing
The CI pipeline automatically validates:
- Metrics endpoint returns 200 and contains `http_requests_total`
- API health endpoint is accessible
- Structured logging produces JSON output
- Docker containers build and run successfully

### Manual Validation
```bash
# Start API locally
cd services/api
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Make test requests
curl -s http://localhost:8000/health
curl -s http://localhost:8000/metrics | grep http_requests_total

# Cleanup
pkill -f uvicorn
```

## Next Steps

After Phase 1 implementation:
1. Set up Prometheus scraping in your deployment environment
2. Import the Grafana dashboard and configure alerts
3. Configure log aggregation (ELK stack, Loki, etc.)
4. Set up alerting rules for critical metrics (error rates, latency thresholds)
5. Implement distributed tracing (Phase 7 - OTEL)