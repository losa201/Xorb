# Xorb 2.0 Changelog

## [Phase 1] - 2024-01-15 - Observability MVP

### Added
- **Prometheus Metrics Integration**
  - Added `prometheus-fastapi-instrumentator` to requirements.txt
  - Configured comprehensive API metrics collection in `services/api/app/main.py`
  - Created Temporal worker metrics in `packages/xorb_core/xorb_core/workflows/worker_entry.py`
  - Metrics exposed at `/metrics` endpoint (API: port 8000, Worker: port 9000)

- **Structured JSON Logging**
  - Created centralized logging configuration in `packages/xorb_core/xorb_core/logging.py`
  - Implemented service-specific logging with structured JSON output
  - Updated API service to use centralized logging configuration
  - Added ISO timestamp formatting and log level filtering

- **Grafana Dashboard**
  - Created `grafana/xorb_api_latency.json` with comprehensive observability panels
  - Includes request count, P95 latency, endpoint breakdown, and worker health monitoring
  - Dashboard optimized for Xorb API and worker metrics

- **Development Tooling**
  - Added `make metrics` command for local metrics validation
  - Enhanced Makefile with proper metrics endpoint checking
  - Created comprehensive documentation in `docs/runbook.md`

- **CI/CD Pipeline**
  - Implemented GitHub Actions workflow in `.github/workflows/ci.yml`
  - Added automated metrics endpoint validation
  - Includes code quality checks, testing, and Docker build validation
  - Observability-specific job validates metrics availability

### Enhanced
- **API Service (`services/api/app/main.py`)**
  - Enhanced Prometheus instrumentator configuration
  - Added service-specific structured logging
  - Improved error handling and request tracking

- **Worker Architecture**
  - Created proper worker entry point with metrics collection
  - Added workflow and activity execution tracking
  - Implemented health monitoring and status reporting

### Technical Details
- **Metrics Collected:**
  - `http_requests_total` - API request counters by method, handler, status
  - `http_request_duration_seconds` - API response time histograms
  - `xorb_workflow_executions_total` - Workflow execution counters
  - `xorb_activity_executions_total` - Activity execution counters
  - `xorb_worker_health` - Worker health status gauge

- **Log Format:** Structured JSON with service name, timestamp, log level, and contextual data
- **Endpoints:** 
  - API metrics: `GET /metrics` (port 8000)
  - Worker metrics: `GET /metrics` (port 9000)
  - Health check: `GET /health` (port 8000)

### Dependencies Added
- `prometheus-fastapi-instrumentator>=6.1.0` - FastAPI metrics instrumentation

### Configuration
- Services now support `ENABLE_METRICS` environment variable
- Logging level configurable via `configure_logging(level="INFO")`
- Metrics server ports configurable via `METRICS_PORT` environment variable

### Documentation
- Added comprehensive observability runbook (`docs/runbook.md`)
- Includes troubleshooting guide and Prometheus configuration examples
- CI pipeline documentation and local testing instructions

---

## Acceptance Criteria Met ✅

1. **✅ `make run-api` equivalent**: Service starts and exposes metrics at `/metrics`
2. **✅ `make metrics` exits 0**: Command successfully validates metrics endpoints  
3. **✅ CI pipeline passes**: GitHub Actions validates metrics endpoint returns `http_requests_total`
4. **✅ Grafana dashboard**: JSON template ready for 60-second data rendering in demo stack

---

*This release implements the complete Observability MVP as specified in the Xorb 2.0 roadmap Phase 1. All acceptance criteria have been validated and the implementation is ready for production deployment.*