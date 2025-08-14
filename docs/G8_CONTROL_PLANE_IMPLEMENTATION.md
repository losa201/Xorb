# XORB Phase G8: Control Plane Implementation

## Overview

**Phase G8** completes the XORB execution plan (G5-G8) by implementing an advanced **Control Plane with Weighted Fair Queuing (WFQ) scheduler and per-tenant quotas**. This provides fair resource allocation across tenants with real-time fairness monitoring and automatic load balancing.

## 🎯 Implementation Summary

### Core Components

#### 1. **Weighted Fair Queuing (WFQ) Scheduler**
- **File**: `src/api/app/services/g8_control_plane_service.py` (WeightedFairQueueScheduler class)
- **Algorithm**: Implements classic WFQ with virtual time progression
- **Features**:
  - Per-tenant weight-based scheduling
  - Virtual finish time calculation
  - Priority queue with heap-based ordering
  - Thread-safe request queuing and dequeuing
  - Fairness guarantees across tenants

#### 2. **Quota Manager**
- **File**: `src/api/app/services/g8_control_plane_service.py` (QuotaManager class)
- **Features**:
  - Per-tenant resource quotas with burst allowance
  - Six resource types: `api_requests`, `scan_jobs`, `storage_gb`, `compute_hours`, `bandwidth_mbps`, `concurrent_scans`
  - Three tenant tiers: `enterprise` (weight: 10.0), `professional` (weight: 3.0), `starter` (weight: 1.0)
  - Usage tracking with time windows
  - Persistent storage of tenant profiles

#### 3. **Fairness Engine**
- **File**: `src/api/app/services/g8_control_plane_service.py` (FairnessEngine class)
- **Features**:
  - Jain's Fairness Index calculation
  - Resource starvation detection
  - Fairness violation alerts
  - Automatic rebalancing recommendations

#### 4. **Control Plane Service**
- **File**: `src/api/app/services/g8_control_plane_service.py` (G8ControlPlaneService class)
- **Features**:
  - Orchestrates WFQ, quota management, and fairness monitoring
  - Background processing loop
  - Request lifecycle management
  - System health monitoring

## 🔌 API Integration

### REST API Endpoints
- **Router**: `src/api/app/routers/g8_control_plane.py`
- **Base Path**: `/api/v1/control-plane`

#### Key Endpoints:
```bash
POST   /control-plane/requests/submit          # Submit resource request
GET    /control-plane/requests/{id}/status     # Get request status
POST   /control-plane/tenants/create           # Create tenant profile
GET    /control-plane/tenants/{id}/status      # Get tenant status
PUT    /control-plane/tenants/{id}/quotas      # Update tenant quotas
GET    /control-plane/system/status            # Get system status
GET    /control-plane/system/fairness-report  # Get fairness analysis
POST   /control-plane/system/rebalance         # Trigger rebalancing
```

### Integration with Main App
- **Added to**: `src/api/app/main.py`
- **Router included**: `g8_control_plane.router` with `/api/v1` prefix
- **Auto-initialization**: Service starts with FastAPI application

## 🛠️ Management Tools

### CLI Tool
- **File**: `tools/scripts/g8_control_plane_cli.py`
- **Executable**: `chmod +x tools/scripts/g8_control_plane_cli.py`

#### Available Commands:
```bash
# Tenant Management
python3 tools/scripts/g8_control_plane_cli.py create-tenant <tenant_id> <tier>
python3 tools/scripts/g8_control_plane_cli.py tenant-status <tenant_id>

# Request Management  
python3 tools/scripts/g8_control_plane_cli.py submit-request <tenant> <resource> [priority] [amount]

# System Monitoring
python3 tools/scripts/g8_control_plane_cli.py system-status
python3 tools/scripts/g8_control_plane_cli.py monitor-fairness [duration]

# Testing & Load Generation
python3 tools/scripts/g8_control_plane_cli.py load-test [tenants] [requests]

# Quota Management
python3 tools/scripts/g8_control_plane_cli.py update-quota <tenant> <resource> <limit>
```

### Make Targets
Added to `Makefile`:

```bash
# Initialization
make g8-control-plane-init        # Create default tenants
make g8-control-plane-status      # Get system status

# Tenant Operations
make g8-tenant-status TENANT_ID=t-enterprise
make g8-submit-request TENANT_ID=t-enterprise RESOURCE_TYPE=api_requests
make g8-update-quota TENANT_ID=t-enterprise RESOURCE_TYPE=api_requests NEW_LIMIT=5000

# Testing & Monitoring
make g8-monitor-fairness          # Real-time fairness monitoring
make g8-load-test                 # WFQ scheduler load test
make g8-test-fairness            # Multi-phase fairness test
make g8-api-test                 # REST API endpoint test

# Management
make g8-cleanup                  # Clean storage and test data
```

## 📊 Resource Types & Quotas

### Resource Types
1. **`api_requests`** - API requests per time window (rate limiting)
2. **`scan_jobs`** - Security scan jobs per time window
3. **`storage_gb`** - Storage capacity in gigabytes  
4. **`compute_hours`** - Compute time allocation in hours
5. **`bandwidth_mbps`** - Network bandwidth in Mbps
6. **`concurrent_scans`** - Maximum concurrent security scans

### Tier-Based Default Quotas

| Resource Type | Starter | Professional | Enterprise |
|---------------|---------|--------------|------------|
| **API Requests** | 500/hr | 2,000/hr | 10,000/hr |
| **Scan Jobs** | 5/day | 25/day | 100/day |
| **Storage GB** | 10 GB | 100 GB | 1,000 GB |
| **Compute Hours** | 20/month | 100/month | 500/month |
| **Bandwidth Mbps** | 50 Mbps | 200 Mbps | 1,000 Mbps |
| **Concurrent Scans** | 2 | 10 | 50 |

### Burst Allowance
- **Formula**: 20% above quota limit for short periods
- **Purpose**: Handle traffic spikes without rejection
- **Example**: Starter API requests = 500 + 100 burst = 600 total

## 🧪 Testing Framework

### Unit Tests
- **WFQ Scheduler**: `tests/unit/g8_control_plane/test_wfq_scheduler.py`
  - Scheduler initialization and state management
  - Fair queuing algorithm validation
  - Virtual time progression
  - Thread safety verification
  
- **Quota Manager**: `tests/unit/g8_control_plane/test_quota_manager.py`
  - Tenant profile creation and persistence
  - Quota consumption and release
  - Usage tracking and statistics
  - Multi-tenant isolation

### Integration Tests
- WFQ fairness under different load patterns
- Quota recovery after window reset  
- Multi-tenant resource isolation
- Burst handling scenarios

## 🎯 Fairness Algorithms

### 1. **Weighted Fair Queuing (WFQ)**
```
Virtual Finish Time = Virtual Start Time + (Service Time / Weight)
```
- Ensures tenants get bandwidth proportional to their weight
- Prevents monopolization by high-traffic tenants
- Provides bounded delay guarantees

### 2. **Jain's Fairness Index**
```
Fairness Index = (Σxi)² / (n × Σxi²)
```
- Range: 0-1 (1 = perfectly fair)
- Measures fairness across all tenants
- Used for system-wide fairness monitoring

### 3. **Resource Starvation Detection**
- Tracks rejected requests per tenant
- Triggers alerts when threshold exceeded
- Enables automatic rebalancing

## 📈 Monitoring & Observability

### System Health Metrics
- Control plane operational status
- Total tenants and queued requests
- Processing statistics (processed/rejected)
- Fairness index and violation count

### Per-Tenant Metrics
- Queue length and wait times
- Resource utilization percentages
- Fairness scores and starvation counts
- Quota usage across all resource types

### Real-Time Monitoring
- **CLI Command**: `g8_control_plane_cli.py monitor-fairness`
- **Features**: Live fairness dashboard with violation alerts
- **Auto-refresh**: Configurable interval monitoring
- **Recommendations**: Automatic rebalancing suggestions

## 🔧 Configuration & Tuning

### Tenant Weights (WFQ)
- **Enterprise**: 10.0x (highest priority)
- **Professional**: 3.0x (medium priority)  
- **Starter**: 1.0x (baseline priority)
- **Custom**: Configurable per tenant

### Rate Limiting Windows
- **60 seconds**: Short-term burst handling
- **3600 seconds**: Hourly rate limits
- **86400 seconds**: Daily usage quotas

### Fairness Thresholds
- **Minimum Fairness Score**: 0.7 (triggers rebalancing)
- **Starvation Threshold**: 10 rejected requests
- **System Fairness Target**: >0.8 (healthy operation)

## 🚀 Deployment & Operations

### Startup Integration
```python
# Automatic initialization in src/api/app/main.py
from .services.g8_control_plane_service import get_g8_control_plane_service

# Service starts with FastAPI application
control_plane = await get_g8_control_plane_service()
```

### Storage Requirements
- **Control Plane Data**: `control_plane_storage/`
- **Quota Data**: `control_plane_storage/quotas/`
- **Tenant Profiles**: JSON-based persistence
- **Usage Tracking**: In-memory with periodic resets

### Production Considerations
- Thread-safe concurrent operations
- Graceful shutdown handling
- Error recovery and circuit breaking
- Monitoring integration with existing systems

## 🎉 Phase Completion Status

✅ **All XORB Execution Plan Phases (G5-G8) Complete**

| Phase | Component | Status | Key Features |
|-------|-----------|--------|--------------|
| **G5** | Observability & SLOs | ✅ Complete | Prometheus metrics, Grafana dashboards, SLI tracking |
| **G6** | Tenant Isolation | ✅ Complete | NATS accounts, quotas, leak testing |
| **G7** | Provable Evidence | ✅ Complete | Ed25519 signatures, trusted timestamps, Merkle rollups |
| **G8** | Control Plane | ✅ Complete | WFQ scheduler, quotas, fairness monitoring |

## 📝 Usage Examples

### 1. Initialize Control Plane
```bash
make g8-control-plane-init
```

### 2. Create Custom Tenant
```bash
python3 tools/scripts/g8_control_plane_cli.py create-tenant my-tenant professional
```

### 3. Submit Resource Request
```bash
make g8-submit-request TENANT_ID=my-tenant RESOURCE_TYPE=scan_jobs PRIORITY=high AMOUNT=3
```

### 4. Monitor System Health
```bash
make g8-control-plane-status
```

### 5. Run Fairness Test
```bash
make g8-test-fairness
```

## 🔍 Verification Commands

```bash
# Test WFQ Scheduler
python3 -m pytest tests/unit/g8_control_plane/test_wfq_scheduler.py -v

# Test Quota Manager  
python3 -m pytest tests/unit/g8_control_plane/test_quota_manager.py -v

# Test CLI Help
python3 tools/scripts/g8_control_plane_cli.py help

# Test API Health
curl http://localhost:8000/api/v1/control-plane/health
```

---

**🎛️ G8 Control Plane provides enterprise-grade resource management with mathematically guaranteed fairness across all tenants while maintaining optimal system performance and utilization.**