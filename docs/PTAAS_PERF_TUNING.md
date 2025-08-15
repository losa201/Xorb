# PTaaS Performance Tuning Guide for AMD EPYC 7002

## Overview

This guide provides comprehensive performance tuning instructions for PTaaS (Penetration Testing as a Service) running on AMD EPYC 7002 series processors. The goal is to achieve sub-2 second P95 latency under realistic parallel load while maintaining fairness across tenants.

## Quick Start

### EPYC 7002 Profile Selection

Choose the appropriate profile based on your EPYC configuration:

```bash
# 8-core EPYC 7002 (development)
export PTAAS_PROFILE=epyc_7002_8core

# 16-core EPYC 7002 (small production)
export PTAAS_PROFILE=epyc_7002_16core

# 32-core EPYC 7002 (medium production) - RECOMMENDED
export PTAAS_PROFILE=epyc_7002_32core

# 64-core EPYC 7002 (large production)
export PTAAS_PROFILE=epyc_7002_64core

# Apply profile and start
make ptaas-perf PTAAS_PROFILE=$PTAAS_PROFILE
```

### Quick Performance Test

```bash
# Full performance validation (10 minutes)
make ptaas-perf

# Quick smoke test (2 minutes)
make ptaas-perf-smoke

# Generate performance report
make ptaas-perf-report
```

## Configuration Parameters

### Core EPYC 7002 Settings

#### Worker Pool Configuration
```bash
# AMD EPYC 7002 32-core recommended settings
export PTAAS_WORKERS=28              # 80% of cores for PTaaS workload
export PTAAS_CPU_POOL=16             # CPU-bound task pool size
export PTAAS_IO_CONCURRENCY=128      # High I/O concurrency for network scanning
export PTAAS_NUMA_AWARE=true         # Enable NUMA topology awareness
export PTAAS_MEMORY_POOL_MB=8192     # 8GB memory pool for task execution
```

#### NATS JetStream Optimization
```bash
# High-throughput NATS configuration for EPYC
export NATS_MAX_ACK_PENDING=10000    # High pending ack count
export NATS_ACK_WAIT_MS=30000        # 30 second ack timeout
export NATS_MAX_DELIVER=5            # Maximum redelivery attempts
export NATS_MAX_INFLIGHT=256         # High in-flight message limit
```

#### Weighted Fair Queueing (G8 WFQ)
```bash
# Tenant isolation and fairness settings
export G8_TENANT_MAX_CONCURRENT=8    # Max concurrent jobs per tenant
export G8_FAIRNESS_WINDOW_SEC=60     # Fairness calculation window
export G8_WEIGHT_HIGH=3.0            # High priority weight
export G8_WEIGHT_MEDIUM=2.0          # Medium priority weight
export G8_WEIGHT_LOW=1.0             # Low priority weight
```

#### Performance Targets
```bash
# SLA targets for EPYC 7002
export PTAAS_TARGET_P95_MS=2000      # P95 latency < 2 seconds
export PTAAS_TARGET_ERROR_RATE=0.005 # Error rate < 0.5%
export PTAAS_TARGET_FAIRNESS=0.7     # Fairness index ≥ 0.7
```

### Database Backend Selection

#### PostgreSQL (Recommended for Production)
```bash
export PTAAS_DB=postgres
export POSTGRES_POOL_SIZE=20         # Connection pool size
export POSTGRES_MAX_OVERFLOW=30      # Pool overflow limit
export DATABASE_URL="postgresql://user:pass@localhost:5432/ptaas"
```

#### SQLite (Development/Testing)
```bash
export PTAAS_DB=sqlite
# Automatic file-based storage, no additional configuration needed
```

### Task Execution Timeouts

```bash
# Module execution timeouts by complexity
export PTAAS_TIMEOUT_FAST=30         # Quick scans (70% of workload)
export PTAAS_TIMEOUT_MEDIUM=120      # Medium complexity (25% of workload)
export PTAAS_TIMEOUT_SLOW=300        # Deep analysis (5% of workload)
```

## EPYC 7002 Architecture-Specific Optimizations

### NUMA Configuration

AMD EPYC 7002 processors have multiple NUMA nodes. Optimize memory allocation:

```bash
# Enable NUMA-aware memory allocation
echo "numa_policy=interleave" >> /etc/sysctl.d/99-ptaas.conf

# Set memory allocation strategy
echo "vm.zone_reclaim_mode=0" >> /etc/sysctl.d/99-ptaas.conf
echo "vm.swappiness=10" >> /etc/sysctl.d/99-ptaas.conf

# Apply settings
sysctl -p /etc/sysctl.d/99-ptaas.conf
```

### CPU Affinity and Scheduling

```bash
# Optimize CPU scheduler for EPYC workloads
echo "kernel.sched_migration_cost_ns=500000" >> /etc/sysctl.d/99-ptaas.conf
echo "kernel.sched_nr_migrate=32" >> /etc/sysctl.d/99-ptaas.conf

# Enable transparent huge pages for large memory allocations
echo "vm.nr_hugepages=1024" >> /etc/sysctl.d/99-ptaas.conf
```

### Network Optimization

```bash
# Optimize network stack for high-throughput scanning
echo "net.core.rmem_max=16777216" >> /etc/sysctl.d/99-ptaas.conf
echo "net.core.wmem_max=16777216" >> /etc/sysctl.d/99-ptaas.conf
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.d/99-ptaas.conf
```

## Performance Profiles by System Size

### 8-Core EPYC 7002 (Development)
```bash
# Small development environment
PTAAS_WORKERS=6
PTAAS_CPU_POOL=3
PTAAS_IO_CONCURRENCY=32
PTAAS_MEMORY_POOL_MB=2048
NATS_MAX_ACK_PENDING=1000
TARGET_RATE=1000  # 1K messages/second
```

### 16-Core EPYC 7002 (Small Production)
```bash
# Small production deployment
PTAAS_WORKERS=12
PTAAS_CPU_POOL=6
PTAAS_IO_CONCURRENCY=64
PTAAS_MEMORY_POOL_MB=4096
NATS_MAX_ACK_PENDING=5000
TARGET_RATE=5000  # 5K messages/second
```

### 32-Core EPYC 7002 (Medium Production) - RECOMMENDED
```bash
# Balanced production configuration
PTAAS_WORKERS=28
PTAAS_CPU_POOL=16
PTAAS_IO_CONCURRENCY=128
PTAAS_MEMORY_POOL_MB=8192
NATS_MAX_ACK_PENDING=10000
TARGET_RATE=10000  # 10K messages/second
```

### 64-Core EPYC 7002 (Large Production)
```bash
# High-scale production deployment
PTAAS_WORKERS=48
PTAAS_CPU_POOL=24
PTAAS_IO_CONCURRENCY=256
PTAAS_MEMORY_POOL_MB=16384
NATS_MAX_ACK_PENDING=20000
TARGET_RATE=20000  # 20K messages/second
```

## Performance Testing and Validation

### Load Testing Scenarios

#### Standard Load Test
```bash
# 10 tenants × 8 concurrent jobs = 80 in-flight jobs
make ptaas-perf K6_VUS=80 K6_DURATION=10m
```

#### Spike Test
```bash
# Test EPYC burst capacity (2x normal load)
make ptaas-perf K6_VUS=160 K6_DURATION=5m
```

#### Fairness Validation
```bash
# Validate tenant fairness under load
make ptaas-perf K6_VUS=100 K6_DURATION=15m FAIRNESS_TEST=true
```

### NATS JetStream Load Testing

```bash
# High-throughput NATS testing
./tests/perf/nats/jetstream_load.sh

# Custom NATS load test
PUBLISHER_COUNT=16 CONSUMER_COUNT=8 TARGET_RATE=15000 \
./tests/perf/nats/jetstream_load.sh
```

### Continuous Performance Monitoring

```bash
# Start continuous monitoring (production)
make ptaas-perf-monitor

# Set performance baseline
make ptaas-perf-baseline

# Compare with baseline
make ptaas-perf-compare
```

## Monitoring and Observability

### Prometheus Metrics

Key metrics to monitor for EPYC performance:

```promql
# P95 latency monitoring
histogram_quantile(0.95, rate(ptaas_job_latency_ms_bucket[5m]))

# Fairness index tracking
ptaas_fairness_index

# EPYC CPU utilization by core group
epyc_cpu_utilization_percent

# NUMA memory usage
epyc_numa_memory_usage_mb

# Worker pool utilization
ptaas_worker_utilization
```

### Grafana Dashboard

Import the EPYC-optimized dashboard:

```bash
# Import PTaaS Scale Dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @infra/monitoring/grafana/dashboards/ptaas-scale.json
```

Access at: http://localhost:3000/d/ptaas-scale-epyc

### Alert Rules

Critical alerts for EPYC production:

```yaml
# P95 latency SLA violation
- alert: PTaaS_P95_Latency_High
  expr: histogram_quantile(0.95, rate(ptaas_job_latency_ms_bucket[5m])) > 2000
  for: 2m
  annotations:
    summary: "PTaaS P95 latency exceeds 2s target"

# Fairness index below target
- alert: PTaaS_Fairness_Low
  expr: ptaas_fairness_index < 0.7
  for: 5m
  annotations:
    summary: "PTaaS fairness index below 0.7 target"

# EPYC CPU saturation
- alert: EPYC_CPU_Saturation
  expr: avg(epyc_cpu_utilization_percent) > 90
  for: 1m
  annotations:
    summary: "EPYC CPU utilization above 90%"
```

## Troubleshooting Performance Issues

### Common Issues and Solutions

#### High P95 Latency (>2s)

1. **Check worker pool utilization**:
   ```bash
   # Increase worker count if utilization > 80%
   export PTAAS_WORKERS=$((PTAAS_WORKERS + 4))
   ```

2. **Optimize NUMA placement**:
   ```bash
   # Check NUMA topology
   numactl --hardware

   # Pin processes to NUMA nodes
   numactl --cpunodebind=0 --membind=0 python src/orchestrator/main.py
   ```

3. **Database connection pooling**:
   ```bash
   # Increase PostgreSQL connections
   export POSTGRES_POOL_SIZE=30
   export POSTGRES_MAX_OVERFLOW=50
   ```

#### Low Fairness Index (<0.7)

1. **Adjust WFQ weights**:
   ```bash
   # Increase fairness window
   export G8_FAIRNESS_WINDOW_SEC=120

   # Reduce priority weight differences
   export G8_WEIGHT_HIGH=2.0
   export G8_WEIGHT_MEDIUM=1.5
   export G8_WEIGHT_LOW=1.0
   ```

2. **Check tenant distribution**:
   ```bash
   # Monitor per-tenant metrics
   curl http://localhost:9090/api/v1/query?query=ptaas_tenant_throughput
   ```

#### High Error Rate (>0.5%)

1. **Check timeout settings**:
   ```bash
   # Increase timeouts for slow modules
   export PTAAS_TIMEOUT_MEDIUM=180
   export PTAAS_TIMEOUT_SLOW=450
   ```

2. **NATS consumer lag**:
   ```bash
   # Increase consumer capacity
   export NATS_MAX_ACK_PENDING=15000
   export NATS_MAX_INFLIGHT=512
   ```

#### NATS Consumer Lag (>500 messages)

1. **Scale consumers**:
   ```bash
   # Increase consumer count
   export CONSUMER_COUNT=12
   ```

2. **Optimize acknowledgment**:
   ```bash
   # Reduce ack wait time
   export NATS_ACK_WAIT_MS=20000
   ```

### Performance Debugging Tools

#### System-level monitoring
```bash
# Monitor EPYC core utilization
watch -n 1 "grep 'cpu' /proc/stat"

# Memory bandwidth monitoring
perf stat -e instructions,cycles,cache-references,cache-misses python src/orchestrator/main.py

# NUMA memory usage
numastat -m

# Network throughput
iftop -i eth0
```

#### Application profiling
```bash
# Python profiling
python -m cProfile -o profile.stats src/orchestrator/main.py

# Memory profiling
python -m memory_profiler src/orchestrator/main.py

# Async profiling
python -m aiomonitor --host localhost --port 50101
```

## Advanced Tuning

### Custom Worker Distribution

Override default worker allocation:

```python
# Custom worker distribution in config.py
def get_custom_worker_distribution(total_workers: int) -> Dict[str, int]:
    return {
        "fast_scanners": int(total_workers * 0.5),     # 50% for network scans
        "medium_scanners": int(total_workers * 0.3),   # 30% for vulnerability scans
        "slow_scanners": int(total_workers * 0.1),     # 10% for deep analysis
        "orchestration": int(total_workers * 0.1),     # 10% for coordination
    }
```

### Dynamic Scaling

Implement auto-scaling based on load:

```python
# Auto-scaling logic
async def auto_scale_workers():
    current_load = get_current_load()
    target_utilization = 0.75

    if current_load > target_utilization:
        scale_up_workers()
    elif current_load < target_utilization * 0.5:
        scale_down_workers()
```

### Batch Processing Optimization

Optimize for high-throughput batch processing:

```bash
# Batch processing settings
export PTAAS_BATCH_SIZE=100
export PTAAS_BATCH_TIMEOUT_MS=5000
export PTAAS_PREFETCH_COUNT=1000
```

## Production Deployment Checklist

### Pre-deployment Validation

- [ ] Run full performance test: `make ptaas-perf`
- [ ] Verify P95 latency < 2000ms
- [ ] Confirm error rate < 0.5%
- [ ] Validate fairness index ≥ 0.7
- [ ] Test database backend (PostgreSQL recommended)
- [ ] Verify NATS consumer lag < 500 messages
- [ ] Check EPYC CPU utilization < 80% under load
- [ ] Validate memory usage within limits

### System Configuration

- [ ] Apply EPYC-specific kernel parameters
- [ ] Configure NUMA memory policy
- [ ] Set up transparent huge pages
- [ ] Optimize network stack settings
- [ ] Configure CPU governor to performance mode
- [ ] Set process priority and nice values

### Monitoring Setup

- [ ] Deploy Prometheus with EPYC metrics
- [ ] Import Grafana PTaaS Scale dashboard
- [ ] Configure alerting rules
- [ ] Set up performance baseline
- [ ] Enable continuous monitoring

### Security Considerations

- [ ] Validate performance under security scanning load
- [ ] Test with realistic penetration testing workloads
- [ ] Verify tenant isolation under high load
- [ ] Check for performance-related security issues

## References

### EPYC 7002 Resources
- [AMD EPYC 7002 Series Processor Architecture](https://www.amd.com/en/processors/epyc-7002-series)
- [NUMA Optimization Guide](https://docs.kernel.org/admin-guide/numastat.html)
- [Linux Performance Tuning for EPYC](https://developer.amd.com/resources/epyc-tuning-guides/)

### PTaaS Documentation
- [PTaaS API Reference](../api/ptaas.md)
- [NATS JetStream Configuration](../infrastructure/nats.md)
- [Monitoring and Observability](../monitoring/README.md)

### Performance Testing Tools
- [K6 Documentation](https://k6.io/docs/)
- [NATS Benchmarking](https://docs.nats.io/nats-concepts/subject_mapping)
- [Prometheus Monitoring](https://prometheus.io/docs/)

## Support

For performance-related issues:

1. **GitHub Issues**: [PTaaS Performance Issues](https://github.com/org/xorb/issues?q=is%3Aissue+label%3Aperformance)
2. **Performance Reports**: Use `make ptaas-perf-report` to generate detailed reports
3. **Community**: [XORB Performance Discussions](https://github.com/org/xorb/discussions/categories/performance)

---

**Document Version**: 2025.08-rc2
**Last Updated**: August 14, 2025
**Target Architecture**: AMD EPYC 7002 Series
**Performance Target**: P95 < 2s, Error Rate < 0.5%, Fairness ≥ 0.7
