# XORB Rollback Runbook

**Version**: v2025.08-rc1
**Last Updated**: August 14, 2025
**Target Release**: v2025.08-rc1 → v2025.07-stable

---

## 🚨 **Emergency Rollback - Quick Commands**

### **Critical Path (< 5 minutes):**
```bash
# 1. Emergency service rollback
kubectl set image deployment/xorb-api xorb-api=xorb/api:v2025.07-stable
kubectl set image deployment/xorb-orchestrator xorb-orchestrator=xorb/orchestrator:v2025.07-stable
kubectl set image deployment/xorb-control-plane xorb-control-plane=xorb/control-plane:v2025.07-stable

# 2. Database schema rollback (if needed)
alembic downgrade v2025.07-stable

# 3. Configuration rollback
kubectl apply -f config/v2025.07-stable/
```

### **Git Repository Rollback:**
```bash
# Rollback to previous stable tag
git checkout v2025.07-stable
git tag v2025.08-rollback
git push origin v2025.08-rollback
```

---

## 📋 **Rollback Decision Matrix**

| Issue Severity | Rollback Trigger | Decision Time | Approval Required |
|----------------|------------------|---------------|-------------------|
| **P0 - Critical** | Service unavailable >5min, Data corruption, Security breach | Immediate | On-call engineer |
| **P1 - High** | SLO breach >30min, Major functionality broken | 15 minutes | Team lead |
| **P2 - Medium** | Performance degradation >2x, Minor functionality issues | 2 hours | Engineering manager |
| **P3 - Low** | Monitoring issues, Non-critical bugs | 24 hours | Product owner |

---

## 🔄 **Complete Rollback Procedures**

### **Phase 1: Pre-Rollback Assessment (2-5 minutes)**

```bash
# 1. Capture current system state
kubectl get deployments -o wide > rollback-pre-state.txt
kubectl get configmaps -o yaml > rollback-configs-backup.yaml
kubectl get secrets -o yaml > rollback-secrets-backup.yaml

# 2. Check data integrity
python3 tools/scripts/data_integrity_check.py --quick-scan

# 3. Verify rollback target availability
docker pull xorb/api:v2025.07-stable
docker pull xorb/orchestrator:v2025.07-stable
docker pull xorb/control-plane:v2025.07-stable

# 4. Notify stakeholders
curl -X POST https://hooks.slack.com/services/ROLLBACK_WEBHOOK \
  -H 'Content-type: application/json' \
  -d '{"text":"🚨 ROLLBACK INITIATED: v2025.08-rc1 → v2025.07-stable"}'
```

### **Phase 2: Service Rollback (5-10 minutes)**

#### **2.1 Application Services:**
```bash
# API Service rollback
kubectl set image deployment/xorb-api \
  xorb-api=xorb/api:v2025.07-stable \
  --record

kubectl rollout status deployment/xorb-api --timeout=300s

# Orchestrator rollback
kubectl set image deployment/xorb-orchestrator \
  xorb-orchestrator=xorb/orchestrator:v2025.07-stable \
  --record

kubectl rollout status deployment/xorb-orchestrator --timeout=300s

# Control Plane rollback
kubectl set image deployment/xorb-control-plane \
  xorb-control-plane=xorb/control-plane:v2025.07-stable \
  --record

kubectl rollout status deployment/xorb-control-plane --timeout=300s

# Worker Services rollback
kubectl set image deployment/xorb-worker \
  xorb-worker=xorb/worker:v2025.07-stable \
  --record

kubectl rollout status deployment/xorb-worker --timeout=300s
```

#### **2.2 Configuration Rollback:**
```bash
# Apply previous stable configurations
kubectl apply -f infra/k8s/v2025.07-stable/configmaps/
kubectl apply -f infra/k8s/v2025.07-stable/secrets/
kubectl apply -f infra/k8s/v2025.07-stable/network-policies/

# Update environment-specific configs
kubectl patch configmap xorb-config \
  --patch-file infra/k8s/v2025.07-stable/config-patch.yaml

# Rollback monitoring configuration
kubectl apply -f infra/monitoring/v2025.07-stable/prometheus.yml
kubectl apply -f infra/monitoring/v2025.07-stable/alertmanager.yml
```

### **Phase 3: Database Rollback (10-15 minutes)**

#### **3.1 Schema Migration Rollback:**
```bash
# Backup current database state
pg_dump -h $DB_HOST -U $DB_USER -d xorb_production > rollback-db-backup-$(date +%Y%m%d-%H%M%S).sql

# Check migration status
alembic current
alembic history --verbose

# Rollback to stable schema version
alembic downgrade 1a5be11  # v2025.07-stable migration ID

# Verify schema rollback
psql -h $DB_HOST -U $DB_USER -d xorb_production -c "\dt"
psql -h $DB_HOST -U $DB_USER -d xorb_production -c "SELECT version_num FROM alembic_version;"
```

#### **3.2 Data Migration Rollback (if applicable):**
```bash
# Check for data migrations that need rollback
python3 tools/migrations/check_data_migrations.py --version v2025.08-rc1

# Rollback tenant isolation data (G6)
python3 tools/migrations/rollback_tenant_data.py --to-version v2025.07-stable

# Rollback evidence system data (G7)
python3 tools/migrations/rollback_evidence_data.py --to-version v2025.07-stable

# Rollback control plane data (G8)
python3 tools/migrations/rollback_control_plane_data.py --to-version v2025.07-stable

# Verify data consistency
python3 tools/scripts/data_integrity_check.py --full-scan
```

### **Phase 4: Infrastructure Rollback (15-20 minutes)**

#### **4.1 NATS Configuration Rollback:**
```bash
# Rollback NATS account configuration
kubectl apply -f infra/nats/v2025.07-stable/accounts/

# Rollback stream definitions
nats stream rm xorb-tenant-streams --force
kubectl apply -f infra/nats/v2025.07-stable/streams.yaml

# Restart NATS cluster
kubectl rollout restart statefulset/nats-cluster

# Verify NATS rollback
nats account info
nats stream ls
```

#### **4.2 Redis Configuration Rollback:**
```bash
# Backup current Redis data
redis-cli --rdb redis-backup-$(date +%Y%m%d-%H%M%S).rdb

# Clear quota enforcement data (if incompatible)
redis-cli flushdb 1  # Quota database
redis-cli flushdb 2  # Control plane cache

# Rollback Redis configuration
kubectl apply -f infra/redis/v2025.07-stable/redis-config.yaml
kubectl rollout restart deployment/redis

# Verify Redis rollback
redis-cli ping
redis-cli config get "*"
```

#### **4.3 Monitoring Stack Rollback:**
```bash
# Rollback Grafana dashboards
kubectl apply -f infra/monitoring/v2025.07-stable/grafana/

# Rollback Prometheus configuration
kubectl apply -f infra/monitoring/v2025.07-stable/prometheus.yml
kubectl rollout restart deployment/prometheus

# Rollback alert rules
kubectl apply -f infra/monitoring/v2025.07-stable/alert-rules.yml
kubectl rollout restart deployment/alertmanager

# Verify monitoring rollback
curl -s http://prometheus:9090/-/healthy
curl -s http://grafana:3000/api/health
```

### **Phase 5: Verification & Validation (20-30 minutes)**

#### **5.1 Service Health Verification:**
```bash
# Comprehensive health check
./tools/xorbctl/xorbctl status

# API endpoints verification
curl -s http://localhost:8000/api/v1/health | jq
curl -s http://localhost:8000/api/v1/info | jq

# Core functionality tests
make test-fast
python3 tools/scripts/smoke_test.py --version v2025.07-stable

# Performance baseline check
python3 tools/scripts/performance_test.py --baseline v2025.07-stable
```

#### **5.2 Business Logic Verification:**
```bash
# Tenant isolation verification (should work on v2025.07)
python3 tools/scripts/tenant_isolation_test.py

# Evidence system verification (may be limited on v2025.07)
python3 tools/scripts/evidence_system_test.py --legacy-mode

# Quota system verification (may not exist on v2025.07)
python3 tools/scripts/quota_system_test.py --skip-if-missing

# PTaaS functionality verification
curl -X POST "http://localhost:8000/api/v1/ptaas/sessions" \
  -H "Content-Type: application/json" \
  -d '{"targets":[{"host":"scanme.nmap.org","ports":[80,443]}]}'
```

#### **5.3 Data Integrity Verification:**
```bash
# Full data integrity check
python3 tools/scripts/data_integrity_check.py --full-scan --report

# Cross-reference critical data
python3 tools/scripts/critical_data_verification.py

# Check for data loss
python3 tools/scripts/data_loss_assessment.py \
  --before-rollback rollback-db-backup-*.sql \
  --after-rollback current

# Audit log verification
python3 tools/scripts/audit_log_integrity.py --since "1 hour ago"
```

---

## 🔧 **Selective Rollback Procedures**

### **API-Only Rollback:**
```bash
# Rollback just the API service
kubectl set image deployment/xorb-api xorb-api=xorb/api:v2025.07-stable
kubectl rollout status deployment/xorb-api

# Update API-specific configuration
kubectl patch configmap xorb-api-config \
  --patch-file infra/k8s/v2025.07-stable/api-config-patch.yaml

# Verify API rollback
curl -s http://localhost:8000/api/v1/info | jq '.version'
```

### **Database-Only Rollback:**
```bash
# Rollback only schema changes
alembic downgrade -1

# Rollback specific data migration
python3 tools/migrations/rollback_specific_migration.py --migration-id abc123

# Verify database state
python3 tools/scripts/db_version_check.py
```

### **Configuration-Only Rollback:**
```bash
# Rollback configuration without service restart
kubectl apply -f infra/k8s/v2025.07-stable/configmaps/
kubectl apply -f infra/k8s/v2025.07-stable/secrets/

# Trigger configuration reload
kubectl annotate deployment xorb-api deployment.kubernetes.io/revision-
kubectl annotate deployment xorb-orchestrator deployment.kubernetes.io/revision-
```

---

## 🚫 **Rollback Limitations & Risks**

### **Known Limitations:**

1. **Evidence System (G7)**:
   ```bash
   # Evidence created in v2025.08-rc1 may not be compatible with v2025.07
   # Action: Preserve evidence data, disable creation in rolled-back version
   kubectl patch deployment xorb-api \
     -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","env":[{"name":"EVIDENCE_SYSTEM_ENABLED","value":"false"}]}]}}}}'
   ```

2. **Control Plane (G8)**:
   ```bash
   # Control plane features don't exist in v2025.07
   # Action: Remove control plane components, fallback to basic scheduling
   kubectl delete deployment xorb-control-plane
   kubectl delete service xorb-control-plane
   ```

3. **Enhanced Tenant Isolation (G6)**:
   ```bash
   # Advanced isolation features may not work in v2025.07
   # Action: Enable basic tenant isolation mode
   kubectl patch configmap xorb-config \
     -p '{"data":{"TENANT_ISOLATION_MODE":"basic"}}'
   ```

### **Data Loss Risks:**

1. **Evidence Data**: Ed25519-signed evidence may become inaccessible
2. **Quota Metrics**: Control plane usage data will be lost
3. **Advanced Audit Logs**: Enhanced audit features won't be available
4. **Tenant Metrics**: Advanced tenant metrics may be lost

### **Mitigation Strategies:**
```bash
# Preserve critical data before rollback
mkdir -p rollback-data-preservation/
kubectl cp xorb-api-pod:/app/evidence-storage ./rollback-data-preservation/evidence/
redis-cli --rdb ./rollback-data-preservation/redis-backup.rdb
pg_dump xorb_production > ./rollback-data-preservation/postgres-backup.sql
```

---

## 📊 **Post-Rollback Monitoring**

### **Critical Metrics to Watch:**
```bash
# Service availability
curl -s "http://prometheus:9090/api/v1/query?query=up{job='xorb-api'}"

# Error rates
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~'5..'}[5m])"

# Response times
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket[5m]))"

# Database connections
curl -s "http://prometheus:9090/api/v1/query?query=postgres_connections_active"
```

### **Automated Monitoring Checks:**
```bash
# Set up post-rollback monitoring
python3 tools/scripts/post_rollback_monitor.py \
  --duration "2 hours" \
  --baseline v2025.07-stable \
  --alert-threshold 10% \
  --report-interval 15min
```

---

## ⚡ **Emergency Procedures**

### **Rollback Failure Recovery:**
```bash
# If rollback fails, emergency recovery
kubectl apply -f infra/emergency/minimal-service.yaml

# Restore from last known good backup
pg_restore -h $DB_HOST -U $DB_USER -d xorb_production \
  /backups/xorb-production-last-known-good.sql

# Bring up minimal service
kubectl scale deployment xorb-api --replicas=1
kubectl scale deployment xorb-orchestrator --replicas=0
kubectl scale deployment xorb-control-plane --replicas=0
```

### **Cascade Failure Prevention:**
```bash
# Disable non-critical services during rollback
kubectl patch deployment xorb-worker \
  -p '{"spec":{"replicas":0}}'
kubectl patch deployment xorb-scheduler \
  -p '{"spec":{"replicas":0}}'

# Enable circuit breakers
kubectl patch configmap xorb-config \
  -p '{"data":{"CIRCUIT_BREAKER_ENABLED":"true"}}'
```

---

## 📝 **Rollback Checklist**

### **Pre-Rollback:**
- [ ] Rollback decision approved by required stakeholders
- [ ] System state captured (configs, data, metrics)
- [ ] Rollback target version verified available
- [ ] Stakeholders notified
- [ ] Maintenance window scheduled (if applicable)

### **During Rollback:**
- [ ] Services rolled back in correct order
- [ ] Database migrations rolled back
- [ ] Configuration rolled back
- [ ] Infrastructure components updated
- [ ] Health checks performed at each step

### **Post-Rollback:**
- [ ] All services healthy and responsive
- [ ] Critical functionality verified
- [ ] Data integrity confirmed
- [ ] Performance baselines met
- [ ] Monitoring restored
- [ ] Stakeholders notified of completion
- [ ] Post-rollback report generated

### **Follow-up:**
- [ ] Root cause analysis scheduled
- [ ] Rollback lessons learned documented
- [ ] Process improvements identified
- [ ] Next release planning updated

---

## 📞 **Communication Templates**

### **Rollback Initiation:**
```
🚨 ROLLBACK INITIATED
Release: v2025.08-rc1 → v2025.07-stable
Reason: [Brief description]
Impact: [Expected downtime/degradation]
ETA: 30 minutes
Updates: #rollback-status
```

### **Rollback Progress:**
```
🔄 ROLLBACK UPDATE
Progress: [Phase X of 5 complete]
Current: [Current step]
Issues: [Any problems encountered]
ETA: [Updated estimate]
```

### **Rollback Complete:**
```
✅ ROLLBACK COMPLETE
Duration: [Total time]
Status: All services healthy
Impact: Resolved
Next: Post-rollback monitoring for 2 hours
Report: [Link to detailed report]
```

---

## 🛠️ **Tools & Scripts**

### **Automated Rollback Script:**
```bash
#!/bin/bash
# tools/scripts/automated_rollback.sh
# Usage: ./automated_rollback.sh --from v2025.08-rc1 --to v2025.07-stable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLBACK_LOG="rollback-$(date +%Y%m%d-%H%M%S).log"

source "$SCRIPT_DIR/rollback_functions.sh"

log_action() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ROLLBACK_LOG"
}

# Main rollback execution
main() {
    log_action "Starting automated rollback from $FROM_VERSION to $TO_VERSION"

    # Phase 1: Pre-rollback
    capture_system_state
    verify_rollback_target
    notify_stakeholders "Rollback initiated"

    # Phase 2: Service rollback
    rollback_services

    # Phase 3: Database rollback
    rollback_database

    # Phase 4: Infrastructure rollback
    rollback_infrastructure

    # Phase 5: Verification
    verify_rollback_success

    log_action "Rollback completed successfully"
    notify_stakeholders "Rollback complete"
}

main "$@"
```

### **Rollback Validation Scripts:**
```bash
# Health check script
tools/scripts/rollback_health_check.py

# Data integrity verification
tools/scripts/rollback_data_integrity.py

# Performance baseline comparison
tools/scripts/rollback_performance_check.py
```

---

**Last Updated**: August 14, 2025
**Version**: v2025.08-rc1
**Next Review**: September 14, 2025
**Emergency Contact**: @xorb-oncall (Slack)
