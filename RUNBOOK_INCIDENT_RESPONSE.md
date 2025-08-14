# XORB Incident Response Runbook

**Version**: v2025.08-rc1
**Last Updated**: August 14, 2025
**Owner**: XORB Operations Team

---

## 📋 **Quick Reference**

### **Emergency Contacts**
- **On-Call Engineer**: `@xorb-oncall` (Slack)
- **Security Team**: `@xorb-security` (Slack)
- **Platform Team**: `@xorb-platform` (Slack)
- **Escalation**: `@xorb-leadership` (Slack)

### **Critical Systems**
- **API Health**: http://localhost:8000/api/v1/health
- **Grafana Dashboard**: http://localhost:3010/d/xorb-overview
- **AlertManager**: http://localhost:9093
- **NATS Monitoring**: http://localhost:8222

---

## 🚨 **Incident Classification**

| Severity | Description | Response Time | Escalation |
|----------|-------------|---------------|------------|
| **P0 - Critical** | Complete service outage, data loss, security breach | 5 minutes | Immediate |
| **P1 - High** | Major functionality impacted, SLO breach | 15 minutes | Within 30 min |
| **P2 - Medium** | Partial functionality impacted | 1 hour | Within 2 hours |
| **P3 - Low** | Minor issues, monitoring alerts | 4 hours | Next business day |

---

## 🔧 **Incident Response Procedures**

### **1. Evidence Verification Failures**

**Symptoms:**
- Evidence verification API returning 5xx errors
- `evidence_verification_failure_rate` > 1%
- Ed25519 signature validation failures
- Chain of custody breaks

**Immediate Actions:**
```bash
# Check evidence service health
curl -s http://localhost:8000/api/v1/provable-evidence/health | jq

# Review evidence verification logs
kubectl logs -l app=xorb-api --tail=100 | grep -i "evidence.*error"

# Check Ed25519 key status
kubectl get secret xorb-evidence-keys -o yaml

# Verify certificate chain
openssl verify -CAfile /path/to/ca.crt /path/to/evidence.crt
```

**Diagnosis Steps:**
1. **Key Rotation Issues**:
   ```bash
   # Check key rotation status
   kubectl describe secret xorb-evidence-keys

   # Verify key permissions
   kubectl auth can-i get secrets --as=system:serviceaccount:xorb:evidence-service
   ```

2. **Timestamp Authority Problems**:
   ```bash
   # Test RFC 3161 timestamp service
   curl -s http://timestamp.authority.com/rfc3161 \
     -H "Content-Type: application/timestamp-query" \
     --data-binary @test.tsq

   # Check timestamp authority certificates
   openssl ts -verify -in evidence.tsr -CAfile tsa-ca.crt
   ```

3. **Chain of Custody Corruption**:
   ```bash
   # Verify evidence chain integrity
   python3 tools/scripts/verify_evidence_chain.py --evidence-id <ID>

   # Check for merkle tree corruption
   python3 src/api/app/services/g7_provable_evidence_service.py --verify-merkle
   ```

**Resolution:**
```bash
# Emergency key rotation
kubectl create secret generic xorb-evidence-keys-new \
  --from-file=private.pem \
  --from-file=public.pem

# Update evidence service configuration
kubectl patch deployment xorb-evidence-service \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"evidence-service","env":[{"name":"KEY_SECRET","value":"xorb-evidence-keys-new"}]}]}}}}'

# Restart evidence verification pods
kubectl rollout restart deployment/xorb-evidence-service

# Verify resolution
make g7-evidence-test
```

**Escalation Triggers:**
- Evidence verification failures > 5%
- Chain of custody breaks detected
- Key rotation failures
- Unable to restore service within 30 minutes

---

### **2. Tenant Isolation Breaches**

**Symptoms:**
- Cross-tenant data access detected
- `tenant_isolation_violation_count` > 0
- Unauthorized subject access in NATS
- Row-level security bypass attempts

**Immediate Actions:**
```bash
# Emergency tenant isolation check
python3 tools/scripts/g6_tenant_isolation_validator.py --emergency-scan

# Check NATS account isolation
nats account info --account tenant-a
nats account info --account tenant-b

# Review tenant access logs
kubectl logs -l app=xorb-api --tail=500 | grep -E "(tenant|isolation|unauthorized)"

# Check PostgreSQL row-level security
psql -c "SELECT * FROM pg_policies WHERE schemaname = 'public';"
```

**Diagnosis Steps:**
1. **NATS Account Breach**:
   ```bash
   # Check NATS account permissions
   nats stream subjects xorb-tenant-a
   nats stream subjects xorb-tenant-b

   # Verify account isolation
   nats pub xorb.tenant-b.scan.result "test" --account tenant-a-creds.json
   # Should fail with permission denied

   # Check NATS server logs
   kubectl logs -l app=nats-server | grep -i "permission\|denied\|unauthorized"
   ```

2. **Database RLS Bypass**:
   ```bash
   # Check RLS policies
   psql -c "SELECT schemaname, tablename, policyname, cmd, permissive FROM pg_policies;"

   # Verify tenant context
   psql -c "SELECT current_setting('app.tenant_id', true);"

   # Test cross-tenant query
   psql -c "SET app.tenant_id = 'tenant-a'; SELECT * FROM tenant_data WHERE tenant_id = 'tenant-b';"
   # Should return no rows
   ```

3. **API Authorization Bypass**:
   ```bash
   # Check JWT token validation
   curl -H "Authorization: Bearer <tenant-a-token>" \
        http://localhost:8000/api/v1/tenant-b/data
   # Should return 403 Forbidden

   # Review middleware logs
   kubectl logs -l app=xorb-api | grep -E "(auth|tenant|middleware)"
   ```

**Resolution:**
```bash
# Emergency tenant isolation enforcement
kubectl apply -f infra/security/emergency-network-policies.yaml

# Restart tenant context middleware
kubectl rollout restart deployment/xorb-api

# Force NATS account refresh
kubectl delete pod -l app=nats-server
kubectl wait --for=condition=Ready pod -l app=nats-server

# Verify isolation restoration
make g6-tenant-validate

# Audit potentially affected data
python3 tools/scripts/tenant_breach_audit.py --start-time "2025-08-14T12:00:00Z"
```

**Escalation Triggers:**
- Confirmed cross-tenant data access
- RLS policy bypass detected
- NATS account isolation failure
- Unable to restore isolation within 15 minutes

---

### **3. Quota Enforcement Anomalies**

**Symptoms:**
- `quota_enforcement_failure_rate` > 0.1%
- Tenants exceeding configured limits
- Resource exhaustion in control plane
- WFQ scheduler imbalance

**Immediate Actions:**
```bash
# Check quota enforcement status
curl -s http://localhost:8000/api/v1/control-plane/quotas | jq

# Review current resource usage
kubectl top pods -l tier=tenant-workloads

# Check WFQ scheduler metrics
curl -s http://localhost:9092/api/v1/query?query=wfq_scheduler_queue_depth | jq

# Verify Redis quota tracking
redis-cli --scan --pattern "quota:*" | head -20
```

**Diagnosis Steps:**
1. **WFQ Scheduler Issues**:
   ```bash
   # Check scheduler health
   python3 src/api/app/services/g8_control_plane_service.py --health-check

   # Review scheduler logs
   kubectl logs -l app=xorb-scheduler --tail=100

   # Check queue imbalances
   curl -s "http://localhost:9092/api/v1/query?query=wfq_scheduler_fairness_index" | jq
   ```

2. **Redis Quota State Corruption**:
   ```bash
   # Check Redis connectivity
   redis-cli ping

   # Verify quota counters
   redis-cli hgetall "quota:tenant-high:scan_requests"
   redis-cli hgetall "quota:tenant-medium:scan_requests"

   # Check for counter drift
   python3 tools/scripts/quota_state_audit.py
   ```

3. **Control Plane Overload**:
   ```bash
   # Check control plane CPU/memory
   kubectl top pods -l app=xorb-control-plane

   # Review admission controller logs
   kubectl logs -l app=admission-controller | grep -E "(reject|throttle|limit)"

   # Check backlog depth
   curl -s "http://localhost:9092/api/v1/query?query=control_plane_backlog_depth" | jq
   ```

**Resolution:**
```bash
# Emergency quota enforcement
kubectl apply -f infra/k8s/emergency-resource-quotas.yaml

# Scale control plane components
kubectl scale deployment xorb-control-plane --replicas=3
kubectl scale deployment xorb-scheduler --replicas=2

# Reset quota counters if corrupted
redis-cli flushdb 1  # Quota database only
python3 tools/scripts/quota_state_rebuild.py

# Restart quota enforcement
kubectl rollout restart deployment/xorb-control-plane

# Verify quota enforcement
make g8-control-plane-validate
```

**Escalation Triggers:**
- Control plane unavailable > 5 minutes
- Massive quota violations (>10x limits)
- Redis quota state corruption
- Scheduler fairness index < 0.7

---

### **4. Replay Traffic Impacting Live Workloads**

**Symptoms:**
- Live workload latency increased > 2x
- `replay_traffic_impact_ratio` > 0.1
- Production SLO breaches during replay
- Resource contention alerts

**Immediate Actions:**
```bash
# Check replay traffic status
curl -s http://localhost:8000/api/v1/replay/status | jq

# Review live workload impact
curl -s "http://localhost:9092/api/v1/query?query=live_workload_p99_latency" | jq

# Check resource contention
kubectl top nodes
kubectl top pods -l workload-type=live

# Verify replay isolation
python3 tools/scripts/replay_isolation_check.py
```

**Diagnosis Steps:**
1. **Resource Starvation**:
   ```bash
   # Check CPU/memory limits
   kubectl describe limitrange replay-limits
   kubectl describe resourcequota replay-quota

   # Review node resource allocation
   kubectl describe nodes | grep -A 5 "Allocated resources"

   # Check for CPU throttling
   kubectl top pods -l workload-type=replay --containers
   ```

2. **Network Bandwidth Competition**:
   ```bash
   # Check network policies
   kubectl get networkpolicies -o wide

   # Verify QoS classes
   kubectl get pods -o custom-columns=NAME:.metadata.name,QOS:.status.qosClass

   # Check bandwidth utilization
   curl -s "http://localhost:9092/api/v1/query?query=node_network_transmit_bytes_total" | jq
   ```

3. **Storage I/O Contention**:
   ```bash
   # Check disk I/O metrics
   curl -s "http://localhost:9092/api/v1/query?query=node_disk_io_time_seconds_total" | jq

   # Review storage class priorities
   kubectl get storageclasses -o wide

   # Check persistent volume usage
   kubectl get pv -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,CLAIM:.spec.claimRef.name
   ```

**Resolution:**
```bash
# Emergency replay traffic throttling
kubectl patch deployment replay-traffic-generator \
  -p '{"spec":{"replicas":1,"template":{"spec":{"containers":[{"name":"replay","resources":{"limits":{"cpu":"100m","memory":"256Mi"}}}]}}}}'

# Implement priority classes
kubectl apply -f - <<EOF
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: live-workload-priority
value: 1000
globalDefault: false
description: "High priority for live workloads"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: replay-traffic-priority
value: 100
globalDefault: false
description: "Low priority for replay traffic"
EOF

# Apply network throttling
kubectl apply -f infra/k8s/replay-network-policies.yaml

# Restart affected workloads with priorities
kubectl patch deployment xorb-api \
  -p '{"spec":{"template":{"spec":{"priorityClassName":"live-workload-priority"}}}}'

# Verify impact resolution
make replay-validate
```

**Escalation Triggers:**
- Live SLO breach > 5 minutes
- Production workload failure
- Unable to isolate replay traffic
- Customer-facing service degradation

---

## 📊 **Monitoring & Metrics**

### **Key SLIs to Monitor:**
```yaml
# Evidence System
evidence_verification_success_rate: >99.9%
evidence_creation_p99_latency: <2s
merkle_rollup_completion_rate: >99%

# Tenant Isolation
tenant_isolation_violation_count: 0
cross_tenant_access_attempts: <1/day
rls_policy_bypass_count: 0

# Quota Enforcement
quota_enforcement_accuracy: >99.5%
wfq_scheduler_fairness_index: >0.8
control_plane_availability: >99.9%

# Replay Safety
replay_traffic_impact_ratio: <0.05
live_workload_slo_compliance: >99.5%
resource_contention_incidents: <1/week
```

### **Alert Thresholds:**
```yaml
# Critical (P0)
- evidence_verification_success_rate < 95%
- tenant_isolation_violation_count > 0
- quota_enforcement_accuracy < 90%
- live_workload_slo_compliance < 95%

# High (P1)
- evidence_creation_p99_latency > 5s
- wfq_scheduler_fairness_index < 0.7
- replay_traffic_impact_ratio > 0.1

# Medium (P2)
- merkle_rollup_completion_rate < 99%
- cross_tenant_access_attempts > 5/hour
- control_plane_backlog_depth > 1000
```

---

## 🔄 **Post-Incident Actions**

### **Immediate (0-2 hours):**
1. **Incident Documentation**:
   ```bash
   # Create incident report
   cp templates/incident-report.md incidents/$(date +%Y%m%d-%H%M)-incident.md

   # Collect logs and metrics
   kubectl logs -l app=xorb-api --since=2h > incident-logs.txt
   curl -s "http://localhost:9092/api/v1/query_range?query=up&start=$(date -d '2 hours ago' +%s)&end=$(date +%s)&step=60" > incident-metrics.json
   ```

2. **Service Restoration Verification**:
   ```bash
   # Run comprehensive health check
   make doctor
   make security-scan
   ./tools/xorbctl/xorbctl status

   # Verify SLO compliance
   python3 tools/scripts/slo_compliance_check.py --since "2 hours ago"
   ```

### **Short Term (2-24 hours):**
1. **Root Cause Analysis**:
   - Timeline reconstruction
   - Contributing factor analysis
   - Impact assessment
   - Customer communication

2. **Preventive Measures**:
   - Monitoring gap identification
   - Alert tuning
   - Runbook updates
   - Automation improvements

### **Long Term (1-4 weeks):**
1. **System Improvements**:
   - Architecture changes
   - Resilience enhancements
   - Testing improvements
   - Documentation updates

2. **Team Learning**:
   - Post-mortem review
   - Training updates
   - Process improvements
   - Tool enhancements

---

## 📞 **Communication Templates**

### **Initial Alert (0-5 minutes):**
```
🚨 INCIDENT: [P0/P1/P2] - [Brief Description]

STATUS: Investigating
IMPACT: [Customer/Service Impact]
ETA: Investigating
OWNER: [On-call Engineer]

Updates: #incident-response
```

### **Status Update (Every 30 minutes for P0/P1):**
```
🔄 UPDATE: [Incident Brief]

PROGRESS: [Current actions/findings]
ETA: [Updated ETA or "Investigating"]
NEXT: [Next steps]

Full details: [Link to incident doc]
```

### **Resolution:**
```
✅ RESOLVED: [Incident Brief]

DURATION: [Total time]
ROOT CAUSE: [Brief summary]
PREVENTION: [Actions taken]

Post-mortem: [Link to full analysis]
```

---

## 🛠️ **Tools & Resources**

### **Diagnostic Commands:**
```bash
# System health overview
./tools/xorbctl/xorbctl status

# Comprehensive validation
make doctor && make security-scan

# Service-specific health checks
make g6-tenant-validate    # Tenant isolation
make g7-evidence-test      # Evidence system
make g8-control-plane-validate  # Quota enforcement
make replay-validate       # Replay safety
```

### **Emergency Contacts:**
- **Slack Channels**: `#xorb-incidents`, `#xorb-oncall`, `#xorb-platform`
- **Escalation Matrix**: See `docs/escalation-matrix.md`
- **Vendor Contacts**: See `docs/vendor-contacts.md`

### **Additional Resources:**
- **Architecture Diagrams**: `docs/architecture/`
- **API Documentation**: http://localhost:8000/docs
- **Monitoring Dashboards**: http://localhost:3010
- **Log Aggregation**: http://localhost:3100 (Loki)

---

**Last Updated**: August 14, 2025
**Version**: v2025.08-rc1
**Next Review**: September 14, 2025
