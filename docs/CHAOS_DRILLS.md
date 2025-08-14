# XORB Chaos Engineering Drills

**Version**: v2025.08-rc1
**Last Updated**: August 14, 2025
**Target Release**: v2025.08-rc1 Post-Release Operational Readiness

---

## 🎯 **Chaos Engineering Objectives**

The XORB chaos engineering program validates system resilience through controlled failure injection, ensuring the platform maintains SLO compliance under adverse conditions and validates auto-remediation capabilities.

### **Core Principles:**
- **Blast Radius Control**: Limited scope with immediate rollback capability
- **SLO-Driven**: Validate SLO compliance during failure scenarios
- **Auto-Remediation**: Test autonomous recovery mechanisms
- **Production-Safe**: Non-destructive experiments with safety controls

---

## 🧪 **Chaos Experiment #1: NATS Node Kill**

### **Experiment Overview**
**Objective**: Validate NATS cluster resilience and message delivery guarantees during node failure
**Duration**: 10 minutes
**Blast Radius**: Single NATS node in 3-node cluster
**SLO Target**: Live P95 latency < 100ms maintained during failure

### **Hypothesis**
*"When we kill a single NATS node in a 3-node cluster, live message delivery P95 latency will remain under 100ms due to cluster failover, and the system will auto-heal within 2 minutes."*

### **Pre-Conditions**
```bash
# Verify NATS cluster health
nats server check
nats stream ls | grep -E "(live|replay)"
kubectl get pods -l app=nats-server -o wide

# Confirm 3-node cluster
kubectl get statefulset nats-cluster | grep "3/3"

# Baseline SLO measurement
curl -s "http://localhost:9092/api/v1/query?query=histogram_quantile(0.95,sum(rate(nats_request_duration_seconds_bucket{stream_class=\"live\"}[5m]))by(le))" | jq
```

### **Experiment Steps**

#### **Phase 1: Preparation (2 minutes)**
```bash
# Enable chaos testing metrics
kubectl label namespace default chaos-experiment=nats-node-kill
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: chaos-experiment-config
data:
  experiment_name: "nats-node-kill"
  experiment_id: "chaos-001"
  start_time: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  target_service: "nats-cluster"
  expected_duration: "10m"
EOF

# Start chaos metrics collection
kubectl apply -f infra/chaos/chaos-metrics-collector.yaml

# Record baseline metrics
echo "CHAOS_BASELINE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /tmp/chaos-experiment.env
```

#### **Phase 2: Failure Injection (30 seconds)**
```bash
# Identify target NATS node (non-leader preferred)
NATS_LEADER=$(nats server request '$SYS.REQ.SERVER.PING' | jq -r '.servers[] | select(.cluster.leader==true) | .name')
NATS_TARGET=$(kubectl get pods -l app=nats-server -o name | grep -v "$NATS_LEADER" | head -1)

echo "Target NATS node for termination: $NATS_TARGET"

# Kill target NATS node
kubectl delete pod $NATS_TARGET --grace-period=0 --force

# Record failure injection time
echo "CHAOS_FAILURE_INJECTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/chaos-experiment.env
```

#### **Phase 3: SLO Monitoring (8 minutes)**
```bash
# Continuous SLO monitoring during experiment
for i in {1..16}; do
  sleep 30

  # Check live stream P95 latency
  LIVE_P95=$(curl -s "http://localhost:9092/api/v1/query?query=histogram_quantile(0.95,sum(rate(nats_request_duration_seconds_bucket{stream_class=\"live\"}[5m]))by(le))" | jq -r '.data.result[0].value[1] // "0"')
  LIVE_P95_MS=$(echo "$LIVE_P95 * 1000" | bc)

  # Check cluster recovery status
  NATS_READY=$(kubectl get pods -l app=nats-server | grep -c "Running")

  echo "T+${i}m30s: Live P95=${LIVE_P95_MS}ms, NATS Ready=${NATS_READY}/3"

  # SLO violation check
  if (( $(echo "$LIVE_P95_MS > 100" | bc -l) )); then
    echo "⚠️  SLO VIOLATION: Live P95 latency ${LIVE_P95_MS}ms > 100ms at T+${i}m30s"
    echo "CHAOS_SLO_VIOLATION_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/chaos-experiment.env
  fi

  # Auto-remediation check
  if [[ "$NATS_READY" == "3" ]] && [[ ! -f /tmp/chaos-recovered ]]; then
    echo "✅ Auto-remediation complete: NATS cluster recovered at T+${i}m30s"
    echo "CHAOS_AUTO_RECOVERY_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/chaos-experiment.env
    touch /tmp/chaos-recovered
  fi
done
```

#### **Phase 4: Validation & Cleanup (1.5 minutes)**
```bash
# Final health verification
nats server check
kubectl get pods -l app=nats-server

# Verify no message loss
TOTAL_PUBLISHED=$(curl -s "http://localhost:9092/api/v1/query?query=sum(nats_jetstream_stream_messages{stream_class=\"live\"})" | jq -r '.data.result[0].value[1] // "0"')
TOTAL_DELIVERED=$(curl -s "http://localhost:9092/api/v1/query?query=sum(nats_jetstream_consumer_delivered{stream_class=\"live\"})" | jq -r '.data.result[0].value[1] // "0"')
MESSAGE_LOSS=$(echo "$TOTAL_PUBLISHED - $TOTAL_DELIVERED" | bc)

echo "Message Loss Assessment: ${MESSAGE_LOSS} messages (Published: ${TOTAL_PUBLISHED}, Delivered: ${TOTAL_DELIVERED})"

# Record experiment completion
echo "CHAOS_EXPERIMENT_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/chaos-experiment.env
echo "CHAOS_MESSAGE_LOSS=${MESSAGE_LOSS}" >> /tmp/chaos-experiment.env

# Cleanup chaos resources
kubectl delete configmap chaos-experiment-config
rm -f /tmp/chaos-recovered
```

### **Success Criteria**
- ✅ Live P95 latency remains < 100ms throughout experiment
- ✅ NATS cluster recovers automatically within 2 minutes
- ✅ Message loss ≤ 10 messages during failover
- ✅ No cascading failures to dependent services

### **Auto-Remediation Paths**
1. **Kubernetes Pod Recreation**: StatefulSet controller recreates failed pod
2. **NATS Cluster Rebalancing**: Remaining nodes handle traffic load
3. **Consumer Reconnection**: JetStream consumers reconnect to healthy nodes
4. **Circuit Breaker Activation**: Temporary throttling if latency spikes

### **Rollback Procedures**
```bash
# Emergency experiment termination
kubectl delete -f infra/chaos/chaos-metrics-collector.yaml
kubectl label namespace default chaos-experiment-

# Force NATS cluster restart if needed
kubectl rollout restart statefulset/nats-cluster
kubectl wait --for=condition=Ready pod -l app=nats-server --timeout=300s
```

---

## 🌊 **Chaos Experiment #2: Replay Storm Injection**

### **Experiment Overview**
**Objective**: Validate replay traffic isolation and WFQ scheduler fairness under extreme load
**Duration**: 15 minutes
**Blast Radius**: Replay traffic lane only
**SLO Target**: Live workloads maintain <100ms P95 latency despite 10x replay load

### **Hypothesis**
*"When we inject 10x normal replay traffic load, the WFQ scheduler and resource isolation will maintain live workload P95 latency under 100ms, proving replay traffic isolation effectiveness."*

### **Pre-Conditions**
```bash
# Verify WFQ scheduler health
curl -s http://localhost:8000/api/v1/control-plane/scheduler/status | jq

# Check current replay traffic baseline
kubectl top pods -l workload-type=replay
curl -s "http://localhost:9092/api/v1/query?query=rate(nats_jetstream_consumer_bytes{stream_class=\"replay\"}[5m])" | jq

# Confirm resource quotas are active
kubectl describe limitrange replay-limits
kubectl describe resourcequota replay-quota
```

### **Experiment Steps**

#### **Phase 1: Storm Preparation (3 minutes)**
```bash
# Deploy replay storm generators
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: replay-storm-generator
  labels:
    app: replay-storm
    chaos-experiment: replay-storm
spec:
  replicas: 10
  selector:
    matchLabels:
      app: replay-storm
  template:
    metadata:
      labels:
        app: replay-storm
        workload-type: replay
    spec:
      priorityClassName: replay-traffic-priority
      containers:
      - name: storm-generator
        image: xorb/replay-generator:latest
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "200m"
            memory: "256Mi"
        env:
        - name: STORM_MODE
          value: "10x"
        - name: TARGET_STREAM
          value: "xorb-replay-historical"
        - name: MESSAGES_PER_SECOND
          value: "1000"
        - name: CHAOS_EXPERIMENT
          value: "replay-storm"
EOF

# Record baseline metrics
BASELINE_LIVE_P95=$(curl -s "http://localhost:9092/api/v1/query?query=histogram_quantile(0.95,sum(rate(nats_request_duration_seconds_bucket{stream_class=\"live\"}[5m]))by(le))" | jq -r '.data.result[0].value[1] // "0"')
echo "BASELINE_LIVE_P95_MS=$(echo "$BASELINE_LIVE_P95 * 1000" | bc)" > /tmp/replay-storm.env
echo "CHAOS_STORM_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/replay-storm.env
```

#### **Phase 2: Storm Activation (30 seconds)**
```bash
# Scale storm generators to full capacity
kubectl scale deployment replay-storm-generator --replicas=10
kubectl wait --for=condition=Available deployment/replay-storm-generator --timeout=120s

# Verify storm traffic generation
sleep 30
STORM_RATE=$(curl -s "http://localhost:9092/api/v1/query?query=rate(nats_jetstream_consumer_bytes{stream_class=\"replay\"}[1m])" | jq -r '.data.result[0].value[1] // "0"')
echo "Storm traffic rate: $(echo "$STORM_RATE / 1024 / 1024" | bc)MB/s"
echo "CHAOS_STORM_RATE_MBPS=$(echo "$STORM_RATE / 1024 / 1024" | bc)" >> /tmp/replay-storm.env
```

#### **Phase 3: Isolation Validation (10 minutes)**
```bash
# Monitor live workload impact during storm
for i in {1..20}; do
  sleep 30

  # Check live workload SLO compliance
  LIVE_P95=$(curl -s "http://localhost:9092/api/v1/query?query=histogram_quantile(0.95,sum(rate(nats_request_duration_seconds_bucket{stream_class=\"live\"}[5m]))by(le))" | jq -r '.data.result[0].value[1] // "0"')
  LIVE_P95_MS=$(echo "$LIVE_P95 * 1000" | bc)

  # Check WFQ scheduler fairness
  FAIRNESS_INDEX=$(curl -s "http://localhost:9092/api/v1/query?query=wfq_scheduler_fairness_index" | jq -r '.data.result[0].value[1] // "0"')

  # Check resource contention
  CPU_THROTTLING=$(kubectl top pods -l workload-type=live | awk 'NR>1 {print $3}' | sed 's/[^0-9]//g' | awk '{sum+=$1} END {print sum/NR}')

  echo "T+${i}×30s: Live P95=${LIVE_P95_MS}ms, Fairness=${FAIRNESS_INDEX}, CPU=${CPU_THROTTLING}%"

  # SLO violation tracking
  if (( $(echo "$LIVE_P95_MS > 100" | bc -l) )); then
    echo "🚨 SLO VIOLATION: Live P95 ${LIVE_P95_MS}ms > 100ms during replay storm"
    echo "CHAOS_SLO_VIOLATION_COUNT=$((${CHAOS_SLO_VIOLATION_COUNT:-0} + 1))" >> /tmp/replay-storm.env
  fi

  # Fairness degradation check
  if (( $(echo "$FAIRNESS_INDEX < 0.7" | bc -l) )); then
    echo "⚖️  FAIRNESS DEGRADATION: Index ${FAIRNESS_INDEX} < 0.7"
  fi
done
```

#### **Phase 4: Storm Termination & Recovery (1.5 minutes)**
```bash
# Gracefully terminate replay storm
kubectl scale deployment replay-storm-generator --replicas=0
sleep 30

# Verify storm termination
FINAL_RATE=$(curl -s "http://localhost:9092/api/v1/query?query=rate(nats_jetstream_consumer_bytes{stream_class=\"replay\"}[1m])" | jq -r '.data.result[0].value[1] // "0"')
echo "Final replay rate: $(echo "$FINAL_RATE / 1024 / 1024" | bc)MB/s"

# Check live workload recovery
sleep 60
RECOVERY_P95=$(curl -s "http://localhost:9092/api/v1/query?query=histogram_quantile(0.95,sum(rate(nats_request_duration_seconds_bucket{stream_class=\"live\"}[5m]))by(le))" | jq -r '.data.result[0].value[1] // "0"')
RECOVERY_P95_MS=$(echo "$RECOVERY_P95 * 1000" | bc)

echo "Live workload recovery: ${RECOVERY_P95_MS}ms P95 latency"
echo "CHAOS_RECOVERY_P95_MS=${RECOVERY_P95_MS}" >> /tmp/replay-storm.env
echo "CHAOS_STORM_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/replay-storm.env

# Cleanup storm resources
kubectl delete deployment replay-storm-generator
```

### **Success Criteria**
- ✅ Live workload P95 latency remains < 100ms during 10x replay load
- ✅ WFQ scheduler fairness index stays > 0.7
- ✅ No CPU throttling of live workloads during storm
- ✅ Live workloads recover within 1 minute of storm termination

### **Auto-Remediation Paths**
1. **Priority Class Enforcement**: Kubernetes scheduler prioritizes live workloads
2. **WFQ Traffic Shaping**: Replay traffic automatically throttled when fairness degrades
3. **Resource Quota Enforcement**: Replay containers hit limits before impacting live workloads
4. **Circuit Breaker Activation**: Temporary replay suspension if live SLO breach detected

---

## 🔧 **Chaos Experiment #3: Corrupted Evidence Injection**

### **Experiment Overview**
**Objective**: Validate evidence verification resilience and chain of custody protection
**Duration**: 12 minutes
**Blast Radius**: Evidence verification service only
**SLO Target**: Evidence verification maintains >99% success rate with graceful degradation

### **Hypothesis**
*"When we inject corrupted evidence with invalid signatures and timestamps, the evidence verification system will reject malformed evidence while maintaining >99% success rate for valid evidence and preserving chain of custody integrity."*

### **Pre-Conditions**
```bash
# Verify evidence system health
curl -s http://localhost:8000/api/v1/provable-evidence/health | jq

# Check current evidence verification rates
curl -s "http://localhost:9092/api/v1/query?query=rate(evidence_verification_total[5m])" | jq
curl -s "http://localhost:9092/api/v1/query?query=rate(evidence_verification_success_total[5m])" | jq

# Verify Ed25519 key availability
kubectl get secret xorb-evidence-keys -o yaml | grep -E "(private.pem|public.pem)"
```

### **Experiment Steps**

#### **Phase 1: Corruption Generator Setup (2 minutes)**
```bash
# Deploy evidence corruption generator
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: evidence-corruption-generator
spec:
  template:
    spec:
      containers:
      - name: corruption-generator
        image: xorb/evidence-tester:latest
        env:
        - name: CORRUPTION_MODE
          value: "signatures,timestamps,chain"
        - name: CORRUPTION_RATE
          value: "20"  # 20% of evidence corrupted
        - name: EXPERIMENT_DURATION
          value: "10m"
        - name: TARGET_ENDPOINT
          value: "http://xorb-api:8000/api/v1/provable-evidence"
        command: ["/bin/bash"]
        args:
          - -c
          - |
            echo "Starting evidence corruption injection..."
            for i in {1..600}; do  # 10 minutes * 60 seconds

              # Generate valid evidence (80% of traffic)
              if (( $RANDOM % 100 > 20 )); then
                python3 /opt/evidence/generate_valid.py --endpoint $TARGET_ENDPOINT
              else
                # Inject corrupted evidence (20% of traffic)
                corruption_type=$(( $RANDOM % 3 ))
                case $corruption_type in
                  0) python3 /opt/evidence/corrupt_signature.py --endpoint $TARGET_ENDPOINT ;;
                  1) python3 /opt/evidence/corrupt_timestamp.py --endpoint $TARGET_ENDPOINT ;;
                  2) python3 /opt/evidence/corrupt_chain.py --endpoint $TARGET_ENDPOINT ;;
                esac
              fi

              sleep 1
            done
      restartPolicy: Never
EOF

# Record baseline success rate
BASELINE_SUCCESS_RATE=$(curl -s "http://localhost:9092/api/v1/query?query=rate(evidence_verification_success_total[5m])/rate(evidence_verification_total[5m])" | jq -r '.data.result[0].value[1] // "0"')
echo "BASELINE_SUCCESS_RATE=${BASELINE_SUCCESS_RATE}" > /tmp/evidence-corruption.env
echo "CHAOS_CORRUPTION_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/evidence-corruption.env
```

#### **Phase 2: Corruption Monitoring (8 minutes)**
```bash
# Monitor evidence verification during corruption injection
for i in {1..16}; do
  sleep 30

  # Check current success rate
  SUCCESS_RATE=$(curl -s "http://localhost:9092/api/v1/query?query=rate(evidence_verification_success_total[5m])/rate(evidence_verification_total[5m])" | jq -r '.data.result[0].value[1] // "0"')
  SUCCESS_PCT=$(echo "$SUCCESS_RATE * 100" | bc)

  # Check corruption detection rate
  CORRUPTION_DETECTED=$(curl -s "http://localhost:9092/api/v1/query?query=rate(evidence_corruption_detected_total[5m])" | jq -r '.data.result[0].value[1] // "0"')

  # Check chain of custody violations
  CHAIN_VIOLATIONS=$(curl -s "http://localhost:9092/api/v1/query?query=sum(evidence_chain_violations_total)" | jq -r '.data.result[0].value[1] // "0"')

  # Check evidence service CPU/memory under load
  EVIDENCE_CPU=$(kubectl top pods -l app=evidence-service | awk 'NR==2 {print $3}' | sed 's/%//')
  EVIDENCE_MEM=$(kubectl top pods -l app=evidence-service | awk 'NR==2 {print $4}' | sed 's/Mi//')

  echo "T+${i}×30s: Success=${SUCCESS_PCT}%, Corruption Detected=${CORRUPTION_DETECTED}/s, Chain Violations=${CHAIN_VIOLATIONS}, CPU=${EVIDENCE_CPU}%, Mem=${EVIDENCE_MEM}Mi"

  # SLO compliance check
  if (( $(echo "$SUCCESS_PCT < 99" | bc -l) )); then
    echo "🚨 SLO VIOLATION: Evidence success rate ${SUCCESS_PCT}% < 99%"
    echo "CHAOS_SLO_VIOLATION_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/evidence-corruption.env
  fi

  # Chain of custody breach check (critical)
  if (( $(echo "$CHAIN_VIOLATIONS > 0" | bc -l) )); then
    echo "🔐 CRITICAL: Chain of custody violations detected: ${CHAIN_VIOLATIONS}"
    echo "CHAOS_CHAIN_BREACH=true" >> /tmp/evidence-corruption.env
  fi
done
```

#### **Phase 3: Verification & Cleanup (2 minutes)**
```bash
# Stop corruption generator
kubectl delete job evidence-corruption-generator

# Wait for processing queue to clear
sleep 60

# Verify system recovery
FINAL_SUCCESS_RATE=$(curl -s "http://localhost:9092/api/v1/query?query=rate(evidence_verification_success_total[5m])/rate(evidence_verification_total[5m])" | jq -r '.data.result[0].value[1] // "0"')
FINAL_SUCCESS_PCT=$(echo "$FINAL_SUCCESS_RATE * 100" | bc)

echo "Final evidence verification success rate: ${FINAL_SUCCESS_PCT}%"

# Verify no persistent corruption
MERKLE_TREE_HEALTH=$(python3 src/api/app/services/g7_provable_evidence_service.py --verify-merkle-tree)
echo "Merkle tree integrity: $MERKLE_TREE_HEALTH"

# Audit evidence created during experiment
EVIDENCE_COUNT_EXPERIMENT=$(curl -s "http://localhost:9092/api/v1/query?query=increase(evidence_created_total[10m])" | jq -r '.data.result[0].value[1] // "0"')
EVIDENCE_REJECTED=$(curl -s "http://localhost:9092/api/v1/query?query=increase(evidence_verification_failed_total[10m])" | jq -r '.data.result[0].value[1] // "0"')

echo "Evidence processed: ${EVIDENCE_COUNT_EXPERIMENT}, Rejected: ${EVIDENCE_REJECTED}"
echo "CHAOS_EVIDENCE_PROCESSED=${EVIDENCE_COUNT_EXPERIMENT}" >> /tmp/evidence-corruption.env
echo "CHAOS_EVIDENCE_REJECTED=${EVIDENCE_REJECTED}" >> /tmp/evidence-corruption.env
echo "CHAOS_CORRUPTION_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/evidence-corruption.env
```

### **Success Criteria**
- ✅ Evidence verification success rate remains >99% for valid evidence
- ✅ All corrupted evidence is properly rejected (100% detection rate)
- ✅ No chain of custody violations occur
- ✅ Merkle tree integrity maintained throughout experiment
- ✅ Evidence service performance remains stable under load

### **Auto-Remediation Paths**
1. **Signature Validation**: Ed25519 signature verification rejects invalid signatures
2. **Timestamp Verification**: RFC 3161 timestamp validation detects temporal anomalies
3. **Chain Validation**: Merkle tree verification prevents chain tampering
4. **Rate Limiting**: Evidence ingestion throttled if corruption rate exceeds thresholds
5. **Circuit Breaker**: Evidence service enters read-only mode if integrity compromised

---

## 📊 **Chaos Experiment Metrics & Reporting**

### **Key Metrics Collected**
```yaml
# Experiment Execution Metrics
chaos_experiment_duration_seconds: Time taken for each experiment
chaos_experiment_success_rate: Percentage of successful chaos experiments
chaos_auto_recovery_time_seconds: Time to auto-recovery after failure injection

# SLO Compliance Metrics
chaos_slo_violation_count: Number of SLO violations during experiments
chaos_slo_compliance_rate: Overall SLO compliance percentage during chaos
chaos_error_budget_burn_rate: Error budget consumption during experiments

# Service Resilience Metrics
chaos_service_recovery_time_seconds: Time to service recovery post-experiment
chaos_cascade_failure_count: Number of cascading failures triggered
chaos_auto_remediation_success_rate: Success rate of auto-remediation mechanisms

# System Performance Metrics
chaos_throughput_degradation_percent: Performance degradation during experiments
chaos_resource_utilization_peak: Peak resource usage during chaos injection
chaos_error_rate_increase_percent: Error rate increase during experiments
```

### **Automated Reporting**
```bash
#!/bin/bash
# Generate chaos experiment report

EXPERIMENT_NAME=$1
EXPERIMENT_LOG_FILE="/tmp/${EXPERIMENT_NAME}.env"

if [[ -f "$EXPERIMENT_LOG_FILE" ]]; then
  source "$EXPERIMENT_LOG_FILE"

  cat > "/tmp/chaos-report-${EXPERIMENT_NAME}.md" <<EOF
# Chaos Experiment Report: ${EXPERIMENT_NAME}

**Date**: $(date)
**Duration**: ${CHAOS_EXPERIMENT_END} - ${CHAOS_EXPERIMENT_START}
**Status**: $(if [[ -n "$CHAOS_SLO_VIOLATION_TIME" ]]; then echo "❌ SLO Violation"; else echo "✅ Success"; fi)

## Summary
- **Auto-Recovery Time**: ${CHAOS_AUTO_RECOVERY_TIME:-"N/A"}
- **SLO Violations**: ${CHAOS_SLO_VIOLATION_COUNT:-0}
- **Error Budget Burn**: ${CHAOS_ERROR_BUDGET_BURN:-"Normal"}

## Detailed Results
$(cat "$EXPERIMENT_LOG_FILE" | grep -E "(VIOLATION|SUCCESS|RECOVERY)" | sed 's/^/- /')

## Recommendations
$(if [[ -n "$CHAOS_SLO_VIOLATION_TIME" ]]; then
  echo "- Investigate root cause of SLO violation"
  echo "- Review auto-remediation timing"
  echo "- Consider additional circuit breakers"
else
  echo "- Experiment successful, system resilient"
  echo "- Consider increasing chaos complexity"
  echo "- Validate results in production"
fi)
EOF

  echo "Chaos report generated: /tmp/chaos-report-${EXPERIMENT_NAME}.md"
fi
```

---

## 🎮 **Chaos Experiment Execution Framework**

### **Automated Chaos Runner**
```bash
#!/bin/bash
# tools/scripts/run_chaos_experiments.sh

set -euo pipefail

EXPERIMENT_DIR="/opt/chaos-experiments"
REPORT_DIR="/tmp/chaos-reports"
mkdir -p "$REPORT_DIR"

run_experiment() {
  local experiment_name=$1
  local experiment_script=$2

  echo "🧪 Starting chaos experiment: $experiment_name"

  # Pre-flight checks
  if ! ./tools/scripts/chaos_preflight_check.sh; then
    echo "❌ Pre-flight checks failed, aborting experiment"
    return 1
  fi

  # Execute experiment
  if timeout 1800 bash "$experiment_script"; then
    echo "✅ Experiment $experiment_name completed successfully"
    ./tools/scripts/generate_chaos_report.sh "$experiment_name"
  else
    echo "❌ Experiment $experiment_name failed or timed out"
    ./tools/scripts/chaos_emergency_cleanup.sh "$experiment_name"
    return 1
  fi
}

# Execute all chaos experiments
main() {
  echo "🚀 Starting XORB Chaos Engineering Drills"

  # Experiment 1: NATS Node Kill
  if run_experiment "nats-node-kill" "$EXPERIMENT_DIR/nats_node_kill.sh"; then
    echo "📊 NATS resilience validated"
  fi

  # Experiment 2: Replay Storm
  if run_experiment "replay-storm" "$EXPERIMENT_DIR/replay_storm.sh"; then
    echo "🌊 Replay isolation validated"
  fi

  # Experiment 3: Evidence Corruption
  if run_experiment "evidence-corruption" "$EXPERIMENT_DIR/evidence_corruption.sh"; then
    echo "🔐 Evidence integrity validated"
  fi

  # Generate consolidated report
  ./tools/scripts/generate_chaos_summary.sh "$REPORT_DIR"

  echo "✅ All chaos experiments completed"
}

main "$@"
```

### **Pre-Flight Safety Checks**
```bash
#!/bin/bash
# tools/scripts/chaos_preflight_check.sh

preflight_check() {
  echo "🔍 Running chaos experiment pre-flight checks..."

  # Check system health
  if ! ./tools/xorbctl/xorbctl status >/dev/null 2>&1; then
    echo "❌ System health check failed"
    return 1
  fi

  # Verify monitoring stack
  if ! curl -sf http://localhost:9092/-/healthy >/dev/null; then
    echo "❌ Prometheus not healthy"
    return 1
  fi

  # Check resource availability
  if [[ $(kubectl top nodes | awk 'NR>1 {print $3}' | sed 's/%//' | awk '{sum+=$1} END {print sum/NR}') -gt 80 ]]; then
    echo "❌ System CPU usage too high for chaos testing"
    return 1
  fi

  # Verify backup systems
  if ! curl -sf http://localhost:8000/api/v1/health >/dev/null; then
    echo "❌ Primary API not responsive"
    return 1
  fi

  echo "✅ Pre-flight checks passed"
  return 0
}

preflight_check "$@"
```

### **Emergency Cleanup Procedures**
```bash
#!/bin/bash
# tools/scripts/chaos_emergency_cleanup.sh

emergency_cleanup() {
  local experiment_name=$1

  echo "🚨 Emergency cleanup for experiment: $experiment_name"

  # Kill all chaos-related resources
  kubectl delete pods,deployments,jobs -l chaos-experiment="$experiment_name" --grace-period=0 --force 2>/dev/null || true

  # Restore service replicas
  kubectl scale statefulset nats-cluster --replicas=3
  kubectl scale deployment xorb-api --replicas=3
  kubectl scale deployment xorb-orchestrator --replicas=2

  # Clear chaos labels and configs
  kubectl label namespace default chaos-experiment- 2>/dev/null || true
  kubectl delete configmap chaos-experiment-config 2>/dev/null || true

  # Verify system recovery
  sleep 30
  if ./tools/xorbctl/xorbctl status; then
    echo "✅ Emergency cleanup successful, system recovered"
  else
    echo "❌ System still unhealthy after cleanup, escalate to on-call"
    return 1
  fi
}

emergency_cleanup "$@"
```

---

**Last Updated**: August 14, 2025
**Version**: v2025.08-rc1
**Next Review**: September 14, 2025
**Chaos Engineering Contact**: @xorb-chaos-team (Slack)
