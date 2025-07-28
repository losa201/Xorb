# XORB Operational Guide
**Enterprise Cybersecurity Platform - Production Operations Manual**

## 🚀 Quick Start

### Prerequisites
- Kubernetes cluster (v1.28+)
- AMD EPYC processor (recommended: 64+ cores)
- 64GB+ RAM
- 1TB+ SSD storage
- Helm 3.12+
- kubectl configured

### Rapid Deployment
```bash
# Clone and setup
git clone https://github.com/your-org/xorb.git
cd xorb
make setup

# Deploy to production
make k8s-apply ENV=production
make helm-install
```

---

## 📊 Service Architecture

### Core Services

#### 1. **XORB API Gateway** 
- **Port**: 8000
- **Health**: `/api/v1/gateway/health`
- **Metrics**: `/api/v1/gateway/stats`
- **Rate Limits**: 50/sec, 1000/min, 10000/hour

#### 2. **XORB Orchestrator**
- **Port**: 8001  
- **Health**: `/health`
- **Agents**: Up to 64 concurrent
- **SLO**: 99.95% availability

#### 3. **XORB Workers**
- **Port**: 8002
- **Scaling**: 2-20 replicas (HPA)
- **Throughput**: 100+ tasks/sec
- **Queue**: NATS JetStream

#### 4. **Data Ingestion**
- **Processing**: 10,000+ events/minute
- **Priorities**: Critical, High, Medium, Low, Background
- **Storage**: Redis (hot) + PostgreSQL (warm)

---

## 🔧 Operations

### Daily Operations

#### Health Monitoring
```bash
# Overall system health
kubectl get pods -n xorb-system

# Service-specific health checks
curl -s http://xorb-api:8000/api/v1/gateway/health | jq
curl -s http://xorb-orchestrator:8001/health | jq

# Check swarm intelligence
kubectl logs -f deployment/xorb-orchestrator | grep "swarm"
```

#### Performance Monitoring
```bash
# Real-time metrics
kubectl port-forward svc/prometheus 9090:9090
kubectl port-forward svc/grafana 3000:3000

# Key metrics to watch
# - xorb_agent_executions_total
# - xorb_api_requests_total  
# - xorb_error_budget_health
# - xorb_cost_optimization_savings
```

#### Agent Management
```bash
# View active agents
kubectl exec -it deployment/xorb-orchestrator -- python -c "
from xorb_core.agents.swarm_intelligence import SwarmIntelligence
swarm = SwarmIntelligence()
print(f'Active agents: {len(swarm.agents)}')
"

# Check agent performance
curl -s http://xorb-orchestrator:8001/api/v1/agents/metrics | jq

# Force role switching
curl -X POST http://xorb-orchestrator:8001/api/v1/swarm/optimize-roles
```

### Weekly Operations

#### Performance Review
```bash
# Generate weekly report
kubectl exec -it deployment/xorb-orchestrator -- python scripts/weekly_report.py

# Review error budgets
kubectl get prometheusrules xorb-error-budget-rules -o yaml

# Cost optimization review
kubectl logs deployment/xorb-cost-optimizer | tail -100
```

#### Security Audit
```bash
# RBAC compliance check
kubectl exec -it deployment/xorb-rbac-tracer -- python scripts/rbac_audit.py

# Vulnerability scan
kubectl get vulnerabilityreports -A

# Review audit logs
kubectl logs deployment/xorb-rbac-tracer | grep "SECURITY"
```

### Monthly Operations

#### Capacity Planning
```bash
# Resource utilization analysis
kubectl top nodes
kubectl top pods -n xorb-system

# Scale recommendations
kubectl get hpa -n xorb-system
kubectl describe hpa xorb-worker-hpa
```

#### Backup and Disaster Recovery
```bash
# Database backup
kubectl exec -it postgresql-0 -- pg_dump xorb > backup_$(date +%Y%m%d).sql

# Configuration backup
kubectl get configmaps -n xorb-system -o yaml > configmaps_backup.yaml
kubectl get secrets -n xorb-system -o yaml > secrets_backup.yaml

# Test disaster recovery
make test-disaster-recovery
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. **High API Latency**
```bash
# Check symptoms
curl -w "%{time_total}" http://xorb-api:8000/api/v1/status

# Investigate causes
kubectl logs deployment/xorb-api | grep "slow"
kubectl describe pod xorb-api-xxx

# Solutions
kubectl scale deployment xorb-api --replicas=5  # Scale up
kubectl rollout restart deployment xorb-api     # Restart if needed
```

#### 2. **Agent Failures**
```bash
# Check agent status
kubectl exec -it deployment/xorb-orchestrator -- python -c "
from xorb_core.orchestration.enhanced_orchestrator import EnhancedOrchestrator
orch = EnhancedOrchestrator()
print(f'Failed agents: {orch.get_failed_agents()}')
"

# Restart failed agents
curl -X POST http://xorb-orchestrator:8001/api/v1/agents/restart-failed

# Check logs
kubectl logs deployment/xorb-worker | grep "ERROR"
```

#### 3. **Circuit Breaker Opened**
```bash
# Check circuit breaker status
curl -s http://xorb-api:8000/api/v1/gateway/stats | jq '.circuit_breaker'

# Manual reset (if safe)
curl -X POST http://xorb-api:8000/api/v1/gateway/circuit-breaker/reset

# Root cause analysis
kubectl logs deployment/xorb-api | grep "circuit"
```

#### 4. **Data Ingestion Backlog**
```bash
# Check queue status
kubectl exec -it deployment/xorb-orchestrator -- python -c "
from xorb_core.data_ingestion.parallel_ingestion import ParallelIngestionEngine
engine = ParallelIngestionEngine()
print(f'Backlog: {engine.priority_queue.size()}')
"

# Scale ingestion workers
kubectl patch deployment xorb-worker -p '{"spec":{"replicas":15}}'

# Clear backlog (emergency)
curl -X POST http://xorb-orchestrator:8001/api/v1/ingestion/clear-backlog
```

### Emergency Procedures

#### Complete System Restart
```bash
# Graceful restart sequence
kubectl scale deployment xorb-worker --replicas=0
kubectl scale deployment xorb-api --replicas=0  
kubectl scale deployment xorb-orchestrator --replicas=0

sleep 30

kubectl scale deployment xorb-orchestrator --replicas=1
sleep 15
kubectl scale deployment xorb-api --replicas=3
kubectl scale deployment xorb-worker --replicas=5
```

#### Disaster Recovery
```bash
# Restore from backup
kubectl apply -f disaster-recovery/

# Verify system integrity
make test-all
make validate-deployment

# Gradual traffic restoration
kubectl patch service xorb-api -p '{"spec":{"selector":{"version":"new"}}}'
```

---

## 📈 Performance Tuning

### EPYC Optimization

#### CPU Affinity
```bash
# Check current affinity
kubectl exec -it xorb-orchestrator-0 -- cat /proc/self/status | grep Cpus_allowed_list

# Optimize for NUMA
kubectl patch deployment xorb-orchestrator -p '{
  "spec":{
    "template":{
      "spec":{
        "containers":[{
          "name":"orchestrator",
          "resources":{
            "requests":{"cpu":"8000m"},
            "limits":{"cpu":"16000m"}
          }
        }]
      }
    }
  }
}'
```

#### Memory Optimization
```bash
# Huge pages configuration
kubectl patch daemonset node-config -p '{
  "spec":{
    "template":{
      "spec":{
        "containers":[{
          "name":"config",
          "env":[{
            "name":"HUGEPAGES_2MB",
            "value":"1024"
          }]
        }]
      }
    }
  }
}'
```

### Database Tuning

#### PostgreSQL Optimization
```bash
kubectl exec -it postgresql-0 -- psql -c "
ALTER SYSTEM SET shared_buffers = '16GB';
ALTER SYSTEM SET effective_cache_size = '48GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
SELECT pg_reload_conf();
"
```

#### Redis Optimization
```bash
kubectl exec -it redis-0 -- redis-cli CONFIG SET maxmemory 8gb
kubectl exec -it redis-0 -- redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

---

## 🔒 Security Operations

### Access Management

#### Add New User
```bash
# Create service account
kubectl create serviceaccount new-user -n xorb-system

# Assign role
kubectl create rolebinding new-user-binding \
  --clusterrole=xorb-operator \
  --serviceaccount=xorb-system:new-user \
  -n xorb-system
```

#### Rotate Secrets
```bash
# Generate new API keys
kubectl delete secret xorb-api-keys
kubectl create secret generic xorb-api-keys \
  --from-literal=primary-key=$(openssl rand -hex 32) \
  --from-literal=backup-key=$(openssl rand -hex 32)

# Restart services to pick up new keys
kubectl rollout restart deployment/xorb-api
```

### Security Monitoring

#### Audit Log Analysis
```bash
# Check recent security events
kubectl logs deployment/xorb-rbac-tracer | grep "$(date +'%Y-%m-%d')" | grep "SECURITY"

# Unauthorized access attempts
kubectl logs deployment/xorb-rbac-tracer | grep "access_denied"

# Privilege escalation attempts
kubectl logs deployment/xorb-rbac-tracer | grep "privilege_escalation"
```

#### Threat Detection
```bash
# Run security scan
kubectl exec -it deployment/xorb-orchestrator -- python scripts/security_scan.py

# Check for anomalies
curl -s http://xorb-orchestrator:8001/api/v1/analytics/anomalies | jq
```

---

## 💰 Cost Management

### Cost Monitoring
```bash
# Current resource costs
kubectl get pods -n xorb-system -o custom-columns=NAME:.metadata.name,CPU:.spec.containers[*].resources.requests.cpu,MEMORY:.spec.containers[*].resources.requests.memory

# Cost optimization recommendations
kubectl logs deployment/xorb-cost-optimizer | grep "RECOMMENDATION"
```

### Right-sizing
```bash
# Analyze resource usage
kubectl top pods -n xorb-system --sort-by=cpu
kubectl top pods -n xorb-system --sort-by=memory

# Apply right-sizing recommendations
kubectl patch deployment xorb-api -p '{
  "spec":{
    "template":{
      "spec":{
        "containers":[{
          "name":"api",
          "resources":{
            "requests":{"cpu":"500m","memory":"1Gi"},
            "limits":{"cpu":"2000m","memory":"4Gi"}
          }
        }]
      }
    }
  }
}'
```

---

## 🔧 Configuration Management

### Environment Configuration

#### Development
```yaml
# values-dev.yaml
replicaCount: 1
resources:
  requests:
    cpu: 100m
    memory: 128Mi
autoscaling:
  enabled: false
```

#### Staging  
```yaml
# values-staging.yaml
replicaCount: 2
resources:
  requests:
    cpu: 500m
    memory: 1Gi
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5
```

#### Production
```yaml
# values-production.yaml
replicaCount: 3
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
```

### Feature Flags
```bash
# Enable new feature
kubectl patch configmap xorb-config -p '{
  "data":{
    "FEATURE_ADVANCED_ANALYTICS":"true",
    "FEATURE_REAL_TIME_EVOLUTION":"true"
  }
}'

# Restart to apply changes
kubectl rollout restart deployment/xorb-orchestrator
```

---

## 📞 Support Contacts

### Emergency Contacts
- **Critical Issues**: support-critical@xorb.com
- **Security Incidents**: security@xorb.com  
- **On-call Engineer**: +1-555-XORB-911

### Support Channels
- **Documentation**: https://docs.xorb.com
- **Community**: https://community.xorb.com
- **GitHub Issues**: https://github.com/xorb/platform/issues
- **Slack**: #xorb-support

### SLA Commitments
- **Critical (P0)**: 15 minutes response, 4 hours resolution
- **High (P1)**: 1 hour response, 24 hours resolution  
- **Medium (P2)**: 4 hours response, 72 hours resolution
- **Low (P3)**: 24 hours response, 1 week resolution

---

*This operational guide provides comprehensive instructions for running XORB in production. For additional support, consult the documentation or contact the support team.*