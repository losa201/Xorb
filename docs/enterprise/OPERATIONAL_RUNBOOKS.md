# XORB Platform Operational Runbooks

- **Version**: 2.0
- **Last Updated**: January 2025
- **Audience**: DevOps Engineers, SRE Teams, Operations Staff

##  📋 **Overview**

This document provides step-by-step operational procedures for managing the XORB cybersecurity platform in production environments. These runbooks are designed for 24/7 operations teams and incident response.

- --

##  🚨 **Incident Response Runbooks**

###  **Runbook 1: High CPU Usage Alert**

- **Trigger**: CPU usage > 80% for 5+ minutes
- **Severity**: Warning
- **Response Time**: 15 minutes

####  **Investigation Steps**

```bash
# 1. Check current CPU usage across all pods
kubectl top pods -n xorb-production --sort-by=cpu

# 2. Identify high CPU consuming processes
kubectl exec -it $(kubectl get pods -n xorb-production -l app=xorb-api -o jsonpath='{.items[0].metadata.name}') -- top

# 3. Check application metrics
curl -s https://api.your-domain.com/metrics | grep -E "(cpu_usage|process_cpu)"

# 4. Review recent application logs for errors
kubectl logs -l app=xorb-api -n xorb-production --since=15m | grep -E "(ERROR|WARN)"
```

####  **Resolution Actions**

```bash
# Option 1: Scale horizontally (immediate relief)
kubectl scale deployment xorb-api --replicas=5 -n xorb-production

# Option 2: Restart high-CPU pods (if memory leak suspected)
kubectl delete pod -l app=xorb-api -n xorb-production
kubectl rollout status deployment/xorb-api -n xorb-production

# Option 3: Adjust resource limits (if consistently high)
kubectl patch deployment xorb-api -n xorb-production -p='
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "xorb-api",
          "resources": {
            "limits": {"cpu": "2000m"},
            "requests": {"cpu": "1000m"}
          }
        }]
      }
    }
  }
}'
```

####  **Follow-up Actions**

1. Monitor CPU usage for next 30 minutes
2. Check if HPA is properly configured
3. Review application performance metrics
4. Create Jira ticket if pattern continues

- --

###  **Runbook 2: Database Connection Pool Exhaustion**

- **Trigger**: "Connection pool exhausted" errors in logs
- **Severity**: Critical
- **Response Time**: 5 minutes

####  **Investigation Steps**

```bash
# 1. Check database connection status
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT state, count(*)
    FROM pg_stat_activity
    WHERE datname = 'xorb_production'
    GROUP BY state;"

# 2. Check application connection pool metrics
kubectl logs -l app=xorb-api -n xorb-production --since=10m | \
  grep -E "(connection.*pool|database.*error)"

# 3. Identify long-running queries
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT pid, state, query_start, query
    FROM pg_stat_activity
    WHERE state = 'active'
    AND query_start < NOW() - INTERVAL '30 seconds';"
```

####  **Resolution Actions**

```bash
# Option 1: Kill long-running queries (emergency)
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE query_start < NOW() - INTERVAL '2 minutes'
    AND state = 'active';"

# Option 2: Restart application pods to reset connections
kubectl rollout restart deployment/xorb-api -n xorb-production

# Option 3: Scale down and up to reset connection pools
kubectl scale deployment xorb-api --replicas=1 -n xorb-production
sleep 30
kubectl scale deployment xorb-api --replicas=3 -n xorb-production

# Option 4: Increase connection pool size (temporary)
kubectl set env deployment/xorb-api DATABASE_POOL_SIZE=30 -n xorb-production
```

####  **Follow-up Actions**

1. Monitor database connections for 1 hour
2. Review database performance metrics
3. Optimize slow queries identified
4. Consider increasing database resources

- --

###  **Runbook 3: Redis Memory Usage Critical**

- **Trigger**: Redis memory usage > 90%
- **Severity**: Critical
- **Response Time**: 10 minutes

####  **Investigation Steps**

```bash
# 1. Check Redis memory usage
kubectl exec -it xorb-redis-master-0 -n xorb-production -- \
  redis-cli info memory

# 2. Check Redis key distribution
kubectl exec -it xorb-redis-master-0 -n xorb-production -- \
  redis-cli --bigkeys

# 3. Check application cache usage patterns
kubectl logs -l app=xorb-api -n xorb-production --since=15m | \
  grep -E "(cache|redis)" | head -20
```

####  **Resolution Actions**

```bash
# Option 1: Flush specific cache namespaces (safest)
kubectl exec -it xorb-redis-master-0 -n xorb-production -- \
  redis-cli EVAL "
    for i, key in ipairs(redis.call('KEYS', 'cache:*')) do
      redis.call('DEL', key)
    end
    return 'OK'" 0

# Option 2: Set memory policies (if not set)
kubectl exec -it xorb-redis-master-0 -n xorb-production -- \
  redis-cli config set maxmemory-policy allkeys-lru

# Option 3: Emergency - flush all data (data loss!)
# Only use if system is completely down
kubectl exec -it xorb-redis-master-0 -n xorb-production -- \
  redis-cli flushall

# Option 4: Scale Redis memory
kubectl patch statefulset xorb-redis-master -n xorb-production -p='
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "redis",
          "resources": {
            "limits": {"memory": "4Gi"},
            "requests": {"memory": "2Gi"}
          }
        }]
      }
    }
  }
}'
```

####  **Follow-up Actions**

1. Monitor Redis memory for next 2 hours
2. Review cache TTL policies
3. Implement cache key expiration
4. Consider Redis cluster setup

- --

###  **Runbook 4: Application 5xx Error Rate High**

- **Trigger**: 5xx error rate > 5% for 5+ minutes
- **Severity**: High
- **Response Time**: 10 minutes

####  **Investigation Steps**

```bash
# 1. Check error rate by endpoint
kubectl logs -l app=xorb-api -n xorb-production --since=10m | \
  grep "HTTP/1.1\" 5" | awk '{print $7}' | sort | uniq -c | sort -nr

# 2. Check application health endpoints
curl -f https://api.your-domain.com/health
curl -f https://api.your-domain.com/readiness

# 3. Check recent application logs for errors
kubectl logs -l app=xorb-api -n xorb-production --since=10m | \
  grep -E "(ERROR|Exception|Traceback)" | tail -20

# 4. Check resource utilization
kubectl top pods -n xorb-production
```

####  **Resolution Actions**

```bash
# Option 1: Restart failing pods
kubectl get pods -n xorb-production -l app=xorb-api -o jsonpath='{.items[*].metadata.name}' | \
  xargs -I {} kubectl delete pod {} -n xorb-production

# Option 2: Scale up to handle load
kubectl scale deployment xorb-api --replicas=5 -n xorb-production

# Option 3: Rollback to previous version (if recent deployment)
kubectl rollout undo deployment/xorb-api -n xorb-production

# Option 4: Enable circuit breaker (if external dependency issue)
kubectl set env deployment/xorb-api CIRCUIT_BREAKER_ENABLED=true -n xorb-production
```

####  **Follow-up Actions**

1. Monitor error rates for 30 minutes
2. Review application and infrastructure metrics
3. Check external dependencies
4. Post-incident review if pattern continues

- --

##  🔧 **Maintenance Runbooks**

###  **Runbook 5: Planned Database Maintenance**

- **Purpose**: Database upgrades, index maintenance, backup verification
- **Duration**: 2-4 hours
- **Downtime**: 15-30 minutes

####  **Pre-Maintenance Checklist**

```bash
# 1. Verify backup integrity
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  pg_dump -U postgres xorb_production | head -20

# 2. Check database performance metrics
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
    FROM pg_stat_user_tables
    ORDER BY n_tup_ins+n_tup_upd+n_tup_del DESC
    LIMIT 10;"

# 3. Scale application to minimum
kubectl scale deployment xorb-api --replicas=1 -n xorb-production
kubectl scale deployment xorb-orchestrator --replicas=1 -n xorb-production

# 4. Enable maintenance mode
kubectl set env deployment/xorb-api MAINTENANCE_MODE=true -n xorb-production
```

####  **Maintenance Steps**

```bash
# 1. Create full backup
DATE=$(date +%Y%m%d_%H%M%S)
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  pg_dump -U postgres xorb_production > "backup_${DATE}.sql"

# 2. Run VACUUM and ANALYZE
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "VACUUM ANALYZE;"

# 3. Reindex critical tables
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "REINDEX DATABASE xorb_production;"

# 4. Update statistics
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "ANALYZE;"
```

####  **Post-Maintenance Checklist**

```bash
# 1. Verify database functionality
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "SELECT COUNT(*) FROM users;"

# 2. Disable maintenance mode
kubectl set env deployment/xorb-api MAINTENANCE_MODE- -n xorb-production

# 3. Scale back to normal
kubectl scale deployment xorb-api --replicas=3 -n xorb-production
kubectl scale deployment xorb-orchestrator --replicas=2 -n xorb-production

# 4. Run health checks
curl -f https://api.your-domain.com/health
curl -f https://api.your-domain.com/readiness

# 5. Monitor for 30 minutes
kubectl logs -l app=xorb-api -n xorb-production --follow
```

- --

###  **Runbook 6: Certificate Renewal**

- **Purpose**: SSL/TLS certificate renewal
- **Duration**: 30 minutes
- **Downtime**: None (rolling update)

####  **Certificate Check**

```bash
# 1. Check current certificate expiration
echo | openssl s_client -servername api.your-domain.com \
  -connect api.your-domain.com:443 2>/dev/null | \
  openssl x509 -noout -dates

# 2. Check cert-manager status
kubectl get certificates -n xorb-production
kubectl describe certificate xorb-tls-secret -n xorb-production
```

####  **Renewal Process**

```bash
# 1. Force certificate renewal (cert-manager)
kubectl annotate certificate xorb-tls-secret \
  -n xorb-production cert-manager.io/issue-temporary-certificate=true

# 2. Monitor renewal
kubectl get certificaterequests -n xorb-production -w

# 3. Verify new certificate
kubectl get secret xorb-tls-secret -n xorb-production -o yaml | \
  grep tls.crt | awk '{print $2}' | base64 -d | \
  openssl x509 -noout -dates
```

####  **Verification**

```bash
# 1. Test HTTPS connectivity
curl -I https://api.your-domain.com/health

# 2. Check certificate chain
echo | openssl s_client -servername api.your-domain.com \
  -connect api.your-domain.com:443 2>/dev/null | \
  openssl x509 -noout -text | grep -E "(Issuer|Subject|Not After)"

# 3. Update monitoring
# Update certificate expiration monitoring alerts
```

- --

##  📊 **Monitoring Runbooks**

###  **Runbook 7: Grafana Dashboard Issues**

- **Purpose**: Resolve Grafana dashboard and alerting issues

####  **Common Issues & Solutions**

```bash
# 1. Grafana pod not starting
kubectl describe pod -l app=grafana -n xorb-production
kubectl logs -l app=grafana -n xorb-production

# 2. Data source connection issues
kubectl exec -it $(kubectl get pods -l app=grafana -n xorb-production -o jsonpath='{.items[0].metadata.name}') -- \
  curl -f http://prometheus:9090/api/v1/query?query=up

# 3. Reset Grafana admin password
kubectl exec -it $(kubectl get pods -l app=grafana -n xorb-production -o jsonpath='{.items[0].metadata.name}') -- \
  grafana-cli admin reset-admin-password newpassword123

# 4. Import missing dashboards
kubectl apply -f services/infrastructure/monitoring/grafana/dashboards/
```

###  **Runbook 8: Prometheus Storage Issues**

- **Purpose**: Resolve Prometheus storage and retention issues

####  **Storage Management**

```bash
# 1. Check Prometheus storage usage
kubectl exec -it prometheus-0 -n xorb-production -- \
  df -h /prometheus

# 2. Check retention settings
kubectl exec -it prometheus-0 -n xorb-production -- \
  cat /etc/prometheus/prometheus.yml | grep retention

# 3. Compact old data (if needed)
kubectl exec -it prometheus-0 -n xorb-production -- \
  promtool tsdb create-blocks-from openmetrics /prometheus/data

# 4. Clean up old data
kubectl exec -it prometheus-0 -n xorb-production -- \
  find /prometheus -name "*.tmp" -delete
```

- --

##  🔐 **Security Runbooks**

###  **Runbook 9: Security Incident Response**

- **Purpose**: Respond to security incidents and breaches
- **Severity**: Critical
- **Response Time**: Immediate

####  **Immediate Actions**

```bash
# 1. Isolate affected components
kubectl patch networkpolicy xorb-network-policy -n xorb-production -p='
{
  "spec": {
    "ingress": [],
    "egress": []
  }
}'

# 2. Scale down to minimum
kubectl scale deployment xorb-api --replicas=0 -n xorb-production

# 3. Capture evidence
kubectl logs -l app=xorb-api -n xorb-production --since=24h > incident_logs.txt
kubectl get events -n xorb-production --sort-by='.lastTimestamp' > incident_events.txt

# 4. Create secure backup
kubectl create job incident-backup --from=cronjob/postgres-backup -n xorb-production
```

####  **Investigation**

```bash
# 1. Check for unauthorized access
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT * FROM auth_logs
    WHERE created_at > NOW() - INTERVAL '24 hours'
    AND status = 'failed'
    ORDER BY created_at DESC;"

# 2. Review audit logs
kubectl logs -l app=xorb-api -n xorb-production --since=24h | \
  grep -E "(authentication|authorization|security)" > security_audit.log

# 3. Check for malicious activity
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT DISTINCT user_agent, source_ip, COUNT(*)
    FROM request_logs
    WHERE created_at > NOW() - INTERVAL '24 hours'
    GROUP BY user_agent, source_ip
    HAVING COUNT(*) > 1000;"
```

####  **Recovery**

```bash
# 1. Patch security vulnerabilities
kubectl set image deployment/xorb-api xorb-api=ghcr.io/your-org/xorb:v2.1.1-security-patch

# 2. Rotate all secrets
kubectl delete secret xorb-config -n xorb-production
kubectl create secret generic xorb-config --from-env-file=.env.production.new

# 3. Restore normal operations
kubectl patch networkpolicy xorb-network-policy -n xorb-production --type=merge -p='
{
  "spec": {
    "ingress": [{"from": [{"namespaceSelector": {"matchLabels": {"name": "ingress-nginx"}}}]}]
  }
}'

kubectl scale deployment xorb-api --replicas=3 -n xorb-production
```

- --

###  **Runbook 10: Secret Rotation**

- **Purpose**: Regular rotation of secrets and credentials
- **Frequency**: Quarterly or on-demand

####  **Rotation Process**

```bash
# 1. Generate new secrets
NEW_JWT_SECRET=$(openssl rand -base64 32)
NEW_DB_PASSWORD=$(openssl rand -base64 16)

# 2. Update Vault (if using)
kubectl exec -it vault-0 -n xorb-production -- \
  vault kv put secret/xorb/config \
  jwt_secret="${NEW_JWT_SECRET}" \
  database_password="${NEW_DB_PASSWORD}"

# 3. Update Kubernetes secrets
kubectl patch secret xorb-config -n xorb-production -p='
{
  "data": {
    "jwt-secret": "'$(echo -n $NEW_JWT_SECRET | base64)'",
    "database-password": "'$(echo -n $NEW_DB_PASSWORD | base64)'"
  }
}'

# 4. Rolling restart to pick up new secrets
kubectl rollout restart deployment/xorb-api -n xorb-production
kubectl rollout restart deployment/xorb-orchestrator -n xorb-production

# 5. Verify functionality
sleep 60
curl -f https://api.your-domain.com/health
```

- --

##  📈 **Performance Optimization Runbooks**

###  **Runbook 11: Database Performance Tuning**

- **Purpose**: Optimize database performance during high load

####  **Performance Analysis**

```bash
# 1. Identify slow queries
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT query, calls, total_time, mean_time
    FROM pg_stat_statements
    ORDER BY total_time DESC
    LIMIT 10;"

# 2. Check index usage
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
    FROM pg_stat_user_indexes
    WHERE idx_scan = 0;"

# 3. Analyze table statistics
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, n_live_tup, n_dead_tup
    FROM pg_stat_user_tables
    ORDER BY n_dead_tup DESC;"
```

####  **Optimization Actions**

```bash
# 1. Create missing indexes
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    CREATE INDEX CONCURRENTLY idx_audit_logs_created_at ON audit_logs(created_at);
    CREATE INDEX CONCURRENTLY idx_threat_data_severity ON threat_data(severity, created_at);"

# 2. Update table statistics
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "ANALYZE;"

# 3. Optimize configuration
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "
    ALTER SYSTEM SET shared_buffers = '1GB';
    ALTER SYSTEM SET effective_cache_size = '3GB';
    ALTER SYSTEM SET maintenance_work_mem = '256MB';
    SELECT pg_reload_conf();"
```

- --

##  📚 **Reference Information**

###  **Key Contacts**

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | +1-555-0199 | Primary |
| Database Admin | +1-555-0299 | Critical DB issues |
| Security Team | +1-555-0399 | Security incidents |
| Platform Team Lead | +1-555-0499 | Major incidents |

###  **Key Metrics Thresholds**

| Metric | Warning | Critical |
|--------|---------|----------|
| CPU Usage | 70% | 85% |
| Memory Usage | 80% | 90% |
| Response Time | 1s | 2s |
| Error Rate | 2% | 5% |
| Database Connections | 80% | 95% |

###  **Common Commands**

```bash
# Quick health check
kubectl get pods -n xorb-production
curl -s https://api.your-domain.com/health | jq

# View recent logs
kubectl logs -l app=xorb-api -n xorb-production --since=10m --tail=100

# Check resource usage
kubectl top pods -n xorb-production --sort-by=cpu
kubectl top nodes

# Emergency scale down
kubectl scale deployment xorb-api --replicas=0 -n xorb-production

# Emergency scale up
kubectl scale deployment xorb-api --replicas=5 -n xorb-production
```

- --

- **Document Revision History**:
- v2.0 - January 2025 - Initial enterprise runbooks
- v1.9 - December 2024 - Added security incident procedures
- v1.8 - November 2024 - Added performance optimization runbooks