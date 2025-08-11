# 🔧 XORB Platform Operational Runbook

[![Operational Status](https://img.shields.io/badge/Operations-Production%20Ready-green)](#production-operations)
[![Monitoring](https://img.shields.io/badge/Monitoring-Comprehensive-blue)](#monitoring-operations)
[![Incident Response](https://img.shields.io/badge/Incident%20Response-Automated-orange)](#incident-response)

> **Consolidated Operational Documentation**: Complete operational procedures and runbooks for the XORB platform in production environments.

##  🎯 Operational Overview

This runbook consolidates all operational insights from XORB platform strategic implementations and remediation activities, providing comprehensive procedures for day-to-day operations, monitoring, and incident response.

###  Operational Principles
- **Proactive Monitoring**: Continuous health and performance monitoring
- **Automated Response**: Self-healing and automated incident response
- **Security First**: All operations maintain security posture
- **Compliance**: Audit trail and compliance requirements met
- **High Availability**: 99.9% uptime SLA maintenance

##  🚀 Daily Operations

###  Health Check Procedures
```bash
# Primary health checks (run every 5 minutes)
curl -f http://localhost:8000/api/v1/health || echo "API service down"
curl -f http://localhost:8000/api/v1/readiness || echo "API not ready"

# Service-specific health checks
docker-compose ps | grep -v "Up" || echo "Container issues detected"
redis-cli -h localhost -p 6379 ping || echo "Redis connectivity issue"
pg_isready -h localhost -p 5432 || echo "PostgreSQL connectivity issue"
```text

###  Performance Monitoring
```bash
# Performance metrics collection
./scripts/performance-benchmark.sh
./scripts/health-monitor.sh

# Resource utilization monitoring
docker stats --no-stream
df -h  # Disk usage
free -h  # Memory usage
top -n 1 -b | head -20  # CPU usage
```text

###  Log Management
```bash
# Application log monitoring
tail -f logs/xorb-api.log | grep -E "(ERROR|WARN|CRITICAL)"
tail -f logs/orchestrator.log | grep -E "(FAILED|ERROR)"
tail -f logs/audit.log | grep -E "(SECURITY|VIOLATION)"

# Container log monitoring
docker-compose logs -f --tail=100 api
docker-compose logs -f --tail=100 orchestrator
docker-compose logs -f --tail=100 redis
```text

##  🔐 Security Operations

###  Certificate Management
```bash
# Daily certificate health check
./scripts/validate/test_tls.sh
./scripts/validate/test_mtls.sh

# Certificate expiry monitoring (run daily)
find ./secrets/tls -name "*.pem" -exec openssl x509 -noout -dates -in {} \; | \
grep "notAfter" | while read line; do
    echo "Certificate expiry: $line"
done

# Automated certificate rotation (when needed)
./scripts/rotate-certs.sh --check-expiry
```text

###  Security Monitoring
```bash
# Security event monitoring
grep -E "(FAILED_LOGIN|UNAUTHORIZED|SECURITY_VIOLATION)" logs/audit.log
grep -E "(RATE_LIMIT_EXCEEDED|SUSPICIOUS_ACTIVITY)" logs/security.log

# Vulnerability scan status
curl -s http://localhost:8000/api/v1/ptaas/health | jq '.scanner_status'

# Security policy validation
conftest test --policy policies/security-policy.rego infra/docker-compose.tls.yml
```text

###  Access Control Verification
```bash
# Verify mTLS authentication
openssl s_client -connect api:8443 -cert secrets/tls/api-client/cert.pem \
  -key secrets/tls/api-client/key.pem -CAfile secrets/tls/ca/ca.pem

# API authentication testing
curl -H "Authorization: Bearer invalid_token" \
  http://localhost:8000/api/v1/protected/endpoint
# Should return 401 Unauthorized
```text

##  📊 Monitoring Operations

###  Prometheus Metrics Monitoring
```bash
# Key metrics to monitor
curl -s http://localhost:9092/api/v1/query?query=up | jq '.data.result'
curl -s http://localhost:9092/api/v1/query?query=http_requests_total | jq '.data.result'
curl -s http://localhost:9092/api/v1/query?query=ptaas_scans_active | jq '.data.result'

# Alert rule validation
curl -s http://localhost:9092/api/v1/rules | jq '.data.groups[].rules[].alerts'
```text

###  Grafana Dashboard Monitoring
```bash
# Dashboard health check
curl -f http://localhost:3010/api/health
curl -f http://localhost:3010/api/datasources/proxy/1/api/v1/query?query=up

# Alert notification testing
curl -X POST http://localhost:3010/api/annotations \
  -H "Content-Type: application/json" \
  -d '{"text":"Test annotation","tags":["test"]}'
```text

###  Database Operations
```bash
# PostgreSQL health and performance
psql -h localhost -U xorb_user -d xorb -c "SELECT version();"
psql -h localhost -U xorb_user -d xorb -c "SELECT pg_stat_database.datname,
  pg_stat_database.tup_returned, pg_stat_database.tup_fetched
  FROM pg_stat_database WHERE datname='xorb';"

# Redis operations monitoring
redis-cli -h localhost -p 6379 info replication
redis-cli -h localhost -p 6379 info memory
redis-cli -h localhost -p 6379 info stats
```text

##  🎯 PTaaS Operations

###  Scanner Health Monitoring
```bash
# Scanner availability check
nmap --version || echo "Nmap not available"
nuclei -version || echo "Nuclei not available"
nikto -Version || echo "Nikto not available"

# Scanner service status
curl -s http://localhost:8000/api/v1/ptaas/scanners/status | jq '.'

# Active scan monitoring
curl -s http://localhost:8000/api/v1/ptaas/sessions/active | jq '.count'
```text

###  Scan Queue Management
```bash
# Queue depth monitoring
curl -s http://localhost:8000/api/v1/ptaas/queue/status | jq '.'

# Scan performance metrics
curl -s http://localhost:8000/api/v1/ptaas/metrics | jq '.scan_times'

# Failed scan investigation
curl -s http://localhost:8000/api/v1/ptaas/sessions?status=failed | jq '.'
```text

###  Orchestration Monitoring
```bash
# Temporal workflow health
curl -f http://localhost:8233/api/v1/namespaces/default/workflows

# Workflow execution monitoring
temporal workflow list --namespace default
temporal workflow describe --workflow-id <workflow_id>

# Activity failure investigation
temporal workflow show --workflow-id <workflow_id> --run-id <run_id>
```text

##  🚨 Incident Response

###  Service Recovery Procedures
```bash
# API service restart
docker-compose restart api
sleep 30
curl -f http://localhost:8000/api/v1/health

# Database connection recovery
docker-compose restart postgres
sleep 60
pg_isready -h localhost -p 5432

# Redis service recovery
docker-compose restart redis
sleep 15
redis-cli -h localhost -p 6379 ping
```text

###  Security Incident Response
```bash
# Immediate security lockdown
./scripts/emergency-security-lockdown.sh

# Certificate compromise response
./scripts/emergency-cert-rotation.sh --revoke-all
./scripts/ca/issue-cert.sh api both --emergency

# Suspicious activity investigation
grep -E "(RATE_LIMIT|UNAUTHORIZED|FAILED_AUTH)" logs/audit.log | tail -100
grep -E "suspicious|malicious|attack" logs/security.log | tail -50
```text

###  Performance Issue Resolution
```bash
# High CPU usage investigation
top -p $(pgrep -f "uvicorn\|python")
docker stats --no-stream | grep -E "(api|orchestrator)"

# Memory leak investigation
ps aux --sort=-%mem | head -10
docker stats --format "table {{.Container}}\t{{.MemUsage}}" --no-stream

# Database performance tuning
psql -h localhost -U xorb_user -d xorb -c "SELECT query, mean_exec_time
  FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```text

##  🔄 Backup and Recovery

###  Database Backup Operations
```bash
# Daily database backup
pg_dump -h localhost -U xorb_user -d xorb > backup_$(date +%Y%m%d).sql
gzip backup_$(date +%Y%m%d).sql

# Backup verification
gunzip -c backup_$(date +%Y%m%d).sql.gz | head -50

# Recovery procedure
dropdb -h localhost -U xorb_user xorb_recovery
createdb -h localhost -U xorb_user xorb_recovery
gunzip -c backup_$(date +%Y%m%d).sql.gz | psql -h localhost -U xorb_user -d xorb_recovery
```text

###  Configuration Backup
```bash
# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
  docker-compose*.yml infra/ config/ .env*

# Certificate backup
tar -czf cert_backup_$(date +%Y%m%d).tar.gz secrets/tls/

# Recovery verification
tar -tzf config_backup_$(date +%Y%m%d).tar.gz
tar -tzf cert_backup_$(date +%Y%m%d).tar.gz
```text

##  📈 Capacity Planning

###  Resource Utilization Monitoring
```bash
# CPU utilization trends
sar -u 1 60  # Monitor for 1 minute

# Memory utilization analysis
free -h && cat /proc/meminfo | grep -E "(MemTotal|MemFree|MemAvailable)"

# Disk space monitoring
df -h
du -sh /var/lib/docker/  # Docker storage usage
du -sh logs/  # Log storage usage
```text

###  Scaling Decision Matrix
```bash
# Performance threshold monitoring
avg_response_time=$(curl -s http://localhost:9092/api/v1/query?query=avg_over_time(http_request_duration_seconds[5m]) | jq '.data.result[0].value[1]')
if (( $(echo "$avg_response_time > 0.5" | bc -l) )); then
    echo "Performance degradation detected - consider scaling"
fi

# Active connection monitoring
active_connections=$(curl -s http://localhost:8000/api/v1/metrics | jq '.active_connections')
if [ "$active_connections" -gt 800 ]; then
    echo "High connection count - consider horizontal scaling"
fi
```text

##  🛠️ Maintenance Operations

###  Scheduled Maintenance
```bash
# Weekly maintenance tasks
./scripts/weekly-maintenance.sh

# Database maintenance
psql -h localhost -U xorb_user -d xorb -c "VACUUM ANALYZE;"
psql -h localhost -U xorb_user -d xorb -c "REINDEX DATABASE xorb;"

# Log rotation
logrotate -f /etc/logrotate.d/xorb-platform

# Container image updates
docker-compose pull
docker-compose up -d --force-recreate
```text

###  Security Updates
```bash
# Dependency security scanning
safety check -r requirements.lock

# Container security scanning
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image xorb-api:latest

# Security policy updates
git pull origin main -- policies/
conftest test --policy policies/ .
```text

##  📋 Compliance Operations

###  Audit Trail Management
```bash
# Audit log integrity verification
sha256sum logs/audit.log > logs/audit.log.sha256
gpg --sign logs/audit.log.sha256

# Compliance report generation
./scripts/generate-compliance-report.sh --framework SOC2
./scripts/generate-compliance-report.sh --framework PCI-DSS

# Access review procedures
grep -E "(LOGIN|LOGOUT|ACCESS_GRANTED|ACCESS_DENIED)" logs/audit.log | \
  awk '{print $1, $2, $5, $6}' | sort | uniq -c
```text

###  Data Protection Operations
```bash
# Encryption verification
openssl enc -aes-256-cbc -d -in sensitive_data.enc -out /dev/null
echo $?  # Should return 0 if encryption is valid

# GDPR compliance procedures
./scripts/gdpr-data-export.sh --user-id <user_id>
./scripts/gdpr-data-deletion.sh --user-id <user_id> --confirm
```text

##  🎯 Success Metrics

###  Operational KPIs
- **Uptime**: > 99.9%
- **Response Time**: < 200ms (95th percentile)
- **Error Rate**: < 0.1%
- **Security Incidents**: 0 critical incidents per month
- **Recovery Time**: < 15 minutes for service recovery

###  Monitoring Alerts
- Service health check failures
- Certificate expiry warnings (7 days)
- High resource utilization (CPU > 80%, Memory > 85%)
- Security policy violations
- Backup failures

- --

- This operational runbook consolidates all strategic operational knowledge from XORB platform implementations, providing comprehensive procedures for production operations and incident response.*