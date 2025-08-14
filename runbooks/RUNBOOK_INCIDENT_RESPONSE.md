# XORB Platform - Incident Response Runbook

## Overview
This runbook provides step-by-step procedures for responding to security incidents and service outages in the XORB platform.

## Incident Classification

### Severity Levels
- **Critical (P0)**: Complete service outage, security breach
- **High (P1)**: Major functionality impaired, potential security risk
- **Medium (P2)**: Minor functionality impaired, non-security issue
- **Low (P3)**: Cosmetic issues, documentation updates

## Immediate Response (First 15 minutes)

### 1. Assessment
```bash
# Quick health check
make health

# Check service status
docker-compose ps

# Review recent logs
docker-compose logs --tail=100 api orchestrator
```

### 2. Communication
- Create incident channel: `#incident-YYYY-MM-DD-HHMMSS`
- Notify on-call team
- Update status page if applicable

### 3. Initial Containment
```bash
# If security incident, rotate certificates immediately
make emergency-rotation

# Check for unauthorized access
grep "UNAUTHORIZED\|FAILED_AUTH" /var/log/xorb/* | tail -20

# Review audit logs
docker logs xorb-api-1 | grep AUDIT | tail -50
```

## Security Incident Response

### Potential Breach Indicators
- Unusual authentication patterns
- Unauthorized API calls
- Certificate validation failures
- Suspicious network traffic

### Response Steps

#### 1. Immediate Isolation
```bash
# Stop affected services
docker-compose stop api

# Backup current state
make backup

# Preserve evidence
docker logs xorb-api-1 > incident-logs-$(date +%Y%m%d-%H%M%S).log
```

#### 2. Investigation
```bash
# Check authentication logs
grep -E "(FAILED_LOGIN|INVALID_TOKEN)" /var/log/xorb/auth.log

# Review API access patterns
grep -E "(POST|PUT|DELETE)" /var/log/xorb/api.log | tail -100

# Analyze certificate status
make cert-status
```

#### 3. Containment & Recovery
```bash
# Generate new certificates
make emergency-rotation

# Reset authentication tokens
# (Implementation depends on auth system)

# Restart services with new certificates
make start
```

## Service Outage Response

### Common Issues

#### Database Connectivity
```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready

# Review connection logs
docker logs xorb-postgres-1 | tail -50

# Test connectivity
docker-compose exec api python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://user:pass@postgres:5432/xorb')
    print('DB connected successfully')
    await conn.close()
asyncio.run(test())
"
```

#### NATS Messaging Issues
```bash
# Check NATS status
docker logs xorb-nats-1

# Test connectivity
docker-compose exec api python -c "
import nats
import asyncio
async def test():
    nc = await nats.connect('nats://nats:4222')
    print('NATS connected successfully')
    await nc.close()
asyncio.run(test())
"
```

#### Certificate Expiry
```bash
# Check certificate status
make cert-status

# Rotate expiring certificates
make rotate-certs

# Emergency rotation if needed
make emergency-rotation
```

### Recovery Procedures

#### 1. Service Restart
```bash
# Graceful restart
make restart

# Check health after restart
make health

# Validate functionality
make ptaas-e2e
```

#### 2. Data Recovery
```bash
# Restore from backup if needed
make restore BACKUP_FILE=backups/YYYY-MM-DD/backup.tar.gz

# Verify data integrity
# (Implementation specific to your data validation)
```

## PTaaS-Specific Incidents

### Scanner Failures
```bash
# Check scanner status
docker logs xorb-scanner-1

# Restart scanner service
docker-compose restart scanner

# Test scanner connectivity
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/api/v1/ptaas/profiles
```

### Orchestration Issues
```bash
# Check Temporal workflow status
curl http://localhost:8233/

# Review orchestrator logs
docker logs xorb-orchestrator-1

# Restart orchestration
docker-compose restart orchestrator
```

## Monitoring & Alerting

### Key Metrics to Monitor
- Service response times
- Error rates
- Certificate expiry dates
- Authentication failure rates
- Resource utilization

### Alert Sources
- Prometheus alerts
- Application logs
- Infrastructure monitoring
- External monitoring services

## Post-Incident Procedures

### 1. Documentation
- Update incident log with timeline
- Document root cause analysis
- Identify prevention measures
- Update runbooks if needed

### 2. Review & Improvement
- Conduct post-incident review meeting
- Update monitoring if gaps identified
- Enhance alerting for early detection
- Improve automation for faster response

### 3. Communication
- Send incident summary to stakeholders
- Update security team on any findings
- Notify customers if customer-facing impact

## Emergency Contacts

- **On-call Engineer**: [Contact Information]
- **Security Team**: security@domain.com
- **Infrastructure Team**: [Contact Information]
- **Management**: [Contact Information]

## Tools & Resources

### Useful Commands
```bash
# Repository health check
make doctor

# Security audit
make audit

# Generate reports
make reports

# View all available targets
make help
```

### Log Locations
- Application logs: `/var/log/xorb/`
- Container logs: `docker-compose logs`
- Audit logs: Container stdout with AUDIT prefix
- Access logs: Reverse proxy logs

### Monitoring Dashboards
- Grafana: http://localhost:3010
- Prometheus: http://localhost:9092
- API Health: http://localhost:8000/api/v1/health

## Appendix: Common Scenarios

### Scenario 1: API Unresponsive
1. Check service status: `docker-compose ps`
2. Review API logs: `docker logs xorb-api-1`
3. Restart if needed: `docker-compose restart api`
4. Validate: `curl http://localhost:8000/api/v1/health`

### Scenario 2: Certificate Expired
1. Check status: `make cert-status`
2. Emergency rotation: `make emergency-rotation`
3. Restart services: `make restart`
4. Validate: `make validate`

### Scenario 3: Database Corruption
1. Stop services: `docker-compose stop`
2. Backup current state: `make backup`
3. Restore from backup: `make restore BACKUP_FILE=...`
4. Restart: `make start`
5. Validate: `make ptaas-e2e`
