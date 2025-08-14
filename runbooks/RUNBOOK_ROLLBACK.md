# XORB Platform - Rollback Runbook

## Overview
This runbook provides procedures for rolling back XORB platform deployments and recovering from failed updates.

## Rollback Decision Matrix

### When to Rollback
- **Critical functionality broken**: Core PTaaS operations non-functional
- **Security vulnerability introduced**: New security issues detected
- **Performance degradation**: >50% performance decrease
- **Data corruption**: Database integrity compromised
- **Certificate issues**: TLS/mTLS functionality broken

### When NOT to Rollback
- Minor UI issues (prefer hotfix)
- Documentation updates
- Non-critical feature issues
- Cosmetic problems

## Pre-Rollback Checklist

### 1. Assessment
```bash
# Document current state
make health > rollback-pre-state-$(date +%Y%m%d-%H%M%S).log

# Backup current configuration
make backup

# Check recent changes
git log --oneline -10

# Identify rollback target
git tag --sort=-version:refname | head -10
```

### 2. Notification
- [ ] Create rollback channel: `#rollback-YYYY-MM-DD-HHMMSS`
- [ ] Notify stakeholders
- [ ] Update status page: "Maintenance in progress"
- [ ] Set maintenance mode if available

## Application Rollback

### Method 1: Git-based Rollback (Preferred)

#### 1. Identify Target Version
```bash
# List recent tags
git tag --sort=-version:refname | head -5

# Show differences
git diff v2025.08.1..HEAD --stat

# Choose rollback target
export ROLLBACK_TARGET="v2025.08.1"
```

#### 2. Perform Rollback
```bash
# Create rollback branch
git checkout -b rollback-to-${ROLLBACK_TARGET}-$(date +%Y%m%d-%H%M%S)

# Reset to target version
git reset --hard ${ROLLBACK_TARGET}

# Force push (if needed for deployment)
git push origin main --force-with-lease
```

#### 3. Validate Rollback
```bash
# Verify version
git describe --tags

# Check application starts
make ptaas-quickstart

# Run health checks
make health

# Test core functionality
make ptaas-e2e
```

### Method 2: Container-based Rollback

#### 1. Identify Previous Images
```bash
# List recent images
docker images | grep xorb | head -10

# Or use registry tags
docker pull registry.example.com/xorb:v2025.08.1
```

#### 2. Update Docker Compose
```bash
# Edit docker-compose.production.yml
# Change image tags to previous version
sed -i 's/:latest/:v2025.08.1/g' docker-compose.production.yml
```

#### 3. Deploy Previous Version
```bash
# Stop current version
docker-compose down

# Pull previous images
docker-compose pull

# Start previous version
docker-compose up -d

# Verify deployment
make health
```

## Database Rollback

### PostgreSQL Schema Rollback

#### 1. Backup Current State
```bash
# Create full backup
docker-compose exec postgres pg_dump -U xorb_user xorb_db > \
  rollback-backup-$(date +%Y%m%d-%H%M%S).sql

# Create schema dump
docker-compose exec postgres pg_dump -U xorb_user -s xorb_db > \
  schema-backup-$(date +%Y%m%d-%H%M%S).sql
```

#### 2. Restore Previous Schema
```bash
# Identify target backup
ls -la backups/*/database-*.sql

# Stop services accessing database
docker-compose stop api orchestrator

# Restore from backup
docker-compose exec -T postgres psql -U xorb_user -d xorb_db < \
  backups/2025-08-10/database-backup.sql
```

#### 3. Validate Database
```bash
# Check schema version
docker-compose exec postgres psql -U xorb_user -d xorb_db -c \
  "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"

# Test connectivity
docker-compose exec api python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://xorb_user:password@postgres:5432/xorb_db')
    result = await conn.fetchval('SELECT COUNT(*) FROM information_schema.tables')
    print(f'Tables count: {result}')
    await conn.close()
asyncio.run(test())
"
```

## Configuration Rollback

### Certificate Rollback
```bash
# Backup current certificates
cp -r secrets/tls/ secrets/tls-backup-$(date +%Y%m%d-%H%M%S)

# Restore from backup
make restore BACKUP_FILE=backups/2025-08-10/certificates-backup.tar.gz

# Restart services with old certificates
make restart

# Validate TLS
make validate-tls
```

### Environment Configuration
```bash
# Backup current configuration
cp .env .env.backup-$(date +%Y%m%d-%H%M%S)

# Restore previous configuration
cp backups/2025-08-10/.env.backup .env

# Restart with new configuration
docker-compose down
docker-compose up -d
```

## Infrastructure Rollback

### Docker Compose Rollback
```bash
# Backup current compose files
cp docker-compose.production.yml docker-compose.production.yml.backup

# Restore previous version
git checkout ${ROLLBACK_TARGET} -- docker-compose.production.yml

# Redeploy infrastructure
docker-compose down
docker-compose up -d
```

### Kubernetes Rollback (if applicable)
```bash
# Check rollout history
kubectl rollout history deployment/xorb-api

# Rollback to previous version
kubectl rollout undo deployment/xorb-api

# Check rollback status
kubectl rollout status deployment/xorb-api
```

## Monitoring & Verification

### Post-Rollback Validation

#### 1. Service Health
```bash
# Comprehensive health check
make health

# API responsiveness
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/api/v1/health

# PTaaS functionality
make ptaas-e2e
```

#### 2. Performance Verification
```bash
# Run performance benchmarks
make performance

# Check response times
curl -w "@curl-format.txt" -s -o /dev/null \
  http://localhost:8000/api/v1/health
```

#### 3. Security Validation
```bash
# Certificate validation
make validate-tls

# Security scan
make security-scan

# Audit log verification
docker logs xorb-api-1 | grep AUDIT | tail -10
```

### Monitoring Checklist
- [ ] All services running and healthy
- [ ] Database connectivity restored
- [ ] API endpoints responding
- [ ] PTaaS scans completing successfully
- [ ] Certificates valid and not expired
- [ ] Monitoring alerts cleared
- [ ] Performance metrics within acceptable range

## Recovery from Failed Rollback

### If Rollback Fails

#### 1. Emergency Recovery
```bash
# Stop all services
docker-compose down

# Restore from known-good backup
make restore BACKUP_FILE=backups/emergency/last-known-good.tar.gz

# Start in minimal mode
docker-compose up -d postgres nats

# Gradually start other services
docker-compose up -d api
sleep 30
docker-compose up -d orchestrator
```

#### 2. Manual Recovery
```bash
# Reset to known-good state
git reset --hard ${KNOWN_GOOD_COMMIT}

# Clean docker state
docker system prune -f
docker volume prune -f

# Rebuild from scratch
make clean-all
make quick-start
```

## Communication Plan

### During Rollback
1. **Start notification**: "Rollback in progress, services may be temporarily unavailable"
2. **Progress updates**: Every 15 minutes during rollback
3. **Completion notification**: "Rollback completed, services restored"

### Post-Rollback
1. **Summary report**: What was rolled back and why
2. **Impact assessment**: What functionality was affected
3. **Next steps**: Plans for re-deployment or fixes

## Prevention & Improvement

### Post-Rollback Actions

#### 1. Root Cause Analysis
- Document what caused the need for rollback
- Identify prevention measures
- Update deployment procedures if needed

#### 2. Improve Monitoring
- Add alerts for early detection of issues
- Enhance health checks
- Improve deployment validation

#### 3. Update Procedures
- Update rollback procedures based on experience
- Document any new commands or tools used
- Train team on improved procedures

## Testing Rollback Procedures

### Regular Rollback Drills
```bash
# Monthly rollback simulation
# 1. Deploy to staging environment
# 2. Introduce controlled failure
# 3. Execute rollback procedure
# 4. Validate recovery
# 5. Document lessons learned
```

### Automated Rollback Testing
```bash
# Create test script for rollback validation
#!/bin/bash
# test-rollback.sh

echo "Testing rollback procedures..."

# Deploy test version
git tag test-rollback-$(date +%s)
make deploy-dev

# Simulate failure and rollback
make rollback

# Validate recovery
make ptaas-e2e

echo "Rollback test completed"
```

## Emergency Contacts

- **On-call Engineer**: [Contact Information]
- **Database Administrator**: [Contact Information]
- **Infrastructure Team**: [Contact Information]
- **Security Team**: security@domain.com

## Appendix: Recovery Commands

### Quick Reference
```bash
# Health check
make health

# Full backup
make backup

# Emergency certificate rotation
make emergency-rotation

# Service restart
make restart

# Validation suite
make ptaas-e2e

# Performance test
make performance

# Security scan
make security-scan
```

### Log Analysis
```bash
# Recent errors
docker-compose logs --since="1h" | grep -i error

# Authentication issues
grep "AUTH" /var/log/xorb/* | tail -20

# Database connections
docker logs xorb-postgres-1 | grep -i connection | tail -10
```
