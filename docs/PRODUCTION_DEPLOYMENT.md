# XORB Production Deployment Guide

##  Overview

This document provides comprehensive instructions for deploying XORB to production following security best practices and performance optimizations based on the pre-flight analysis.

##  ⚠️ Critical Security Requirements

- *BEFORE DEPLOYMENT** - Ensure these security requirements are met:

###  1. Environment Variables (MANDATORY)

Create `/root/Xorb/.env` with strong, unique values:

```bash
# Generate strong secrets:
JWT_SECRET=$(openssl rand -base64 64)
XORB_API_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Required configuration:
ENVIRONMENT=production
DATABASE_URL=postgresql://xorb:${POSTGRES_PASSWORD}@postgres:5432/xorb_db
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
ALLOWED_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
ENABLE_RATE_LIMITING=true
ENABLE_AUDIT_LOGGING=true
ENABLE_MFA=true
LOG_LEVEL=INFO
```text

###  2. SSL Certificates

Ensure SSL certificates are properly configured:
- Private keys: `/root/Xorb/ssl/*.key` (permissions: 600)
- Certificates: `/root/Xorb/ssl/*.crt` (permissions: 644)

##  🚀 Deployment Process

###  Phase 1: Pre-Deployment Validation

```bash
# 1. Set secure environment variables
cp /root/Xorb/.env.example /root/Xorb/.env
# Edit .env with strong values (see above)

# 2. Set file permissions
chmod 600 /root/Xorb/.env
find /root/Xorb/ssl -name "*.key" -exec chmod 600 {} \;
find /root/Xorb/ssl -name "*.crt" -exec chmod 644 {} \;

# 3. Validate environment
source /root/Xorb/.env
if [[ ${#JWT_SECRET} -lt 32 ]]; then echo "ERROR: JWT_SECRET too weak"; exit 1; fi
if [[ -z "$XORB_API_KEY" ]]; then echo "ERROR: XORB_API_KEY not set"; exit 1; fi
```text

###  Phase 2: Automated Deployment

```bash
# Run the production deployment script
cd /root/Xorb
sudo ./deploy-production.sh

# This script will:
# ✓ Validate environment and system resources
# ✓ Create backup of current deployment
# ✓ Apply security hardening
# ✓ Set up database with migrations
# ✓ Deploy all services with production configuration
# ✓ Configure performance optimizations
# ✓ Set up monitoring and alerting
# ✓ Run comprehensive health checks
# ✓ Generate deployment report
```text

###  Phase 3: Post-Deployment Validation

```bash
# Run comprehensive validation tests
./validate-production.sh

# Expected output:
# ✓ Environment validation (JWT secrets, system resources)
# ✓ Security validation (file permissions, container security)
# ✓ Service validation (all containers healthy)
# ✓ API validation (endpoints responding correctly)
# ✓ Performance validation (response times, resource usage)
# ✓ Integration validation (database, Redis, service communication)
# ✓ SSL/TLS validation (certificate validity)
# ✓ Monitoring validation (Prometheus, Grafana)
# ✓ Load testing (concurrent request handling)
```text

##  🔧 Manual Deployment (Alternative)

If you prefer manual control, follow these steps:

###  1. Database Setup

```bash
# Start PostgreSQL
docker-compose -f infra/docker-compose.production.yml up -d postgres

# Wait for database readiness
until docker exec xorb-postgres pg_isready -U xorb; do sleep 2; done

# Run migrations
cd src/api
python -m alembic upgrade head
```text

###  2. Service Deployment

```bash
# Build and start all services
cd /root/Xorb
ENVIRONMENT=production docker-compose -f infra/docker-compose.production.yml up -d

# Verify all services are running
docker-compose -f infra/docker-compose.production.yml ps
```text

###  3. Health Verification

```bash
# Check API health
curl -f http://localhost:8080/api/health
curl -f http://localhost:8000/health

# Check Orchestrator health
curl -f http://localhost:8080/health

# Verify rate limiting
curl -I http://localhost:8080/api/health | grep X-RateLimit
```text

##  📊 Service Architecture

###  Production Services

| Service | Port | Purpose | Resources |
|---------|------|---------|-----------|
| **API** | 8080 | Main REST API | 2 CPU, 4GB RAM |
| **Legacy API** | 8000 | Legacy endpoints | 2 CPU, 2GB RAM |
| **Orchestrator** | 8080 | Workflow engine | 1 CPU, 2GB RAM |
| **PostgreSQL** | 5432 | Primary database | 4 CPU, 4GB RAM |
| **Redis** | 6379 | Cache & sessions | 1 CPU, 1GB RAM |
| **Temporal** | 7233 | Workflow server | 2 CPU, 2GB RAM |
| **Prometheus** | 9090 | Metrics collection | 1 CPU, 1GB RAM |
| **Grafana** | 3000 | Monitoring dashboards | 0.5 CPU, 512MB RAM |

###  Security Features

- ✅ **JWT Authentication** with strong secrets (>64 chars)
- ✅ **API Key Protection** with brute force prevention
- ✅ **Rate Limiting** with adaptive scaling and penalties
- ✅ **CORS Protection** with restrictive origin policies
- ✅ **Input Validation** with safe JSON parsing (no eval())
- ✅ **Container Security** with no-new-privileges and read-only filesystem
- ✅ **SSL/TLS** with strong cipher suites
- ✅ **Audit Logging** for all security events
- ✅ **Resource Limits** to prevent DoS attacks

##  🔒 Security Configuration

###  Rate Limiting Rules (Production)

```yaml
# Global limits
- 1000 requests/second (sliding window)
- 100 requests/minute per IP (token bucket)

# Authentication endpoints
- 5 requests/minute per IP (leaky bucket)
- 15-minute penalty on violations

# API key protection
- 50 requests/hour per IP without valid key
- 30-minute penalty on violations

# Heavy computation endpoints
- 50 requests/hour per user (token bucket)
- 5 burst requests allowed
```text

###  Environment-Based Security

```bash
# Production-only security features
ENABLE_RATE_LIMITING=true      # Strict rate limiting
ENABLE_AUDIT_LOGGING=true      # Full audit trail
ENABLE_MFA=true               # Multi-factor authentication
ALLOWED_ORIGINS=https://...   # Restrictive CORS
LOG_LEVEL=INFO                # Minimal logging
```text

##  📈 Performance Optimizations

###  Database Configuration

```sql
- - PostgreSQL optimizations applied automatically
shared_buffers = '256MB'
effective_cache_size = '1GB'
maintenance_work_mem = '64MB'
checkpoint_completion_target = 0.9
wal_buffers = '16MB'
max_connections = 200
```text

###  Container Resource Limits

```yaml
api:
  resources:
    limits: { cpus: '2.0', memory: '4G' }
    reservations: { cpus: '1.0', memory: '2G' }

orchestrator:
  resources:
    limits: { cpus: '1.0', memory: '2G' }
    reservations: { cpus: '0.5', memory: '1G' }
```text

###  Kernel Optimizations

```bash
# Applied automatically by deployment script
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
```text

##  🔍 Monitoring & Alerting

###  Prometheus Metrics

- **Service Health**: `/api/health`, `/health` endpoints
- **Response Time**: Request duration histograms
- **Rate Limiting**: Violation counts and success rates
- **Resource Usage**: CPU, memory, disk per container
- **Database**: Connection pool, query performance
- **Security**: Failed authentication attempts, suspicious activity

###  Grafana Dashboards

Access dashboards at `http://localhost:3000`:

1. **Service Overview**: All service health and performance
2. **Security Monitor**: Rate limiting, auth failures, violations
3. **Database Performance**: Query times, connection pools
4. **System Resources**: CPU, memory, disk, network
5. **API Analytics**: Endpoint usage, response times, errors

###  Log Monitoring

```bash
# View real-time logs
docker-compose -f infra/docker-compose.production.yml logs -f

# Security events
docker-compose logs api | grep -E "(ERROR|SECURITY|VIOLATION)"

# Performance issues
docker-compose logs api | grep -E "(SLOW|TIMEOUT|HIGH_LOAD)"
```text

##  🔄 Maintenance & Updates

###  Regular Maintenance Tasks

```bash
# Weekly: Rotate logs
docker-compose exec api logrotate /etc/logrotate.d/xorb

# Weekly: Database maintenance
docker exec xorb-postgres psql -U xorb -c "VACUUM ANALYZE;"

# Monthly: Update container images
docker-compose -f infra/docker-compose.production.yml pull
docker-compose -f infra/docker-compose.production.yml up -d

# Monthly: Security scan
docker run --rm -v /root/Xorb:/scan clair-scanner scan /scan
```text

###  Backup Strategy

```bash
# Automated daily backups (configured by deployment script)
# Database backups: /root/Xorb/backups/db/
# Application backups: /root/Xorb/backups/
# SSL certificates: Backed up with application

# Manual backup
tar -czf xorb-backup-$(date +%Y%m%d).tar.gz \
  --exclude="logs" --exclude="__pycache__" /root/Xorb
```text

##  ⚠️ Troubleshooting

###  Common Issues

- *1. Services Won't Start**
```bash
# Check environment variables
source /root/Xorb/.env && env | grep -E "(JWT|API_KEY|PASSWORD)"

# Check logs
docker-compose -f infra/docker-compose.production.yml logs api
```text

- *2. Authentication Failures**
```bash
# Verify JWT secret is set and strong
echo "JWT length: ${#JWT_SECRET}"
[[ ${#JWT_SECRET} -gt 32 ]] && echo "Strong" || echo "Weak"

# Check API key configuration
curl -H "X-API-Key: $XORB_API_KEY" http://localhost:8080/api/health
```text

- *3. Rate Limiting Issues**
```bash
# Check Redis connection
docker exec redis redis-cli ping

# View rate limit violations
curl -s http://localhost:9090/api/v1/query?query=rate_limit_violations_total
```text

- *4. Database Connection Issues**
```bash
# Test database connectivity
docker exec xorb-postgres pg_isready -U xorb -d xorb_db

# Check connection string
echo $DATABASE_URL | grep -v password
```text

###  Emergency Procedures

- *Rollback Deployment**
```bash
# Automatic rollback (if enabled during deployment)
# The deployment script creates backups and can auto-rollback on failure

# Manual rollback
cd /root/Xorb/backups
tar -xzf $(ls -t xorb-backup-*.tar.gz | head -1) -C /
docker-compose -f infra/docker-compose.production.yml restart
```text

- *Security Incident Response**
```bash
# Enable global circuit breaker (stops all requests)
curl -X POST http://localhost:8080/api/admin/circuit-breaker/enable

# Check security violations
docker-compose logs api | grep -i "security\|violation\|attack"

# Temporary IP blocking (if supported by infrastructure)
iptables -A INPUT -s <suspicious-ip> -j DROP
```text

##  ✅ Go-Live Checklist

###  Pre-Go-Live (Required)

- [ ] **Environment variables configured** with strong secrets
- [ ] **SSL certificates installed** and valid for >30 days
- [ ] **File permissions secured** (.env = 600, SSL keys = 600)
- [ ] **All services healthy** (green status in health checks)
- [ ] **Database migrations applied** and tested
- [ ] **Rate limiting functional** (test with multiple requests)
- [ ] **Authentication working** (test API key and JWT)
- [ ] **Monitoring operational** (Prometheus + Grafana accessible)
- [ ] **Backup strategy confirmed** (automated backups configured)
- [ ] **Security scan passed** (no critical vulnerabilities)

###  Post-Go-Live (First 24 Hours)

- [ ] **Monitor service logs** for errors or warnings
- [ ] **Track performance metrics** (response times, resource usage)
- [ ] **Verify rate limiting** is blocking excessive requests
- [ ] **Check security alerts** for suspicious activity
- [ ] **Confirm backup creation** (first backup completed successfully)
- [ ] **Test failover procedures** (database, Redis connectivity)
- [ ] **Monitor SSL certificate** status and expiration alerts

##  📞 Support & Maintenance

###  Key Files & Locations

```text
/root/Xorb/
├── .env                          # Environment configuration (secure)
├── deploy-production.sh          # Automated deployment script
├── validate-production.sh        # Validation and testing script
├── infra/docker-compose.production.yml  # Production services
├── ssl/                          # SSL certificates and keys
├── logs/                         # Application logs
├── backups/                      # Automated backups
└── deployment-report-*.json      # Deployment reports
```text

###  Important Commands

```bash
# Full deployment
./deploy-production.sh

# Validation testing
./validate-production.sh

# Service status
docker-compose -f infra/docker-compose.production.yml ps

# View logs
docker-compose -f infra/docker-compose.production.yml logs -f [service]

# Restart services
docker-compose -f infra/docker-compose.production.yml restart [service]

# Emergency stop
docker-compose -f infra/docker-compose.production.yml down
```text

- --

##  🎯 Production Deployment Summary

- *XORB is now production-ready with:**

✅ **Security Hardened** - Strong secrets, rate limiting, input validation
✅ **Performance Optimized** - Resource limits, database tuning, async operations
✅ **Monitoring Enabled** - Prometheus metrics, Grafana dashboards, health checks
✅ **Backup Strategy** - Automated daily backups with retention policy
✅ **Documentation Complete** - Deployment, validation, and troubleshooting guides

- *Next Steps:**
1. Run `./deploy-production.sh` for automated deployment
2. Run `./validate-production.sh` to confirm everything is working
3. Monitor the first 24 hours closely using Grafana dashboards
4. Set up alerting for critical metrics and security events

- *🚀 XORB is ready for production deployment!**