- --
title: "XORB Platform TLS/mTLS Quick Start Guide"
description: "Get the XORB Platform TLS/mTLS security stack up and running in minutes"
category: "Getting Started"
tags: ["quickstart", "tls", "mtls", "setup", "deployment"]
last_updated: "2025-01-11"
author: "XORB Platform Team"
difficulty: "Beginner"
estimated_time: "5-15 minutes"
- --

# 🚀 XORB Platform TLS/mTLS Quick Start Guide

Get the XORB Platform TLS/mTLS security stack up and running in minutes with this comprehensive quick start guide.

##  ⚡ 5-Minute Quick Start

###  Prerequisites
- Docker & Docker Compose 20.10+
- OpenSSL 1.1.1+
- Make (optional, for convenience commands)

###  1. Clone and Initialize

```bash
# Clone the repository
git clone <repository-url>
cd xorb-platform

# Initialize the Certificate Authority
./scripts/ca/make-ca.sh
```

###  2. Generate Essential Certificates

```bash
# Using Make (recommended)
make quick-start

# Or manually
./scripts/ca/issue-cert.sh api both
./scripts/ca/issue-cert.sh redis server
./scripts/ca/issue-cert.sh redis-client client
```

###  3. Deploy with TLS

```bash
# Deploy the full TLS stack
make deploy-tls

# Or manually
docker-compose -f infra/docker-compose.tls.yml up -d
```

###  4. Validate Security

```bash
# Run comprehensive validation
make validate

# Or run specific tests
./scripts/validate/test_tls.sh
./scripts/validate/test_mtls.sh
```

###  5. Verify Services

```bash
# Check service health
make health

# Test API endpoint
curl --cacert secrets/tls/ca/ca.pem \
     --cert secrets/tls/api-client/cert.pem \
     --key secrets/tls/api-client/key.pem \
     https://envoy-api:8443/api/v1/health
```

##  🎯 Development Setup

###  Quick Development Environment

```bash
# Setup for development
make dev-setup

# Start minimal services
docker-compose -f docker-compose.development.yml up -d

# Run development tests
make dev-test
```

###  Local Testing

```bash
# Test individual components
make validate-tls     # TLS protocols and ciphers
make validate-mtls    # Mutual TLS authentication
make validate-redis   # Redis TLS configuration
make validate-docker  # Docker-in-Docker TLS
```

##  🏭 Production Deployment

###  Full Production Stack

```bash
# Generate all certificates
make certs-generate

# Deploy production environment
make deploy-prod

# Run comprehensive validation
make validate

# Run security audit
make audit
```

###  Production Checklist

- [ ] Certificate Authority initialized
- [ ] All service certificates generated
- [ ] TLS/mTLS validation passed
- [ ] Security policies validated
- [ ] Performance benchmarks completed
- [ ] Monitoring and alerting configured
- [ ] Certificate rotation scheduled

##  📋 Essential Commands

###  Certificate Management

```bash
# Initialize CA
make ca-init

# Generate specific service certificate
make cert-api
make cert-redis
make cert-postgres

# Rotate certificates
make rotate-certs

# Emergency rotation
make emergency-rotation
```

###  Service Operations

```bash
# Start services
make start

# Stop services
make stop

# Restart services
make restart

# Check health
make health
```

###  Validation and Testing

```bash
# Comprehensive validation
make validate

# Security policy check
make security-scan

# Performance benchmarks
make performance

# Generate reports
make reports
```

##  🔧 Configuration Files

###  Key Configuration Files

- **`infra/docker-compose.tls.yml`** - Main TLS deployment
- **`envoy/api.envoy.yaml`** - API proxy configuration
- **`infra/redis/redis-tls.conf`** - Redis TLS settings
- **`policies/tls-security.rego`** - Security policies

###  Environment Variables

```bash
# Set in your environment or .env file
export TLS_ENABLED=true
export REDIS_URL=rediss://redis:6379
export REDIS_TLS_CERT_FILE=/run/tls/redis-client/cert.pem
export REDIS_TLS_KEY_FILE=/run/tls/redis-client/key.pem
export REDIS_TLS_CA_FILE=/run/tls/ca/ca.pem
```

##  🧪 Testing Examples

###  Manual TLS Testing

```bash
# Test TLS connection
openssl s_client -connect envoy-api:8443 \
  -CAfile secrets/tls/ca/ca.pem \
  -cert secrets/tls/api-client/cert.pem \
  -key secrets/tls/api-client/key.pem

# Test Redis TLS
redis-cli --tls \
  --cert secrets/tls/redis-client/cert.pem \
  --key secrets/tls/redis-client/key.pem \
  --cacert secrets/tls/ca/ca.pem \
  -h redis -p 6379 ping

# Test HTTP endpoint
curl -v --cacert secrets/tls/ca/ca.pem \
  --cert secrets/tls/api-client/cert.pem \
  --key secrets/tls/api-client/key.pem \
  https://envoy-api:8443/api/v1/health
```

###  Automated Testing

```bash
# Run all validation scripts
./scripts/validate/test_comprehensive.sh

# Run performance benchmarks
./scripts/performance-benchmark.sh

# Monitor health continuously
./scripts/health-monitor.sh daemon
```

##  📊 Monitoring and Observability

###  Health Monitoring

```bash
# Start health monitoring daemon
./scripts/health-monitor.sh daemon

# Generate health report
./scripts/health-monitor.sh report

# Check certificate status
make cert-status
```

###  Metrics and Dashboards

- **Prometheus**: http://localhost:9093 (with TLS)
- **Grafana**: http://localhost:3001 (with TLS)
- **Envoy Admin**: http://localhost:9901

###  Log Locations

```bash
# Application logs
docker-compose logs -f

# Security logs
tail -f logs/security/*.log

# Certificate rotation logs
tail -f logs/cert-rotation/*.log
```

##  🚨 Troubleshooting

###  Common Issues

####  Certificate Problems

```bash
# Check certificate validity
openssl x509 -in secrets/tls/api/cert.pem -noout -dates

# Verify certificate chain
openssl verify -CAfile secrets/tls/ca/ca.pem secrets/tls/api/cert.pem

# Regenerate certificate
./scripts/ca/issue-cert.sh api both
```

####  Connection Issues

```bash
# Test basic connectivity
nc -zv envoy-api 8443

# Check TLS handshake
openssl s_client -connect envoy-api:8443 -verify_return_error

# Debug Envoy proxy
curl http://localhost:9901/config_dump
```

####  Service Health

```bash
# Check container status
docker-compose ps

# View service logs
docker-compose logs api
docker-compose logs envoy-api

# Check resource usage
docker stats
```

###  Debug Mode

```bash
# Enable verbose logging
export VERBOSE=true

# Run with debug output
./scripts/validate/test_tls.sh -v
./scripts/rotate-certs.sh -v
```

##  🔗 Next Steps

###  Advanced Configuration

1. **Kubernetes Deployment**: See [k8s/mtls/](../k8s/mtls/) for Kubernetes manifests
2. **Custom Security Policies**: Modify [policies/tls-security.rego](../policies/tls-security.rego)
3. **Performance Tuning**: Review [performance benchmarks](../scripts/performance-benchmark.sh)
4. **Monitoring Integration**: Configure Slack/email alerts in health monitor

###  Security Hardening

1. **Certificate Rotation**: Set up automated rotation schedule
2. **Secret Management**: Integrate with HashiCorp Vault
3. **Network Policies**: Implement additional network segmentation
4. **Audit Logging**: Configure centralized security logging

###  Operational Excellence

1. **Backup Strategy**: Implement certificate backup procedures
2. **Incident Response**: Prepare emergency rotation procedures
3. **Documentation**: Maintain security documentation
4. **Training**: Train team on TLS operations

##  📚 Documentation Links

- [Complete Implementation Guide](TLS_IMPLEMENTATION_GUIDE.md)
- [Security Documentation](SECURITY.md)
- [TLS Operational Runbook](TLS_OPERATIONAL_RUNBOOK.md)
- [TLS Deployment Guide](TLS_DEPLOYMENT_COMPLETION_REPORT.md)

##  🆘 Support

- **Security Issues**: security@xorb.local
- **Technical Support**: Create an issue in the repository
- **Emergency**: Follow incident response procedures

- --

- **⚠️ Security Notice**: This quick start guide provides a secure TLS/mTLS implementation. Ensure you understand the security implications and follow your organization's security policies.

- **🎯 Success Criteria**: After completing this guide, you should have a fully functional TLS/mTLS secured XORB Platform with automated certificate management and comprehensive monitoring.