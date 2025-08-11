#  XORB Platform TLS/mTLS Implementation Guide

##  Overview

This guide provides comprehensive instructions for implementing and managing end-to-end TLS/mTLS security across the XORB Platform. All internal services communicate using mutual TLS (mTLS) for authentication and encryption, while external connections use standard TLS with HSTS.

##  🔐 Security Architecture

###  Transport Security Model

```
┌─────────────────┐    HTTPS/TLS    ┌─────────────────┐
│   External      │◄──────────────►│   Envoy Proxy   │
│   Clients       │                 │   (TLS Term)    │
└─────────────────┘                 └─────────────────┘
                                            │ mTLS
                                            ▼
┌─────────────────┐    mTLS         ┌─────────────────┐
│   API Service   │◄──────────────►│  Orchestrator   │
│  (FastAPI)      │                 │   Service       │
└─────────────────┘                 └─────────────────┘
         │ mTLS                             │ mTLS
         ▼                                  ▼
┌─────────────────┐    mTLS         ┌─────────────────┐
│   Redis         │◄──────────────►│   PTaaS Agent   │
│  (TLS-only)     │                 │   Services      │
└─────────────────┘                 └─────────────────┘
         │ mTLS                             │ mTLS
         ▼                                  ▼
┌─────────────────┐    TLS          ┌─────────────────┐
│   PostgreSQL    │◄──────────────►│   Docker-in-    │
│  (TLS enabled)  │                 │   Docker (TLS)  │
└─────────────────┘                 └─────────────────┘
```

###  Certificate Hierarchy

```
XORB Root CA (10 years)
└── XORB Intermediate CA (5 years)
    ├── Service Certificates (30 days)
    │   ├── api.xorb.local
    │   ├── orchestrator.xorb.local
    │   ├── agent.xorb.local
    │   ├── redis.xorb.local
    │   └── ...
    └── Client Certificates (30 days)
        ├── orchestrator-client.xorb.local
        ├── agent-client.xorb.local
        ├── scanner-client.xorb.local
        └── ...
```

##  🚀 Quick Start

###  1. Certificate Authority Setup

Initialize the certificate authority and generate service certificates:

```bash
#  Create CA infrastructure
./scripts/ca/make-ca.sh

#  Generate certificates for all services
./scripts/ca/issue-cert.sh api both
./scripts/ca/issue-cert.sh orchestrator client
./scripts/ca/issue-cert.sh agent both
./scripts/ca/issue-cert.sh redis server
./scripts/ca/issue-cert.sh redis-client client
```

###  2. Docker Compose Deployment

Deploy with full TLS/mTLS configuration:

```bash
#  Start services with TLS configuration
docker-compose -f infra/docker-compose.tls.yml up -d

#  Verify service health
docker-compose -f infra/docker-compose.tls.yml ps
```

###  3. Validation

Run comprehensive TLS validation:

```bash
#  Test all TLS configurations
./scripts/validate/test_tls.sh

#  Test mTLS authentication
./scripts/validate/test_mtls.sh

#  Test Redis TLS configuration
./scripts/validate/test_redis_tls.sh

#  Test Docker-in-Docker TLS
./scripts/validate/test_dind_tls.sh
```

##  📋 Implementation Details

###  Docker Compose Configuration

The `infra/docker-compose.tls.yml` provides:

- **TLS-only services**: All services configured for encrypted communication
- **mTLS enforcement**: Client certificate verification required
- **Envoy sidecars**: TLS termination and certificate validation
- **Secure networking**: Isolated network with proper firewall rules

Key services configuration:

```yaml
#  Redis with TLS-only
redis:
  command: redis-server /etc/redis/redis.conf
  volumes:
    - ./infra/redis/redis-tls.conf:/etc/redis/redis.conf:ro
    - ./secrets/tls/redis:/run/tls/redis:ro

#  API service behind Envoy proxy
envoy-api:
  image: envoyproxy/envoy:v1.28-latest
  volumes:
    - ./envoy/api.envoy.yaml:/etc/envoy/envoy.yaml:ro
    - ./secrets/tls/api:/run/tls/api:ro
```

###  Envoy Proxy Configuration

Envoy proxies provide mTLS termination with:

- **Client certificate requirement**: `require_client_certificate: true`
- **TLS 1.2+ enforcement**: Minimum protocol version TLSv1.2
- **Secure cipher suites**: ECDHE with AES/ChaCha20
- **Certificate validation**: Full chain verification
- **RBAC policies**: Service-based authorization

Example configuration:

```yaml
transport_socket:
  name: envoy.transport_sockets.tls
  typed_config:
    require_client_certificate: true
    common_tls_context:
      validation_context:
        trusted_ca:
          filename: "/run/tls/ca/ca.pem"
    tls_params:
      tls_minimum_protocol_version: TLSv1_2
      tls_maximum_protocol_version: TLSv1_3
```

###  Redis TLS Configuration

Redis operates in TLS-only mode:

```conf
#  Disable plaintext
port 0

#  Enable TLS
tls-port 6379
tls-cert-file /run/tls/redis/cert.pem
tls-key-file /run/tls/redis/key.pem
tls-ca-cert-file /run/tls/ca/ca.pem

#  Require client certificates
tls-auth-clients yes
tls-protocols "TLSv1.2 TLSv1.3"
```

###  Docker-in-Docker TLS

Docker daemon configured for TLS-only access:

```bash
dockerd \
  --host=0.0.0.0:2376 \
  --tls=true \
  --tlscert=/certs/server/cert.pem \
  --tlskey=/certs/server/key.pem \
  --tlsverify=true \
  --tlscacert=/certs/ca/ca.pem
```

##  🔄 Certificate Management

###  Certificate Lifecycle

1. **Generation**: 30-day validity for short-lived certificates
2. **Distribution**: Secure volume mounts with proper permissions
3. **Rotation**: Automated renewal before expiration
4. **Validation**: Continuous monitoring and testing

###  Automated Rotation

```bash
#  Manual certificate rotation
./scripts/rotate-certs.sh

#  Rotate specific service
./scripts/rotate-certs.sh -s api

#  Force rotation (ignore expiry)
./scripts/rotate-certs.sh -f

#  Dry run to preview changes
./scripts/rotate-certs.sh -d
```

###  Certificate Monitoring

- **Expiry alerts**: 7-day warning threshold
- **Validation checks**: Daily certificate verification
- **Backup retention**: 30-day backup history
- **Audit logging**: All certificate operations logged

##  🧪 Testing and Validation

###  Comprehensive Test Suite

The validation scripts provide thorough testing:

####  TLS Configuration Test (`test_tls.sh`)
- Protocol version validation (TLS 1.2/1.3 only)
- Cipher suite verification (secure ciphers only)
- Certificate chain validation
- HSTS header verification
- Weak protocol detection

####  mTLS Authentication Test (`test_mtls.sh`)
- Client certificate requirement
- Valid certificate authentication
- Invalid certificate rejection
- Wrong CA certificate rejection
- Authorization policy testing

####  Service-Specific Tests
- **Redis**: TLS-only operation, client certificate auth
- **Docker**: TLS daemon access, security scan execution
- **API**: HTTPS endpoints, certificate headers
- **Agent**: Service mesh communication

###  Test Execution

```bash
#  Run all validation tests
./scripts/validate/test_tls.sh
./scripts/validate/test_mtls.sh
./scripts/validate/test_redis_tls.sh
./scripts/validate/test_dind_tls.sh

#  Generate HTML reports
ls reports/*/summary.html
```

##  🏗️ Kubernetes Deployment

###  cert-manager Integration

Deploy with automatic certificate management:

```bash
#  Install cert-manager
kubectl apply -f k8s/mtls/namespace.yaml
kubectl apply -f k8s/mtls/cluster-issuer.yaml
kubectl apply -f k8s/mtls/service-certificates.yaml
```

###  Istio Service Mesh

Enable strict mTLS across the mesh:

```bash
#  Apply Istio policies
kubectl apply -f k8s/mtls/istio-mtls-policy.yaml

#  Verify mTLS status
istioctl authn tls-check
```

##  🔒 Security Policies

###  OPA/Conftest Validation

Automated policy enforcement:

```bash
#  Test configurations against policies
conftest test --policy policies/tls-security.rego infra/docker-compose.tls.yml
conftest test --policy policies/tls-security.rego envoy/*.yaml
conftest test --policy policies/tls-security.rego k8s/mtls/*.yaml
```

###  Policy Rules

The security policies enforce:

- **No plaintext protocols**: All communication must be encrypted
- **mTLS requirement**: Client certificates mandatory for internal services
- **Secure TLS versions**: TLS 1.2+ only, no legacy protocols
- **Strong cipher suites**: ECDHE with AES/ChaCha20 only
- **Certificate constraints**: Maximum 30-day validity
- **Container security**: Non-root users, resource limits

##  📊 Monitoring and Observability

###  Metrics Collection

TLS/mTLS metrics are collected via:

- **Certificate expiry**: Days until expiration
- **TLS handshake success/failure**: Connection metrics
- **Cipher suite usage**: Security compliance tracking
- **Certificate validation errors**: Security incident detection

###  Alerting

Automated alerts for:

- Certificates expiring within 7 days
- TLS handshake failures
- Invalid certificate attempts
- Policy violations

###  Dashboard

Grafana dashboards show:

- Certificate inventory and expiry
- TLS connection success rates
- Security policy compliance
- Service mesh mTLS status

##  🛠️ Troubleshooting

###  Common Issues

####  Certificate Validation Errors
```bash
#  Check certificate chain
openssl verify -CAfile secrets/tls/ca/ca.pem secrets/tls/api/cert.pem

#  Verify certificate dates
openssl x509 -in secrets/tls/api/cert.pem -noout -dates

#  Test TLS connection
openssl s_client -connect api:8443 -CAfile secrets/tls/ca/ca.pem
```

####  Service Connection Issues
```bash
#  Check service logs
docker-compose logs envoy-api

#  Test mTLS connection
curl --cacert secrets/tls/ca/ca.pem \
     --cert secrets/tls/api-client/cert.pem \
     --key secrets/tls/api-client/key.pem \
     https://envoy-api:8443/api/v1/health
```

####  Redis TLS Problems
```bash
#  Test Redis TLS connection
redis-cli --tls \
          --cert secrets/tls/redis-client/cert.pem \
          --key secrets/tls/redis-client/key.pem \
          --cacert secrets/tls/ca/ca.pem \
          -h redis -p 6379 ping
```

###  Debug Mode

Enable verbose logging:

```bash
export VERBOSE=true
./scripts/validate/test_tls.sh -v
./scripts/rotate-certs.sh -v
```

##  📋 Best Practices

###  Certificate Management
- Use short-lived certificates (≤30 days)
- Automate rotation processes
- Maintain secure backups
- Monitor expiry dates
- Use proper file permissions (400 for keys, 444 for certs)

###  Network Security
- Isolate TLS traffic on dedicated networks
- Use network policies for additional security
- Implement proper firewall rules
- Monitor for unauthorized connections

###  Operational Security
- Regular security audits
- Incident response procedures
- Certificate revocation processes
- Backup and recovery plans

##  🔗 References

- [TLS 1.3 RFC 8446](https://tools.ietf.org/html/rfc8446)
- [Envoy TLS Configuration](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/transport_sockets/tls/v3/tls.proto)
- [cert-manager Documentation](https://cert-manager.io/docs/)
- [Istio Security](https://istio.io/latest/docs/concepts/security/)
- [Redis TLS Configuration](https://redis.io/docs/manual/security/encryption/)

---

**Security Notice**: This implementation provides enterprise-grade TLS/mTLS security. Ensure all certificates and private keys are stored securely and access is restricted to authorized personnel only.