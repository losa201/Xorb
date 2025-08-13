# ADR-003: Authentication Artifact Architecture for XORB Platform

**Status:** Accepted  
**Date:** 2025-08-13  
**Deciders:** Chief Architect  

## Context

XORB Discovery-First, Two-Tier Bus, SEaaS architecture requires enterprise-grade authentication supporting multi-tenant isolation, service-to-service authentication, and integration with existing Vault-based secret management. Current JWT/OIDC implementation needs enhancement for gRPC services and cross-tier authentication.

## Decision

### Authentication Artifact Types (locked)

#### Primary Artifacts
1. **Service Identity Certificates** (mTLS for Tier-2 bus) (locked)
2. **JWT Access Tokens** (API authentication, tenant context) (locked)
3. **API Keys** (External integrations, webhook authentication)
4. **Temporal Worker Tokens** (Workflow authentication)
5. **Discovery Job Tickets** (Short-lived, scoped authorization)

#### Artifact Lifecycle Management
- **Issuance**: Vault PKI engine + JWT signing service
- **Distribution**: Secure channels (mTLS, encrypted storage)
- **Validation**: Per-tier validation with cached public keys
- **Revocation**: CRL + JWT blacklist with Redis backing
- **Rotation**: Automated rotation with overlap periods

### Service Identity Architecture

#### mTLS Certificate Hierarchy
```
Root CA (Vault PKI)
├── Intermediate CA (XORB Platform)
│   ├── Service Certificates (30-day TTL)
│   │   ├── discovery-service.xorb.internal
│   │   ├── ptaas-scanner.xorb.internal
│   │   └── intelligence-engine.xorb.internal
│   └── Client Certificates (7-day TTL)
│       ├── orchestrator-worker-{id}.xorb.internal
│       └── tier1-ring-client-{id}.xorb.internal
└── External Integration CA
    ├── webhook-receiver.xorb.external
    └── api-gateway.xorb.external
```

#### Certificate Subject Alternative Names (SAN)
```
Service Certificate SAN:
- DNS: discovery-service.xorb.internal
- DNS: discovery-service.xorb-ns.svc.cluster.local
- URI: spiffe://xorb.platform/service/discovery
- IP: 10.0.0.100 (static service IP)

Client Certificate SAN:
- DNS: worker-{uuid}.xorb.internal
- URI: spiffe://xorb.platform/worker/{uuid}
- Email: worker-{uuid}@xorb.internal (for audit)
```

### JWT Token Architecture

#### Access Token Structure (JWE nested in JWS)
```json
{
  "header": {
    "alg": "ES256",
    "typ": "JWT",
    "kid": "xorb-signing-2025-08-13"
  },
  "payload": {
    "iss": "https://auth.xorb.platform",
    "sub": "user:12345",
    "aud": ["xorb:api", "xorb:discovery", "xorb:ptaas"],
    "exp": 1723545600,
    "iat": 1723542000,
    "jti": "jwt-uuid-12345",
    "tenant_id": "tenant-abc123",
    "roles": ["security_analyst", "scan_operator"],
    "scopes": ["discovery:read", "discovery:write", "scan:execute"],
    "session_id": "session-xyz789",
    "mfa_verified": true,
    "risk_score": 0.2,
    "tier1_access": false,
    "tier2_topics": ["discovery.jobs.v1.tenant-abc123"]
  }
}
```

#### Discovery Job Ticket (Short-lived Authorization)
```json
{
  "header": {
    "alg": "ES256",
    "typ": "JWT",
    "kid": "xorb-discovery-2025-08-13"
  },
  "payload": {
    "iss": "https://discovery.xorb.platform",
    "sub": "job:discovery-job-456789",
    "aud": ["xorb:scanner", "xorb:intelligence"],
    "exp": 1723543800,  // 30 minutes
    "iat": 1723542000,
    "jti": "ticket-discovery-456789",
    "tenant_id": "tenant-abc123",
    "job_id": "discovery-job-456789",
    "target_scope": ["192.168.1.0/24", "scanme.example.com"],
    "scan_profile": "comprehensive",
    "max_concurrency": 10,
    "allowed_tools": ["nmap", "nuclei", "nikto"],
    "evidence_bucket": "s3://xorb-evidence/tenant-abc123/job-456789/"
  }
}
```

### Integration with Current XORB Stack

#### Vault Integration Enhancement
```python
# src/common/vault_client_enhanced.py
class EnhancedVaultClient:
    async def issue_service_certificate(self, service_name: str, ttl: str = "720h") -> CertBundle:
        """Issue mTLS certificate for service identity"""
        cert_data = await self.vault_client.secrets.pki.generate_certificate(
            name="xorb-service-role",
            common_name=f"{service_name}.xorb.internal",
            alt_names=[f"{service_name}.xorb-ns.svc.cluster.local"],
            uri_sans=[f"spiffe://xorb.platform/service/{service_name}"],
            ttl=ttl
        )
        return CertBundle(
            certificate=cert_data["data"]["certificate"],
            private_key=cert_data["data"]["private_key"],
            ca_chain=cert_data["data"]["ca_chain"]
        )
    
    async def issue_discovery_ticket(self, job: DiscoveryJob, user_context: UserContext) -> str:
        """Issue short-lived discovery job authorization ticket"""
        claims = {
            "sub": f"job:{job.id}",
            "tenant_id": job.tenant_id,
            "job_id": job.id,
            "target_scope": job.targets,
            "scan_profile": job.profile,
            "max_concurrency": job.max_concurrency,
            "allowed_tools": job.allowed_tools,
            "evidence_bucket": f"s3://xorb-evidence/{job.tenant_id}/{job.id}/"
        }
        return await self.sign_jwt(claims, ttl=1800)  # 30 minutes
```

#### FastAPI Middleware Enhancement
```python
# src/api/app/middleware/enhanced_auth.py
class EnhancedAuthMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app
        self.cert_validator = mTLSCertificateValidator()
        self.jwt_validator = JWTValidator()
        
    async def __call__(self, request: Request, call_next):
        # Service-to-service: mTLS certificate validation
        if request.url.path.startswith("/internal/"):
            client_cert = request.state.client_cert
            if not await self.cert_validator.validate_service_cert(client_cert):
                raise HTTPException(401, "Invalid service certificate")
            request.state.service_identity = client_cert.subject
            
        # User API: JWT validation with tenant context
        elif request.url.path.startswith("/api/"):
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(401, "Missing or invalid authorization header")
                
            token = auth_header[7:]  # Remove "Bearer "
            claims = await self.jwt_validator.validate_access_token(token)
            request.state.user_context = UserContext.from_claims(claims)
            request.state.tenant_id = claims["tenant_id"]
            
        # Discovery workflows: Job ticket validation
        elif request.url.path.startswith("/discovery/"):
            ticket_header = request.headers.get("X-Discovery-Ticket")
            if ticket_header:
                claims = await self.jwt_validator.validate_discovery_ticket(ticket_header)
                request.state.discovery_context = DiscoveryContext.from_claims(claims)
                
        return await call_next(request)
```

### Cross-Tier Authentication Flow

#### Tier-1 (Local Ring) Authentication
```
1. Service startup: Load mTLS certificate from Vault
2. Ring connection: Present certificate to ring broker
3. Peer verification: Validate certificate chain and SPIFFE ID
4. Message signing: Sign ring messages with service private key
5. Message verification: Validate signatures on received messages
```

#### Tier-2 (Pub/Sub) Authentication
```
1. NATS connection: mTLS handshake with service certificate
2. Topic authorization: OPA policy check with certificate subject
3. Message publish: Include JWT in message headers for consumer validation
4. Message consume: Validate JWT and check tenant isolation
5. Idempotency: Use JWT jti claim for exactly-once semantics
```

### Security Controls

#### Certificate Validation
```go
// platform/bus/pubsub/auth.go
func ValidateServiceCertificate(cert *x509.Certificate) (*ServiceIdentity, error) {
    // Verify certificate chain against root CA
    if err := verifyCertChain(cert, rootCACert); err != nil {
        return nil, fmt.Errorf("certificate chain validation failed: %w", err)
    }
    
    // Extract SPIFFE ID from URI SAN
    spiffeID, err := extractSPIFFEID(cert)
    if err != nil {
        return nil, fmt.Errorf("invalid SPIFFE ID: %w", err)
    }
    
    // Validate certificate is not revoked
    if err := checkCRL(cert); err != nil {
        return nil, fmt.Errorf("certificate revoked: %w", err)
    }
    
    return &ServiceIdentity{
        ServiceName: spiffeID.Service(),
        Namespace:   spiffeID.Namespace(),
        Certificate: cert,
    }, nil
}
```

#### JWT Validation with Redis Cache
```python
# src/api/app/services/jwt_validator.py
class JWTValidator:
    def __init__(self, redis_client: Redis, vault_client: VaultClient):
        self.redis = redis_client
        self.vault = vault_client
        self.public_keys_cache = {}
        
    async def validate_access_token(self, token: str) -> Dict[str, Any]:
        # Check JWT blacklist (revoked tokens)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if await self.redis.exists(f"jwt:blacklist:{token_hash}"):
            raise JWTValidationError("Token has been revoked")
            
        # Decode and validate JWT
        header = jwt.get_unverified_header(token)
        kid = header["kid"]
        
        # Get public key (cached for 1 hour)
        public_key = await self.get_public_key(kid)
        claims = jwt.decode(token, public_key, algorithms=["ES256"])
        
        # Validate claims
        if claims["exp"] < time.time():
            raise JWTValidationError("Token has expired")
            
        if "xorb:api" not in claims["aud"]:
            raise JWTValidationError("Invalid audience")
            
        # Check session validity
        session_id = claims.get("session_id")
        if session_id and not await self.redis.exists(f"session:{session_id}"):
            raise JWTValidationError("Session has been terminated")
            
        return claims
        
    async def get_public_key(self, kid: str) -> str:
        # Try cache first
        if kid in self.public_keys_cache:
            return self.public_keys_cache[kid]
            
        # Fetch from Vault
        key_data = await self.vault.secrets.transit.read_key(name=kid)
        public_key = key_data["data"]["keys"]["1"]["public_key"]
        
        # Cache for 1 hour
        self.public_keys_cache[kid] = public_key
        return public_key
```

### Tenant Isolation

#### Multi-Tenant JWT Claims
```python
# Tenant context extracted from JWT
@dataclass
class TenantContext:
    tenant_id: str
    organization_id: str
    subscription_tier: str  # "free", "pro", "enterprise"
    rate_limits: Dict[str, int]
    allowed_features: List[str]
    data_region: str  # "us-east-1", "eu-west-1", etc.
    
    @property
    def tier2_topic_prefix(self) -> str:
        return f"{self.tenant_id}"
        
    @property
    def evidence_bucket(self) -> str:
        return f"xorb-evidence-{self.data_region}/{self.tenant_id}"
```

#### Topic-Level Authorization
```rego
# policy/tenant_isolation.rego
package xorb.tenant_isolation

# Allow access to tenant-specific topics only
allow {
    input.action == "publish"
    topic_parts := split(input.topic, ".")
    topic_tenant := topic_parts[3]  # discovery.jobs.v1.{tenant_id}
    topic_tenant == input.user.tenant_id
}

# Prevent cross-tenant data access
deny {
    input.action == "consume"
    topic_parts := split(input.topic, ".")
    topic_tenant := topic_parts[3]
    topic_tenant != input.user.tenant_id
    reason := "Cross-tenant access denied"
}
```

### Monitoring and Audit

#### Authentication Metrics
```
# Certificate validation
xorb_auth_cert_validations_total{service, result="success|failure"}
xorb_auth_cert_rotation_total{service}
xorb_auth_cert_expiry_days{service}

# JWT validation  
xorb_auth_jwt_validations_total{tenant_id, token_type, result="success|failure"}
xorb_auth_jwt_blacklist_checks_total{result="hit|miss"}
xorb_auth_session_validations_total{tenant_id, result="valid|invalid|expired"}

# Cross-tier authentication
xorb_auth_tier1_connections_total{service, peer_service}
xorb_auth_tier2_connections_total{service, topic}
```

#### Audit Events
```json
{
  "timestamp": "2025-08-13T12:34:56Z",
  "event_type": "authentication_success",
  "tenant_id": "tenant-abc123",
  "user_id": "user-12345",
  "service": "discovery-service",
  "auth_method": "jwt_bearer",
  "session_id": "session-xyz789",
  "client_ip": "192.168.1.100",
  "user_agent": "XORB-Dashboard/1.0",
  "mfa_verified": true,
  "risk_score": 0.2,
  "trace_id": "abc123def456",
  "immutable_signature": "sha256:..."
}
```

## Integration Points

### Current Vault Configuration Enhancement
```hcl
# Additional Vault configuration for enhanced auth
path "pki/issue/xorb-service-role" {
  capabilities = ["create", "update"]
}

path "transit/sign/xorb-discovery-tickets" {
  capabilities = ["create", "update"]
}

# Service identity policies
path "secret/xorb/services/+/identity" {
  capabilities = ["read"]
}
```

### Temporal Worker Authentication
```python
# src/orchestrator/worker_auth.py
class AuthenticatedTemporalWorker:
    def __init__(self, vault_client: VaultClient):
        self.vault = vault_client
        self.service_cert = None
        
    async def start_worker(self):
        # Get service certificate from Vault
        self.service_cert = await self.vault.issue_service_certificate("temporal-worker")
        
        # Configure Temporal client with mTLS
        client = await Client.connect(
            "temporal.xorb.internal:7233",
            tls=TLSConfig(
                client_cert=self.service_cert.certificate,
                client_private_key=self.service_cert.private_key,
                server_root_ca_cert=self.service_cert.ca_chain
            )
        )
        
        # Start worker with authenticated client
        worker = Worker(
            client,
            task_queue="discovery-workflows",
            workflows=[DiscoveryWorkflow],
            activities=[ScanActivity]
        )
        await worker.run()
```

## Consequences

### Positive
- Strong multi-tenant isolation with cryptographic guarantees
- Integration with existing Vault infrastructure
- Support for both user and service authentication
- Automatic certificate rotation and revocation
- Comprehensive audit trail for compliance

### Negative
- Increased complexity in certificate management
- Additional latency for certificate/JWT validation
- More complex deployment and configuration
- Higher operational overhead for key rotation

### Risk Mitigation
- Cached validation to reduce latency impact
- Automated certificate rotation to prevent outages
- Fallback authentication mechanisms for degraded scenarios
- Comprehensive monitoring and alerting for auth failures

## Implementation Plan

### Phase 1 (Week 1): Certificate Infrastructure
- Vault PKI configuration for service certificates
- mTLS certificate validation in Tier-2 services
- Basic service identity framework

### Phase 2 (Week 2): JWT Enhancement
- Discovery job ticket implementation
- Enhanced JWT validation with caching
- Cross-tier authentication flow

### Phase 3 (Week 3): Tenant Isolation
- Multi-tenant JWT claims and validation
- Topic-level authorization with OPA
- Audit logging for all authentication events

### Phase 4 (Week 4): Integration Testing
- End-to-end authentication testing
- Performance testing with auth overhead
- Security testing and penetration testing