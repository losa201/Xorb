# XORB API Endpoints Matrix

**Audit Date**: 2025-08-15
**Total Endpoints**: 67 documented endpoints
**API Version**: v1
**Base URL**: `http://localhost:8000/api/v1`

## Core API Endpoints

### Authentication & Authorization
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| POST | `/auth/login` | User authentication | None | `LoginRequest` | 200, 401, 422 |
| POST | `/auth/refresh` | Token refresh | Bearer | `RefreshRequest` | 200, 401 |
| POST | `/auth/logout` | User logout | Bearer | None | 204, 401 |
| GET | `/auth/profile` | Get user profile | Bearer | None | 200, 401 |
| PUT | `/auth/profile` | Update profile | Bearer | `ProfileUpdate` | 200, 401, 422 |

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'
```

### PTaaS Core Operations
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| POST | `/ptaas/sessions` | Create scan session | Bearer | `ScanSessionRequest` | 201, 400, 401, 429 |
| GET | `/ptaas/sessions/{session_id}` | Get session status | Bearer | None | 200, 404, 401 |
| PUT | `/ptaas/sessions/{session_id}/pause` | Pause scan session | Bearer | None | 200, 404, 409 |
| PUT | `/ptaas/sessions/{session_id}/resume` | Resume scan session | Bearer | None | 200, 404, 409 |
| DELETE | `/ptaas/sessions/{session_id}/cancel` | Cancel scan session | Bearer | None | 204, 404, 409 |
| GET | `/ptaas/sessions` | List user sessions | Bearer | None | 200, 401 |

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/ptaas/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "targets": [{
      "host": "scanme.nmap.org",
      "ports": [22, 80, 443],
      "scan_profile": "comprehensive"
    }],
    "scan_type": "vulnerability_scan"
  }'
```

### PTaaS Results & Reports
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| GET | `/ptaas/findings` | Get scan findings | Bearer | None | 200, 401 |
| GET | `/ptaas/findings/{finding_id}` | Get specific finding | Bearer | None | 200, 404, 401 |
| GET | `/ptaas/reports/{session_id}` | Generate session report | Bearer | None | 200, 404, 401 |
| GET | `/ptaas/reports/{session_id}/pdf` | Download PDF report | Bearer | None | 200, 404, 401 |
| GET | `/ptaas/reports/{session_id}/json` | Get JSON report | Bearer | None | 200, 404, 401 |

**Rate Limiting**: `X-RateLimit-Limit: 100`, `X-RateLimit-Remaining: 99`

### PTaaS Configuration
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| GET | `/ptaas/profiles` | List scan profiles | Bearer | None | 200, 401 |
| GET | `/ptaas/profiles/{profile_id}` | Get scan profile | Bearer | None | 200, 404, 401 |
| POST | `/ptaas/profiles` | Create scan profile | Admin | `ScanProfileRequest` | 201, 400, 403 |
| PUT | `/ptaas/profiles/{profile_id}` | Update scan profile | Admin | `ScanProfileRequest` | 200, 404, 403 |

### PTaaS Orchestration
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| POST | `/ptaas/orchestration/workflows` | Create workflow | Bearer | `WorkflowRequest` | 201, 400, 401 |
| GET | `/ptaas/orchestration/workflows` | List workflows | Bearer | None | 200, 401 |
| POST | `/ptaas/orchestration/compliance-scan` | Compliance scan | Bearer | `ComplianceRequest` | 202, 400, 401 |
| POST | `/ptaas/orchestration/threat-simulation` | Threat simulation | Bearer | `ThreatSimRequest` | 202, 400, 401 |

## Health & Monitoring

### System Health
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| GET | `/health` | Application health | None | None | 200, 503 |
| GET | `/readiness` | Readiness probe | None | None | 200, 503 |
| GET | `/info` | Service information | None | None | 200 |
| GET | `/ptaas/health` | PTaaS subsystem health | None | None | 200, 503 |

**Example**:
```bash
curl "http://localhost:8000/api/v1/health"
# Response: {"status": "healthy", "timestamp": "2025-08-15T10:30:00Z", "checks": {...}}
```

### Metrics & Telemetry
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| GET | `/metrics` | Prometheus metrics | None | None | 200 |
| GET | `/telemetry/traces` | Get trace data | Admin | None | 200, 403 |
| GET | `/telemetry/performance` | Performance metrics | Admin | None | 200, 403 |

## Discovery & Intelligence

### Target Discovery
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| POST | `/discovery/scan` | Start discovery scan | Bearer | `DiscoveryRequest` | 202, 400, 401 |
| GET | `/discovery/results/{scan_id}` | Get discovery results | Bearer | None | 200, 404, 401 |
| GET | `/discovery/targets` | List discovered targets | Bearer | None | 200, 401 |

### Threat Intelligence
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| GET | `/intelligence/threats` | Get threat data | Bearer | None | 200, 401 |
| POST | `/intelligence/analyze` | Analyze indicators | Bearer | `AnalysisRequest` | 200, 400, 401 |
| GET | `/intelligence/feeds` | List threat feeds | Bearer | None | 200, 401 |

## Compliance & Evidence (G7)

### Evidence Management
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| GET | `/evidence/chains` | List evidence chains | Admin | None | 200, 403 |
| GET | `/evidence/chains/{chain_id}` | Get evidence chain | Admin | None | 200, 404, 403 |
| POST | `/evidence/verify` | Verify evidence | Admin | `VerificationRequest` | 200, 400, 403 |

### Compliance Reporting
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| GET | `/compliance/frameworks` | List frameworks | Bearer | None | 200, 401 |
| POST | `/compliance/assessments` | Start assessment | Bearer | `AssessmentRequest` | 202, 400, 401 |
| GET | `/compliance/reports/{assessment_id}` | Get compliance report | Bearer | None | 200, 404, 401 |

## Resource Management (G8)

### Quota & Fairness
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| GET | `/quotas/current` | Get current quotas | Bearer | None | 200, 401 |
| GET | `/quotas/usage` | Get usage statistics | Bearer | None | 200, 401 |
| GET | `/fairness/metrics` | Get fairness metrics | Admin | None | 200, 403 |

## Advanced Features

### Enterprise Security
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| POST | `/enterprise/campaigns` | Create security campaign | Enterprise | `CampaignRequest` | 201, 400, 403 |
| GET | `/enterprise/dashboards` | Get enterprise dashboard | Enterprise | None | 200, 403 |

### AI & ML Integration
| Method | Path | Summary | AuthZ | Request Schema | Response Codes |
|--------|------|---------|-------|----------------|----------------|
| POST | `/ai/analyze` | AI-powered analysis | Bearer | `AIAnalysisRequest` | 200, 400, 401 |
| GET | `/ai/models` | List available models | Bearer | None | 200, 401 |

## Schema Definitions

### Common Request Schemas

#### ScanSessionRequest
```json
{
  "targets": [
    {
      "host": "string",
      "ports": [80, 443],
      "scan_profile": "comprehensive",
      "stealth_mode": true,
      "authorized": true
    }
  ],
  "scan_type": "vulnerability_scan",
  "metadata": {
    "compliance_framework": "PCI-DSS",
    "priority": "high"
  }
}
```

#### WorkflowRequest
```json
{
  "name": "Weekly Security Scan",
  "targets": ["*.company.com"],
  "triggers": [
    {
      "trigger_type": "scheduled",
      "schedule": "0 2 * * 1"
    }
  ],
  "notifications": {
    "slack_channel": "#security",
    "email": ["team@company.com"]
  }
}
```

### Common Response Schemas

#### ScanSessionResponse
```json
{
  "session_id": "uuid",
  "status": "running",
  "scan_type": "vulnerability_scan",
  "created_at": "2025-08-15T10:30:00Z",
  "estimated_completion": "2025-08-15T11:30:00Z",
  "progress": {
    "targets_completed": 2,
    "targets_total": 5,
    "percentage": 40
  }
}
```

#### HealthResponse
```json
{
  "status": "healthy",
  "timestamp": "2025-08-15T10:30:00Z",
  "version": "2025.08-rc2",
  "checks": {
    "database": "healthy",
    "nats": "healthy",
    "redis": "healthy",
    "temporal": "healthy"
  },
  "metrics": {
    "active_sessions": 12,
    "queue_depth": 3,
    "response_time_ms": 150
  }
}
```

## Error Response Format

All API errors follow RFC 7807 Problem Details format:

```json
{
  "type": "https://docs.xorb.platform/errors/validation-error",
  "title": "Validation Error",
  "status": 422,
  "detail": "One or more validation errors occurred",
  "instance": "/api/v1/ptaas/sessions",
  "errors": [
    {
      "field": "targets[0].host",
      "message": "Invalid hostname format",
      "code": "INVALID_FORMAT"
    }
  ]
}
```

## Rate Limiting

All endpoints implement rate limiting with the following headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1692102600
X-RateLimit-Window: 60
```

### Rate Limits by Role
| Role | Requests/Minute | Burst Limit |
|------|-----------------|-------------|
| User | 100 | 150 |
| Premium | 500 | 750 |
| Enterprise | 2000 | 3000 |
| Admin | 5000 | 7500 |

## Idempotency

POST and PUT endpoints support idempotency via `Idempotency-Key` header:

```bash
curl -X POST "http://localhost:8000/api/v1/ptaas/sessions" \
  -H "Idempotency-Key: unique-key-123" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"targets": [...]}'
```

## Versioning & Compatibility

- **Current Version**: v1
- **Backward Compatibility**: Maintained for 2 major versions
- **Deprecation Notice**: 6 months advance notice
- **Content Negotiation**: Supports `application/json`, `application/xml`

## Security Headers

All responses include security headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

---

*This endpoints matrix provides comprehensive documentation for all XORB API endpoints, including authentication, request/response schemas, and operational characteristics.*
