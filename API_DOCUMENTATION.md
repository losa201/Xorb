# 📚 XORB API Documentation

**Comprehensive REST API Reference for the XORB Cybersecurity Platform**

---

## 🌟 **API Overview**

The XORB platform provides a comprehensive set of RESTful APIs for cybersecurity operations, threat intelligence, and security automation. All APIs use JSON for request/response payloads and follow OpenAPI 3.0 specifications.

### **Base URLs**
```
Production:   https://api.xorb-security.com/v1
Staging:      https://staging-api.xorb-security.com/v1
Local:        http://localhost:8000/v1
```

### **Authentication**
```http
Authorization: Bearer <jwt_token>
X-API-Key: <api_key>
```

---

## 🔐 **Authentication & Authorization**

### **Authentication Methods**

#### **JWT Bearer Token**
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password",
  "mfa_code": "123456"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "role": "security_analyst",
    "permissions": ["read_scans", "create_scans", "view_threats"]
  }
}
```

#### **API Key Authentication**
```http
GET /api/scans
X-API-Key: xorb_api_key_abc123def456
```

### **OAuth2 Integration**
```http
GET /auth/oauth/google
GET /auth/oauth/microsoft
GET /auth/oauth/okta
```

---

## 🛡️ **Core Security APIs**

### **Vulnerability Scanning**

#### **Start Security Scan**
```http
POST /api/v1/scans
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Production Web App Scan",
  "targets": [
    {
      "type": "web",
      "url": "https://app.example.com",
      "authentication": {
        "type": "basic",
        "username": "testuser",
        "password": "testpass"
      }
    },
    {
      "type": "network",
      "cidr": "10.0.0.0/24"
    }
  ],
  "scan_profile": "comprehensive",
  "compliance_frameworks": ["PCI-DSS", "OWASP-Top-10"],
  "stealth_mode": true,
  "schedule": {
    "type": "recurring",
    "interval": "weekly",
    "day_of_week": "sunday",
    "time": "02:00"
  },
  "notifications": {
    "email": ["security-team@example.com"],
    "slack": ["#security-alerts"],
    "webhook": "https://example.com/webhook"
  }
}
```

**Response:**
```json
{
  "scan_id": "scan_abc123def456",
  "status": "queued",
  "estimated_duration": "45 minutes",
  "created_at": "2025-08-08T21:30:00Z",
  "targets_count": 2,
  "scan_profile": "comprehensive",
  "progress_url": "/api/v1/scans/scan_abc123def456/progress"
}
```

#### **Get Scan Results**
```http
GET /api/v1/scans/{scan_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "scan_id": "scan_abc123def456",
  "status": "completed",
  "started_at": "2025-08-08T21:30:00Z",
  "completed_at": "2025-08-08T22:15:00Z",
  "duration": 2700,
  "summary": {
    "total_vulnerabilities": 15,
    "critical": 2,
    "high": 4,
    "medium": 6,
    "low": 3,
    "risk_score": 8.5
  },
  "vulnerabilities": [
    {
      "id": "vuln_001",
      "cve": "CVE-2024-1234",
      "title": "SQL Injection in Login Form",
      "severity": "critical",
      "cvss": 9.8,
      "description": "SQL injection vulnerability allows authentication bypass",
      "location": "https://app.example.com/login",
      "evidence": {
        "request": "POST /login HTTP/1.1...",
        "response": "HTTP/1.1 200 OK...",
        "payload": "admin' OR '1'='1"
      },
      "remediation": "Use parameterized queries and input validation",
      "references": [
        "https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-1234"
      ]
    }
  ],
  "compliance_status": {
    "PCI-DSS": {
      "score": 75,
      "failing_requirements": ["6.5.1", "11.2.1"],
      "recommendations": ["Fix SQL injection", "Implement regular scanning"]
    }
  }
}
```

### **Real-time Threat Intelligence**

#### **Analyze Threat Indicators**
```http
POST /api/v1/intelligence/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "indicators": [
    {
      "type": "ip",
      "value": "192.168.1.100"
    },
    {
      "type": "hash",
      "value": "d41d8cd98f00b204e9800998ecf8427e"
    },
    {
      "type": "domain",
      "value": "suspicious-domain.com"
    }
  ],
  "context": {
    "source": "network_traffic",
    "timestamp": "2025-08-08T21:30:00Z",
    "enterprise_context": {
      "sector": "financial_services",
      "critical_assets": ["payment_gateway", "customer_database"]
    }
  }
}
```

**Response:**
```json
{
  "analysis_id": "analysis_xyz789",
  "results": [
    {
      "indicator": "192.168.1.100",
      "type": "ip",
      "threat_level": "high",
      "confidence": 0.87,
      "reputation": {
        "malicious": true,
        "categories": ["botnet", "malware_c2"],
        "first_seen": "2025-07-15T10:30:00Z",
        "last_seen": "2025-08-08T20:45:00Z"
      },
      "threat_intelligence": {
        "threat_actor": "APT29",
        "campaign": "CozyBear_2025",
        "techniques": ["T1071.001", "T1090"],
        "related_indicators": ["suspicious-domain.com"]
      },
      "recommendations": [
        "Block IP at network perimeter",
        "Hunt for related indicators",
        "Check for compromise signs"
      ]
    }
  ],
  "correlation": {
    "related_incidents": 2,
    "attack_patterns": ["lateral_movement", "data_exfiltration"],
    "risk_assessment": "immediate_threat"
  }
}
```

### **SIEM & Behavioral Analytics**

#### **Get Security Events**
```http
GET /api/v1/siem/events?limit=100&severity=high&time_range=24h
Authorization: Bearer <token>
```

**Response:**
```json
{
  "events": [
    {
      "event_id": "event_123456",
      "timestamp": "2025-08-08T21:30:00Z",
      "severity": "high",
      "category": "authentication",
      "source": {
        "ip": "10.0.1.50",
        "hostname": "workstation-01",
        "user": "john.doe"
      },
      "description": "Multiple failed login attempts detected",
      "raw_log": "Aug 8 21:30:00 auth.log: Failed password for john.doe...",
      "enrichment": {
        "geolocation": {
          "country": "United States",
          "city": "New York",
          "coordinates": [40.7128, -74.0060]
        },
        "user_profile": {
          "normal_login_hours": "09:00-17:00",
          "typical_locations": ["10.0.1.0/24"],
          "risk_score": 0.3
        }
      }
    }
  ],
  "metadata": {
    "total_events": 1543,
    "filtered_events": 87,
    "time_range": "2025-08-07T21:30:00Z to 2025-08-08T21:30:00Z"
  }
}
```

#### **Behavioral Analysis**
```http
GET /api/v1/siem/behavioral/{user_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "user_id": "john.doe",
  "analysis_period": "30_days",
  "baseline_confidence": 0.89,
  "current_risk_score": 0.85,
  "anomalies": [
    {
      "type": "unusual_login_time",
      "severity": "medium",
      "score": 6.2,
      "description": "Login detected at 3:47 AM, outside normal hours",
      "baseline": "Normal hours: 09:00-17:00 EST",
      "observed": "Login at 03:47 EST"
    }
  ],
  "recommendations": [
    "Require additional authentication for after-hours access",
    "Monitor user activity closely for next 24 hours"
  ]
}
```

---

## 🤖 **AI & Automation APIs**

### **AI-Powered Threat Detection**

#### **Advanced Threat Analysis**
```http
POST /api/v1/ai/threat/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "data": {
    "network_traffic": {
      "source_ip": "192.168.1.100",
      "dest_ip": "10.0.0.5",
      "port": 443,
      "protocol": "tcp",
      "bytes": 1048576,
      "duration": 300
    },
    "endpoint_data": {
      "process_name": "powershell.exe",
      "command_line": "powershell.exe -EncodedCommand SGVsbG8gV29ybGQ=",
      "user": "SYSTEM",
      "parent_process": "cmd.exe"
    }
  },
  "context": {
    "environment": "production",
    "asset_criticality": "high",
    "compliance_requirements": ["PCI-DSS", "SOX"]
  }
}
```

**Response:**
```json
{
  "analysis_id": "ai_analysis_def456",
  "threat_assessment": {
    "overall_risk": "critical",
    "confidence": 0.92,
    "threat_types": ["command_injection", "privilege_escalation"],
    "attack_stage": "execution",
    "mitre_techniques": ["T1059.001", "T1055"]
  },
  "ai_insights": {
    "behavioral_deviation": 8.7,
    "pattern_recognition": {
      "known_attack_pattern": "APT_PowerShell_Execution",
      "similarity_score": 0.94,
      "historical_matches": 23
    },
    "prediction": {
      "next_likely_actions": [
        "lateral_movement",
        "credential_dumping",
        "persistence_establishment"
      ],
      "time_to_impact": "15_minutes"
    }
  },
  "automated_response": {
    "immediate_actions": [
      "Isolate affected endpoint",
      "Kill suspicious process",
      "Collect forensic artifacts"
    ],
    "recommended_actions": [
      "Deploy additional monitoring",
      "Notify security team",
      "Initiate incident response"
    ]
  }
}
```

### **Automated Security Orchestration**

#### **Create Security Workflow**
```http
POST /api/v1/orchestration/workflows
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Incident Response - Malware Detection",
  "description": "Automated workflow for malware incident response",
  "trigger": {
    "type": "event",
    "conditions": {
      "event_type": "malware_detected",
      "severity": ["high", "critical"]
    }
  },
  "steps": [
    {
      "name": "isolate_endpoint",
      "type": "action",
      "action": "endpoint_isolation",
      "parameters": {
        "endpoint_id": "{{trigger.endpoint_id}}",
        "isolation_type": "network"
      }
    },
    {
      "name": "collect_artifacts",
      "type": "action",
      "action": "forensic_collection",
      "depends_on": ["isolate_endpoint"],
      "parameters": {
        "endpoint_id": "{{trigger.endpoint_id}}",
        "artifacts": ["memory_dump", "disk_image", "network_logs"]
      }
    },
    {
      "name": "notify_team",
      "type": "notification",
      "depends_on": ["collect_artifacts"],
      "channels": [
        {
          "type": "slack",
          "channel": "#security-incidents",
          "message": "Malware detected on {{trigger.endpoint_id}}. Isolation and artifact collection completed."
        },
        {
          "type": "email",
          "recipients": ["security-team@example.com"],
          "subject": "Critical Security Incident",
          "template": "incident_notification"
        }
      ]
    }
  ]
}
```

**Response:**
```json
{
  "workflow_id": "workflow_ghi789",
  "status": "active",
  "created_at": "2025-08-08T21:30:00Z",
  "execution_url": "/api/v1/orchestration/workflows/workflow_ghi789/executions"
}
```

---

## 🔐 **Quantum Security APIs**

### **Post-Quantum Cryptography**

#### **Encrypt Data**
```http
POST /api/v1/crypto/encrypt
Authorization: Bearer <token>
Content-Type: application/json

{
  "data": "Sensitive financial data requiring quantum-safe protection",
  "algorithm": "CRYSTALS-Kyber",
  "key_id": "master_hybrid_key",
  "additional_data": {
    "classification": "confidential",
    "retention_period": "7_years"
  }
}
```

**Response:**
```json
{
  "operation_id": "crypto_op_jkl012",
  "encrypted_data": "base64_encoded_encrypted_data_here",
  "algorithm": "CRYSTALS-Kyber",
  "key_id": "master_hybrid_key",
  "execution_time_ms": 0.8,
  "metadata": {
    "encryption_timestamp": "2025-08-08T21:30:00Z",
    "data_size": 1024,
    "security_level": 256
  }
}
```

#### **Digital Signature**
```http
POST /api/v1/crypto/sign
Authorization: Bearer <token>
Content-Type: application/json

{
  "data": "Critical security policy document",
  "algorithm": "CRYSTALS-Dilithium",
  "key_id": "signing_key_001"
}
```

**Response:**
```json
{
  "signature": "base64_encoded_signature_here",
  "algorithm": "CRYSTALS-Dilithium",
  "key_id": "signing_key_001",
  "data_hash": "sha256_hash_of_original_data",
  "signature_timestamp": "2025-08-08T21:30:00Z",
  "verification_url": "/api/v1/crypto/verify"
}
```

---

## 📊 **Compliance & Reporting APIs**

### **Compliance Management**

#### **Generate Compliance Report**
```http
POST /api/v1/compliance/reports
Authorization: Bearer <token>
Content-Type: application/json

{
  "framework": "PCI-DSS",
  "version": "4.0",
  "scope": {
    "assets": ["payment_gateway", "customer_database"],
    "time_period": {
      "start": "2025-07-01T00:00:00Z",
      "end": "2025-08-01T00:00:00Z"
    }
  },
  "format": "pdf",
  "include_evidence": true
}
```

**Response:**
```json
{
  "report_id": "report_mno345",
  "status": "generating",
  "estimated_completion": "2025-08-08T21:45:00Z",
  "download_url": "/api/v1/compliance/reports/report_mno345/download",
  "preview_url": "/api/v1/compliance/reports/report_mno345/preview"
}
```

#### **Compliance Status**
```http
GET /api/v1/compliance/status?framework=PCI-DSS
Authorization: Bearer <token>
```

**Response:**
```json
{
  "framework": "PCI-DSS",
  "version": "4.0",
  "overall_compliance": 87.5,
  "last_assessment": "2025-08-01T00:00:00Z",
  "requirements": [
    {
      "requirement": "1.1.1",
      "title": "Network security controls",
      "status": "compliant",
      "score": 95,
      "evidence_count": 12,
      "last_tested": "2025-08-01T10:30:00Z"
    },
    {
      "requirement": "6.5.1",
      "title": "Injection flaws",
      "status": "non_compliant",
      "score": 45,
      "findings": ["SQL injection in payment form"],
      "remediation_deadline": "2025-08-15T00:00:00Z"
    }
  ],
  "upcoming_assessments": [
    {
      "requirement": "11.2.1",
      "scheduled_date": "2025-08-15T00:00:00Z",
      "type": "quarterly_scan"
    }
  ]
}
```

---

## 🔍 **Search & Analytics APIs**

### **Security Data Search**

#### **Advanced Security Search**
```http
POST /api/v1/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": {
    "bool": {
      "must": [
        {
          "term": {
            "event_type": "authentication_failure"
          }
        },
        {
          "range": {
            "timestamp": {
              "gte": "2025-08-07T21:30:00Z",
              "lte": "2025-08-08T21:30:00Z"
            }
          }
        }
      ]
    }
  },
  "aggregations": {
    "by_user": {
      "terms": {
        "field": "user_id",
        "size": 10
      }
    },
    "by_source_ip": {
      "terms": {
        "field": "source_ip",
        "size": 10
      }
    }
  },
  "size": 100,
  "sort": [
    {
      "timestamp": "desc"
    }
  ]
}
```

**Response:**
```json
{
  "total_hits": 1247,
  "max_score": 1.0,
  "hits": [
    {
      "id": "event_pqr678",
      "score": 1.0,
      "source": {
        "timestamp": "2025-08-08T21:29:45Z",
        "event_type": "authentication_failure",
        "user_id": "john.doe",
        "source_ip": "192.168.1.100",
        "message": "Invalid password attempt"
      }
    }
  ],
  "aggregations": {
    "by_user": {
      "buckets": [
        {
          "key": "john.doe",
          "doc_count": 45
        },
        {
          "key": "jane.smith",
          "doc_count": 23
        }
      ]
    }
  }
}
```

---

## 🌐 **Webhook & Integration APIs**

### **Webhook Management**

#### **Register Webhook**
```http
POST /api/v1/webhooks
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://example.com/security-webhook",
  "events": [
    "vulnerability_detected",
    "scan_completed",
    "incident_created",
    "compliance_violation"
  ],
  "filters": {
    "severity": ["high", "critical"],
    "asset_tags": ["production", "critical"]
  },
  "headers": {
    "X-Webhook-Source": "XORB-Security",
    "Authorization": "Bearer webhook_token_123"
  },
  "retry_policy": {
    "max_attempts": 3,
    "backoff_seconds": [1, 5, 15]
  }
}
```

**Response:**
```json
{
  "webhook_id": "webhook_stu901",
  "status": "active",
  "created_at": "2025-08-08T21:30:00Z",
  "secret": "webhook_secret_xyz789",
  "test_url": "/api/v1/webhooks/webhook_stu901/test"
}
```

### **Third-party Integrations**

#### **JIRA Integration**
```http
POST /api/v1/integrations/jira/tickets
Authorization: Bearer <token>
Content-Type: application/json

{
  "vulnerability_id": "vuln_001",
  "project_key": "SEC",
  "issue_type": "Security Vulnerability",
  "priority": "High",
  "assignee": "security-team",
  "labels": ["security", "vulnerability", "sql-injection"],
  "custom_fields": {
    "cvss_score": 9.8,
    "affected_systems": ["payment_gateway"]
  }
}
```

**Response:**
```json
{
  "ticket_id": "SEC-1234",
  "ticket_url": "https://company.atlassian.net/browse/SEC-1234",
  "status": "created",
  "assignee": "security-team",
  "created_at": "2025-08-08T21:30:00Z"
}
```

---

## 📈 **Analytics & Metrics APIs**

### **Security Metrics**

#### **Security Dashboard Metrics**
```http
GET /api/v1/metrics/dashboard?time_range=7d
Authorization: Bearer <token>
```

**Response:**
```json
{
  "time_range": "7d",
  "generated_at": "2025-08-08T21:30:00Z",
  "metrics": {
    "vulnerability_trends": {
      "total_vulnerabilities": 1247,
      "new_this_period": 87,
      "resolved_this_period": 145,
      "critical_open": 12,
      "high_open": 45,
      "medium_open": 234,
      "low_open": 543
    },
    "threat_intelligence": {
      "indicators_processed": 12547,
      "threats_detected": 234,
      "false_positives": 23,
      "accuracy_rate": 0.91
    },
    "compliance_status": {
      "frameworks_monitored": 5,
      "overall_score": 87.5,
      "requirements_failing": 12,
      "requirements_passing": 156
    },
    "system_performance": {
      "avg_response_time_ms": 45,
      "uptime_percentage": 99.97,
      "total_api_requests": 45678,
      "error_rate": 0.02
    }
  },
  "trends": {
    "vulnerability_trend": [
      {"date": "2025-08-01", "value": 1200},
      {"date": "2025-08-02", "value": 1189},
      {"date": "2025-08-08", "value": 1247}
    ]
  }
}
```

---

## ⚡ **Rate Limiting & Performance**

### **Rate Limits**
```yaml
Tier Limits:
  Free:        100 requests/hour
  Professional: 1000 requests/hour
  Enterprise:   10000 requests/hour
  Unlimited:    No limits (custom enterprise)

Headers:
  X-RateLimit-Limit: 1000
  X-RateLimit-Remaining: 999
  X-RateLimit-Reset: 1628097600
```

### **Performance Expectations**
```yaml
Response Times:
  Authentication:    < 100ms
  Simple Queries:    < 200ms
  Complex Analysis:  < 2s
  Report Generation: < 30s
  Bulk Operations:   Async with status endpoints

Pagination:
  Default Page Size: 20
  Maximum Page Size: 100
  Use cursor-based pagination for large datasets
```

---

## 🔧 **Error Handling**

### **Standard Error Response**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    },
    "request_id": "req_abc123",
    "timestamp": "2025-08-08T21:30:00Z"
  }
}
```

### **Common Error Codes**
```yaml
HTTP 400: Bad Request
  - VALIDATION_ERROR
  - INVALID_FORMAT
  - MISSING_PARAMETER

HTTP 401: Unauthorized
  - INVALID_TOKEN
  - TOKEN_EXPIRED
  - INVALID_API_KEY

HTTP 403: Forbidden
  - INSUFFICIENT_PERMISSIONS
  - RATE_LIMIT_EXCEEDED
  - RESOURCE_ACCESS_DENIED

HTTP 404: Not Found
  - RESOURCE_NOT_FOUND
  - ENDPOINT_NOT_FOUND

HTTP 429: Too Many Requests
  - RATE_LIMIT_EXCEEDED

HTTP 500: Internal Server Error
  - INTERNAL_ERROR
  - SERVICE_UNAVAILABLE
```

---

## 📚 **SDK & Code Examples**

### **Python SDK**
```python
from xorb_security import XORBClient

# Initialize client
client = XORBClient(
    api_key="your_api_key",
    base_url="https://api.xorb-security.com/v1"
)

# Start vulnerability scan
scan = client.scans.create(
    name="Production Scan",
    targets=["https://example.com"],
    scan_profile="comprehensive"
)

# Monitor scan progress
while scan.status != "completed":
    scan = client.scans.get(scan.id)
    print(f"Scan progress: {scan.progress}%")
    time.sleep(30)

# Get results
vulnerabilities = scan.get_vulnerabilities()
for vuln in vulnerabilities:
    print(f"{vuln.severity}: {vuln.title}")
```

### **JavaScript/Node.js SDK**
```javascript
import { XORBClient } from '@xorb-security/sdk';

const client = new XORBClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.xorb-security.com/v1'
});

// Analyze threat indicators
const analysis = await client.intelligence.analyze({
  indicators: [
    { type: 'ip', value: '192.168.1.100' },
    { type: 'hash', value: 'd41d8cd98f00b204e9800998ecf8427e' }
  ],
  context: {
    source: 'network_traffic',
    enterprise_context: {
      sector: 'financial_services'
    }
  }
});

console.log('Threat Level:', analysis.results[0].threat_level);
console.log('Confidence:', analysis.results[0].confidence);
```

---

## 🚀 **Getting Started**

1. **Sign up** for a XORB account at https://xorb-security.com
2. **Generate API credentials** in your dashboard
3. **Download SDK** for your preferred language
4. **Start with basic scan** using the examples above
5. **Explore advanced features** like AI analysis and automation

### **Interactive API Explorer**
Visit https://api.xorb-security.com/docs for an interactive API explorer with:
- Live API testing
- Authentication examples
- Response schema validation
- Code generation for multiple languages

---

**📞 Support**: api-support@xorb-security.com  
**📖 Documentation**: https://docs.xorb-security.com  
**🐛 Issues**: https://github.com/xorb-security/api-issues