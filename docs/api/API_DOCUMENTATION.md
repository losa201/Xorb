# XORB Enterprise API Documentation

- *Production-Ready Cybersecurity Platform API**

[![API Version](https://img.shields.io/badge/API-v3.0.0-blue.svg)](https://api.xorb-security.com)
[![Status](https://img.shields.io/badge/Status-Production--Ready-green.svg)](#status)
[![Security](https://img.shields.io/badge/Security-Enterprise--Grade-red.svg)](#security)

- --

##  📋 **Table of Contents**

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [PTaaS API](#ptaas-api)
4. [PTaaS Orchestration](#ptaas-orchestration)
5. [Intelligence API](#intelligence-api)
6. [Platform Management](#platform-management)
7. [Security & Rate Limiting](#security--rate-limiting)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

- --

##  🚀 **Getting Started**

###  **Base URLs**
- **Development**: `http://localhost:8000/api/v1`
- **Production**: `https://api.xorb-security.com/api/v1`

###  **API Documentation**
- **Interactive Docs**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

###  **Quick Test**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# API information
curl http://localhost:8000/api/v1/info
```

- --

##  🔐 **Authentication**

###  **JWT Bearer Token**
All API endpoints require JWT authentication:

```bash
# Login to get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user@company.com", "password": "password"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Use token in requests
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  http://localhost:8000/api/v1/ptaas/sessions
```

###  **API Key Authentication**
For service-to-service communication:

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v1/platform/services
```

- --

##  🎯 **PTaaS API**

###  **Core Endpoints**

####  **Create Scan Session**
```http
POST /api/v1/ptaas/sessions
Content-Type: application/json
Authorization: Bearer {token}

{
  "targets": [
    {
      "host": "target.example.com",
      "ports": [22, 80, 443, 8080],
      "scan_profile": "comprehensive",
      "stealth_mode": true,
      "authorized": true
    }
  ],
  "scan_type": "comprehensive",
  "metadata": {
    "project": "Q1_Security_Assessment",
    "environment": "production"
  }
}
```

- *Response:**
```json
{
  "session_id": "session_1234567890",
  "status": "created",
  "scan_type": "comprehensive",
  "targets_count": 1,
  "created_at": "2025-01-15T10:30:00Z",
  "started_at": null,
  "completed_at": null,
  "results": null
}
```

####  **Get Scan Status**
```http
GET /api/v1/ptaas/sessions/{session_id}
Authorization: Bearer {token}
```

- *Response:**
```json
{
  "session_id": "session_1234567890",
  "status": "completed",
  "scan_type": "comprehensive",
  "targets_count": 1,
  "created_at": "2025-01-15T10:30:00Z",
  "started_at": "2025-01-15T10:30:05Z",
  "completed_at": "2025-01-15T10:45:00Z",
  "results": {
    "scan_results": [...],
    "summary": {
      "total_vulnerabilities": 15,
      "critical_vulnerabilities": 2,
      "high_vulnerabilities": 5,
      "medium_vulnerabilities": 6,
      "low_vulnerabilities": 2,
      "risk_level": "critical"
    }
  }
}
```

####  **Available Scan Profiles**
```http
GET /api/v1/ptaas/profiles
Authorization: Bearer {token}
```

- *Response:**
```json
{
  "profiles": {
    "quick": {
      "description": "Fast network scan with basic service detection",
      "timeout": 300,
      "tools": ["nmap_basic"],
      "max_ports": 100
    },
    "comprehensive": {
      "description": "Full security assessment with vulnerability scanning",
      "timeout": 1800,
      "tools": ["nmap_full", "nuclei", "custom_checks"],
      "max_ports": 1000
    },
    "stealth": {
      "description": "Low-profile scanning to avoid detection",
      "timeout": 3600,
      "tools": ["nmap_stealth", "custom_passive"],
      "max_ports": 500
    },
    "web_focused": {
      "description": "Specialized web application security testing",
      "timeout": 1200,
      "tools": ["nmap_web", "nikto", "dirb", "nuclei_web"],
      "max_ports": 50
    }
  },
  "available_scanners": ["nmap", "nuclei", "nikto", "sslscan", "dirb"]
}
```

####  **Validate Target**
```http
POST /api/v1/ptaas/validate-target
Content-Type: application/json
Authorization: Bearer {token}

{
  "host": "scanme.nmap.org",
  "ports": [22, 80, 443],
  "scan_profile": "comprehensive",
  "authorized": true
}
```

- *Response:**
```json
{
  "valid": true,
  "host": "scanme.nmap.org",
  "reachable": true,
  "authorized": true,
  "port_count": 3,
  "warnings": ["Port 22 requires special authorization"],
  "errors": []
}
```

####  **Get Scan Results**
```http
GET /api/v1/ptaas/scan-results/{session_id}?format=json
Authorization: Bearer {token}
```

- **Supported formats**: `json`, `pdf`, `csv`

- --

##  🔧 **PTaaS Orchestration**

###  **Advanced Workflow Management**

####  **Create Automated Workflow**
```http
POST /api/v1/ptaas/orchestration/workflows
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Weekly Security Assessment",
  "description": "Automated weekly security scanning for all production systems",
  "targets": ["*.production.company.com", "api.company.com"],
  "scan_profiles": ["comprehensive", "web_focused"],
  "triggers": [
    {
      "trigger_type": "scheduled",
      "schedule": "0 2 * * 1",
      "conditions": null
    }
  ],
  "notifications": {
    "email": ["security-team@company.com"],
    "slack": ["#security-alerts"],
    "on_critical": true,
    "on_completion": true
  },
  "retention_days": 90
}
```

####  **Execute Workflow**
```http
POST /api/v1/ptaas/orchestration/workflows/{workflow_id}/execute
Content-Type: application/json
Authorization: Bearer {token}

{
  "trigger_data": {
    "initiated_by": "security_analyst",
    "priority": "high",
    "custom_targets": ["urgent.company.com"]
  }
}
```

####  **Compliance Scanning**
```http
POST /api/v1/ptaas/orchestration/compliance-scan
Content-Type: application/json
Authorization: Bearer {token}

{
  "compliance_framework": "PCI-DSS",
  "scope": {
    "card_data_environment": true,
    "network_segments": ["dmz", "internal"],
    "systems": ["web-servers", "databases", "payment-gateway"]
  },
  "targets": ["web1.company.com", "db1.company.com", "gateway.company.com"],
  "assessment_type": "full"
}
```

- **Supported frameworks**: `PCI-DSS`, `HIPAA`, `SOX`, `ISO-27001`, `GDPR`, `NIST`, `CIS`

####  **Threat Simulation**
```http
POST /api/v1/ptaas/orchestration/threat-simulation
Content-Type: application/json
Authorization: Bearer {token}

{
  "simulation_type": "apt_simulation",
  "target_environment": {
    "network": "10.0.0.0/24",
    "systems": ["workstations", "servers"],
    "critical_assets": ["domain_controller", "file_server"]
  },
  "attack_vectors": ["spear_phishing", "lateral_movement", "privilege_escalation"],
  "duration_hours": 24,
  "stealth_level": "medium"
}
```

- **Supported simulations**:
- `apt_simulation` - Advanced Persistent Threat
- `ransomware_simulation` - Ransomware attack patterns
- `insider_threat` - Malicious insider scenarios
- `phishing_campaign` - Social engineering attacks
- `lateral_movement` - Network traversal testing
- `data_exfiltration` - Data theft simulation

####  **Advanced Scan Workflow**
```http
POST /api/v1/ptaas/orchestration/advanced-scan
Content-Type: application/json
Authorization: Bearer {token}

{
  "targets": ["api.company.com", "web.company.com"],
  "scan_types": [
    "network_discovery",
    "service_enumeration",
    "vulnerability_scan",
    "web_application_scan",
    "ssl_analysis",
    "compliance_check"
  ],
  "priority": "high",
  "constraints": {
    "max_duration_hours": 4,
    "rate_limit": 100,
    "business_hours_only": true
  }
}
```

- --

##  🤖 **Intelligence API**

###  **AI-Powered Threat Analysis**

####  **Analyze Threat Indicators**
```http
POST /api/v1/intelligence/analyze
Content-Type: application/json
Authorization: Bearer {token}

{
  "indicators": [
    "suspicious_network_activity",
    "privilege_escalation",
    "unusual_data_access"
  ],
  "context": {
    "source": "endpoint_logs",
    "timeframe": "24h",
    "environment": "production",
    "user_context": {
      "user_id": "john.doe",
      "department": "finance",
      "access_level": "standard"
    }
  }
}
```

- *Response:**
```json
{
  "analysis_id": "analysis_1234567890",
  "confidence_score": 0.87,
  "threat_level": "high",
  "attack_patterns": [
    {
      "pattern": "insider_threat",
      "confidence": 0.75,
      "indicators": ["unusual_data_access", "privilege_escalation"],
      "mitre_tactics": ["TA0009", "TA0010"]
    }
  ],
  "recommendations": [
    "Investigate user activity logs",
    "Review access permissions",
    "Enable additional monitoring"
  ],
  "next_actions": [
    "behavioral_analysis",
    "access_review",
    "incident_escalation"
  ]
}
```

####  **Behavioral Analysis**
```http
POST /api/v1/intelligence/behavioral/analyze
Content-Type: application/json
Authorization: Bearer {token}

{
  "profile_id": "user_john_doe",
  "activity_data": {
    "login_frequency": 8.5,
    "access_patterns": 6.2,
    "data_transfer_volume": 4.8,
    "geolocation_variability": 3.1,
    "privilege_usage": 4.3,
    "command_sequence_complexity": 5.7
  },
  "timeframe": "7d"
}
```

####  **Threat Hunting**
```http
POST /api/v1/intelligence/threat-hunting/query
Content-Type: application/json
Authorization: Bearer {token}

{
  "query": "FIND processes WHERE name = 'suspicious.exe' AND network_connections > 10",
  "data_source": "endpoint_logs",
  "time_range": {
    "start": "2025-01-15T00:00:00Z",
    "end": "2025-01-15T23:59:59Z"
  }
}
```

- --

##  🏢 **Platform Management**

###  **Unified Platform API**

####  **Service Status**
```http
GET /api/v1/platform/services
Authorization: Bearer {token}
```

- *Response:**
```json
{
  "services": [
    {
      "service_id": "ptaas_orchestrator",
      "name": "PTaaS Orchestration Service",
      "type": "orchestration",
      "status": "running",
      "health": "healthy",
      "uptime_seconds": 86400,
      "restart_count": 0,
      "version": "3.0.0"
    }
  ],
  "total_count": 11,
  "by_type": {
    "core": 3,
    "analytics": 2,
    "security": 3,
    "intelligence": 3
  }
}
```

####  **Platform Health**
```http
GET /api/v1/platform/health
Authorization: Bearer {token}
```

####  **Platform Metrics**
```http
GET /api/v1/platform/metrics
Authorization: Bearer {token}
```

####  **Forensics Evidence Collection**
```http
POST /api/v1/platform/forensics/evidence
Content-Type: application/json
Authorization: Bearer {token}

{
  "case_id": "CASE-2025-001",
  "evidence_type": "network_capture",
  "source": "firewall_logs",
  "collection_method": "automated",
  "data": {
    "capture_file": "base64_encoded_pcap",
    "start_time": "2025-01-15T10:00:00Z",
    "end_time": "2025-01-15T10:05:00Z",
    "size_bytes": 1048576
  }
}
```

####  **Network Microsegmentation**
```http
POST /api/v1/platform/network/segments
Content-Type: application/json
Authorization: Bearer {token}

{
  "segment_name": "finance_dmz",
  "description": "DMZ for finance applications",
  "policies": [
    {
      "name": "pci_dss_compliance",
      "rules": [
        {
          "type": "time_based",
          "start_time": "08:00",
          "end_time": "18:00"
        },
        {
          "type": "user_role",
          "allowed_roles": ["finance", "admin"]
        }
      ]
    }
  ]
}
```

- --

##  🛡️ **Security & Rate Limiting**

###  **Rate Limits**
- **Default Tier**: 60 requests/minute, 1000 requests/hour
- **Enterprise Tier**: Custom limits based on subscription
- **Burst Allowance**: 2x normal rate for 60 seconds

###  **Rate Limit Headers**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642176000
X-RateLimit-Burst-Remaining: 30
```

###  **Security Headers**
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

###  **Request Validation**
- **Input Sanitization**: All inputs are validated and sanitized
- **Schema Validation**: Pydantic models for request/response
- **Size Limits**: 10MB max request size
- **Timeout**: 30 second default timeout

- --

##  ⚠️ **Error Handling**

###  **Error Response Format**
```json
{
  "error": "ValidationError",
  "message": "Invalid scan target format",
  "details": {
    "field": "targets[0].host",
    "issue": "Invalid IP address or hostname"
  },
  "request_id": "req_1234567890",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

###  **HTTP Status Codes**
- **200**: Success
- **201**: Created
- **400**: Bad Request (validation error)
- **401**: Unauthorized (invalid token)
- **403**: Forbidden (insufficient permissions)
- **404**: Not Found
- **409**: Conflict (resource already exists)
- **429**: Too Many Requests (rate limited)
- **500**: Internal Server Error
- **503**: Service Unavailable

###  **Common Error Types**
- `ValidationError`: Invalid request data
- `AuthenticationError`: Invalid or expired token
- `AuthorizationError`: Insufficient permissions
- `RateLimitError`: Too many requests
- `ResourceNotFound`: Requested resource doesn't exist
- `ServiceUnavailable`: Dependent service unavailable

- --

##  💡 **Examples**

###  **Complete Security Assessment Workflow**

```bash
# !/bin/bash
# Complete PTaaS workflow example

BASE_URL="http://localhost:8000/api/v1"
TOKEN="your-jwt-token"

# 1. Validate target
echo "🔍 Validating target..."
curl -X POST "$BASE_URL/ptaas/validate-target" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "host": "scanme.nmap.org",
    "ports": [22, 80, 443, 8080],
    "scan_profile": "comprehensive"
  }'

# 2. Create scan session
echo "🚀 Creating scan session..."
SESSION_RESPONSE=$(curl -X POST "$BASE_URL/ptaas/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "targets": [{
      "host": "scanme.nmap.org",
      "ports": [22, 80, 443, 8080],
      "scan_profile": "comprehensive",
      "stealth_mode": true
    }],
    "scan_type": "comprehensive",
    "metadata": {"project": "security_assessment"}
  }')

SESSION_ID=$(echo $SESSION_RESPONSE | jq -r '.session_id')
echo "📋 Session ID: $SESSION_ID"

# 3. Monitor scan progress
echo "⏳ Monitoring scan progress..."
while true; do
  STATUS_RESPONSE=$(curl -s "$BASE_URL/ptaas/sessions/$SESSION_ID" \
    -H "Authorization: Bearer $TOKEN")

  STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi

  sleep 30
done

# 4. Get results
echo "📊 Retrieving results..."
curl "$BASE_URL/ptaas/scan-results/$SESSION_ID?format=json" \
  -H "Authorization: Bearer $TOKEN" | jq '.'

# 5. Generate compliance report if needed
echo "📋 Generating compliance report..."
curl -X POST "$BASE_URL/ptaas/orchestration/compliance-scan" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "compliance_framework": "PCI-DSS",
    "targets": ["scanme.nmap.org"],
    "assessment_type": "focused"
  }'
```

###  **Automated Threat Hunting**

```python
import requests
import json

class XORBThreatHunter:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}

    def hunt_suspicious_activity(self, timeframe="24h"):
        """Hunt for suspicious activity patterns"""

        # Search for privilege escalation attempts
        escalation_query = {
            "query": "FIND events WHERE action = 'privilege_escalation' AND success = true",
            "data_source": "security_logs",
            "time_range": {"start": f"-{timeframe}", "end": "now"}
        }

        response = requests.post(
            f"{self.base_url}/intelligence/threat-hunting/query",
            headers=self.headers,
            json=escalation_query
        )

        return response.json()

    def analyze_behavioral_anomalies(self, user_id):
        """Analyze user behavioral patterns"""

        analysis_request = {
            "profile_id": user_id,
            "activity_data": {
                "login_frequency": 8.5,
                "access_patterns": 6.2,
                "data_transfer_volume": 4.8
            }
        }

        response = requests.post(
            f"{self.base_url}/intelligence/behavioral/analyze",
            headers=self.headers,
            json=analysis_request
        )

        return response.json()

# Usage
hunter = XORBThreatHunter("http://localhost:8000/api/v1", "your-token")
results = hunter.hunt_suspicious_activity("7d")
print(json.dumps(results, indent=2))
```

###  **Enterprise Compliance Automation**

```python
import asyncio
import aiohttp

class ComplianceAutomation:
    def __init__(self, api_base, token):
        self.api_base = api_base
        self.headers = {'Authorization': f'Bearer {token}'}

    async def run_compliance_suite(self, targets, frameworks):
        """Run comprehensive compliance assessment"""

        async with aiohttp.ClientSession() as session:
            tasks = []

            for framework in frameworks:
                task = self.run_compliance_scan(session, targets, framework)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

    async def run_compliance_scan(self, session, targets, framework):
        """Execute compliance scan for specific framework"""

        scan_config = {
            "compliance_framework": framework,
            "targets": targets,
            "scope": {"full_assessment": True},
            "assessment_type": "full"
        }

        async with session.post(
            f"{self.api_base}/ptaas/orchestration/compliance-scan",
            headers=self.headers,
            json=scan_config
        ) as response:
            return await response.json()

# Usage
async def main():
    automation = ComplianceAutomation("http://localhost:8000/api/v1", "your-token")

    targets = ["web.company.com", "api.company.com", "db.company.com"]
    frameworks = ["PCI-DSS", "HIPAA", "SOX"]

    results = await automation.run_compliance_suite(targets, frameworks)

    for result in results:
        print(f"Compliance scan initiated: {result['scan_id']}")

asyncio.run(main())
```

- --

##  📚 **Additional Resources**

- **[PTaaS Implementation Guide](../services/PTAAS_IMPLEMENTATION_SUMMARY.md)**
- **[Security Best Practices](../best-practices/)**
- **[Deployment Guide](../deployment/)**
- **[Architecture Overview](../architecture/)**

- --

##  🤝 **Support**

- **Documentation**: https://docs.xorb-security.com
- **Support**: enterprise@xorb-security.com
- **Community**: https://discord.gg/xorb-security
- **Issues**: https://github.com/xorb-security/xorb/issues

- --

- *© 2025 XORB Security, Inc. All rights reserved.**