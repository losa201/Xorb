# 🌐 Xorb PTaaS API Documentation

## Overview

The Xorb PTaaS API provides comprehensive access to penetration testing as a service functionality, including asset management, security scanning, bug bounty programs, and researcher gamification.

**Base URL**: `http://localhost:8000`  
**Version**: 2.0.0  
**Authentication**: JWT Bearer tokens (when configured)

## 🚀 Quick Start

### Test API Connectivity

```bash
# Health check
curl http://localhost:8000/health

# API status
curl http://localhost:8000/api/v1/status

# Root endpoint
curl http://localhost:8000/
```

Expected response:
```json
{
  "service": "Xorb PTaaS API",
  "version": "2.0.0",
  "status": "operational",
  "message": "Welcome to Xorb Penetration Testing as a Service",
  "timestamp": "2025-07-24T19:37:22.123456"
}
```

## 📊 Core Endpoints

### System Endpoints

#### `GET /` - Root Information
Returns basic service information and status.

**Response:**
```json
{
  "service": "Xorb PTaaS API",
  "version": "2.0.0", 
  "status": "operational",
  "message": "Welcome to Xorb Penetration Testing as a Service",
  "timestamp": "2025-07-24T19:37:22.123456"
}
```

#### `GET /health` - Health Check
Service health status for load balancers and monitoring.

**Response:**
```json
{
  "status": "healthy",
  "service": "xorb-api",
  "timestamp": "2025-07-24T19:37:22.123456"
}
```

#### `GET /api/v1/status` - API Status
Detailed API and service connectivity status.

**Response:**
```json
{
  "api_status": "operational",
  "services": {
    "database": "ready",
    "redis": "ready",
    "nats": "ready"
  },
  "version": "2.0.0",
  "timestamp": "2025-07-24T19:37:22.123456"
}
```

## 🎯 Asset Management

### `GET /api/v1/assets` - List Assets
Retrieve all managed assets for security testing.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20)
- `type` (string): Asset type filter
- `status` (string): Asset status filter

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/assets?page=1&limit=10&type=web"
```

**Response:**
```json
{
  "assets": [],
  "total": 0,
  "page": 1,
  "limit": 10,
  "message": "Asset management system ready"
}
```

**Future Implementation:**
```json
{
  "assets": [
    {
      "id": "asset_001",
      "name": "Corporate Website",
      "type": "web",
      "url": "https://example.com",
      "status": "active",
      "last_scan": "2024-07-24T10:00:00Z",
      "risk_score": 7.5,
      "tags": ["production", "critical"]
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 10
}
```

### `POST /api/v1/assets` - Create Asset
Add a new asset for security testing.

**Request Body:**
```json
{
  "name": "New Web Application",
  "type": "web",
  "url": "https://new-app.example.com",
  "description": "Customer-facing web application",
  "tags": ["production", "customer-facing"],
  "scan_config": {
    "depth": "full",
    "exclude_paths": ["/admin", "/internal"]
  }
}
```

## 🔍 Security Scanning

### `GET /api/v1/scans` - List Scans
Retrieve security scan history and status.

**Query Parameters:**
- `asset_id` (string): Filter by asset ID
- `status` (string): Filter by scan status
- `from_date` (string): Start date filter (ISO format)
- `to_date` (string): End date filter (ISO format)

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/scans?status=completed&limit=5"
```

**Response:**
```json
{
  "scans": [],
  "total": 0,
  "message": "Security scanning system ready"
}
```

**Future Implementation:**
```json
{
  "scans": [
    {
      "id": "scan_001",
      "asset_id": "asset_001",
      "type": "vulnerability_scan",
      "status": "completed",
      "started_at": "2024-07-24T10:00:00Z",
      "completed_at": "2024-07-24T10:30:00Z",
      "findings_count": 12,
      "critical_findings": 2,
      "high_findings": 5,
      "medium_findings": 3,
      "low_findings": 2
    }
  ],
  "total": 1
}
```

### `POST /api/v1/scans` - Initiate Scan
Start a new security scan on an asset.

**Request Body:**
```json
{
  "asset_id": "asset_001",
  "scan_type": "full_scan",
  "priority": "high",
  "config": {
    "depth": "deep",
    "include_authenticated": true,
    "custom_payloads": true
  }
}
```

## 🔎 Findings Management

### `GET /api/v1/findings` - List Findings
Retrieve security findings from scans.

**Query Parameters:**
- `scan_id` (string): Filter by scan ID
- `asset_id` (string): Filter by asset ID
- `severity` (string): Filter by severity level
- `status` (string): Filter by finding status
- `category` (string): Filter by vulnerability category

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/findings?severity=critical&status=open"
```

**Response:**
```json
{
  "findings": [],
  "total": 0,
  "message": "Findings management system ready"
}
```

**Future Implementation:**
```json
{
  "findings": [
    {
      "id": "finding_001",
      "scan_id": "scan_001",
      "asset_id": "asset_001",
      "title": "SQL Injection Vulnerability",
      "severity": "critical",
      "category": "injection",
      "status": "open",
      "description": "SQL injection found in login form",
      "location": "/login?user=",
      "evidence": {
        "request": "POST /login",
        "payload": "admin' OR '1'='1",
        "response_code": 200
      },
      "remediation": "Use parameterized queries",
      "discovered_at": "2024-07-24T10:15:00Z",
      "cvss_score": 9.1
    }
  ],
  "total": 1
}
```

### `PUT /api/v1/findings/{finding_id}` - Update Finding
Update finding status, notes, or remediation info.

**Request Body:**
```json
{
  "status": "in_progress",
  "assigned_to": "security_team",
  "notes": "Working on remediation",
  "remediation_eta": "2024-07-25T12:00:00Z"
}
```

## 🎮 Gamification System

### `GET /api/gamification/leaderboard` - Researcher Leaderboard
Get the current researcher leaderboard rankings.

**Query Parameters:**
- `period` (string): Time period (daily, weekly, monthly, all-time)
- `limit` (int): Number of top researchers to return

**Example Request:**
```bash
curl "http://localhost:8000/api/gamification/leaderboard?period=monthly&limit=10"
```

**Response:**
```json
{
  "leaderboard": [],
  "message": "Gamification system ready"
}
```

**Future Implementation:**
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "researcher_id": "researcher_001",
      "username": "security_ace",
      "points": 2450,
      "level": "Expert",
      "findings_submitted": 23,
      "findings_validated": 18,
      "critical_findings": 3,
      "bounty_earned": 12500.00,
      "streak_days": 15
    }
  ],
  "period": "monthly",
  "total_researchers": 50
}
```

### `GET /api/gamification/profile/{researcher_id}` - Researcher Profile
Get detailed researcher profile and statistics.

**Response:**
```json
{
  "researcher_id": "researcher_001",
  "username": "security_ace", 
  "level": "Expert",
  "total_points": 15750,
  "current_streak": 15,
  "longest_streak": 42,
  "badges": [
    "SQL Injection Specialist",
    "XSS Hunter",
    "Critical Finder"
  ],
  "statistics": {
    "total_findings": 89,
    "validated_findings": 72,
    "false_positives": 3,
    "average_severity": 6.2,
    "total_bounty": 45250.00
  }
}
```

## 📋 Compliance & Reporting

### `GET /api/compliance/status` - Compliance Status
Get current SOC 2 and compliance readiness status.

**Response:**
```json
{
  "compliance_status": "ready",
  "soc2_readiness": "green",
  "message": "Compliance automation system ready"
}
```

**Future Implementation:**
```json
{
  "compliance_status": "compliant",
  "soc2_readiness": "green",
  "last_audit": "2024-06-15T00:00:00Z",
  "next_audit": "2024-12-15T00:00:00Z",
  "controls": {
    "cc1_control_environment": "implemented",
    "cc2_communication": "implemented", 
    "cc3_risk_assessment": "implemented",
    "cc4_monitoring": "implemented",
    "cc5_control_activities": "implemented"
  },
  "evidence_collection": {
    "access_logs": "automated",
    "change_management": "automated",
    "incident_response": "documented"
  }
}
```

## 🔐 Authentication & Authorization

### JWT Token Authentication

**Login Request:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "secure_password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "refresh_token": "def50200..."
}
```

**Authenticated Request:**
```bash
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  http://localhost:8000/api/v1/protected-endpoint
```

## 🛠️ Development & Testing

### API Testing Examples

**Python with requests:**
```python
import requests

# Test API connectivity
response = requests.get("http://localhost:8000/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# List assets
assets = requests.get("http://localhost:8000/api/v1/assets")
print(f"Assets: {assets.json()}")
```

**cURL Testing Script:**
```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

echo "Testing Xorb PTaaS API..."

# Health check
echo "1. Health Check:"
curl -s "$BASE_URL/health" | jq .

# API status
echo "2. API Status:"
curl -s "$BASE_URL/api/v1/status" | jq .

# List assets
echo "3. Assets:"
curl -s "$BASE_URL/api/v1/assets" | jq .

# Leaderboard
echo "4. Leaderboard:"
curl -s "$BASE_URL/api/gamification/leaderboard" | jq .
```

## 🔍 Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "asset_id",
      "issue": "required field missing"
    },
    "timestamp": "2024-07-24T10:00:00Z",
    "request_id": "req_12345"
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Unprocessable Entity |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## 📊 Rate Limiting

### Current Limits (Future Implementation)

| Endpoint Category | Requests per Minute |
|------------------|---------------------|
| **Authentication** | 5 per IP |
| **Asset Management** | 60 per user |
| **Scan Operations** | 10 per user |
| **Reporting** | 30 per user |
| **Public Endpoints** | 100 per IP |

### Rate Limit Headers

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1627846260
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | API bind address | 0.0.0.0 |
| `API_PORT` | API port | 8000 |
| `JWT_SECRET_KEY` | JWT signing key | (required) |
| `DATABASE_URL` | Database connection | (required) |
| `REDIS_URL` | Redis connection | (required) |

### Feature Flags

| Flag | Description | Status |
|------|-------------|--------|
| `ENABLE_AUTHENTICATION` | JWT auth required | Planned |
| `ENABLE_RATE_LIMITING` | API rate limits | Planned |
| `ENABLE_WEBHOOKS` | Event webhooks | Planned |
| `ENABLE_ASYNC_SCANNING` | Background scans | Planned |

## 📚 OpenAPI Specification

The API follows OpenAPI 3.0 specification. Access the interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## 🚀 What's Next?

The current API provides basic endpoints for system health and status. Full functionality will be implemented with:

1. **Database integration** for asset and scan management
2. **Authentication system** with JWT tokens
3. **Background task processing** for scans
4. **WebSocket support** for real-time updates
5. **Webhook system** for external integrations
6. **Advanced filtering and pagination**
7. **Export capabilities** (PDF, CSV, JSON)

---

**Need help?** Check the deployment verification results or monitoring dashboards for system status.