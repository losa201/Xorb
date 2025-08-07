# XORB API Integration Guide

## Overview

This guide provides comprehensive documentation for integrating with the XORB Cybersecurity Platform API. The API is designed for autonomous security operations, orchestration, and multi-agent coordination.

## Quick Start

### 1. Authentication Setup

First, obtain your client credentials and certificate for mTLS authentication:

```bash
# Get your client certificate (provided by XORB admin)
curl -X POST https://api.xorb.security/v1/auth/certificate \
  -H "Content-Type: application/json" \
  -d '{"client_id":"your_client_id","client_secret":"your_secret"}'
```

### 2. Obtain Access Token

```bash
# Get JWT token
curl -X POST https://api.xorb.security/v1/auth/token \
  --cert client.crt --key client.key \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "scope": "agent:read task:submit security:read"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 86400,
  "scope": "agent:read task:submit security:read"
}
```

### 3. Make Authenticated Requests

```bash
# List active agents
curl -X GET https://api.xorb.security/v1/agents \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer your_jwt_token"
```

## Core API Modules

### Agent Management

#### Create Agent
```bash
curl -X POST https://api.xorb.security/v1/agents \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Threat Hunter Alpha",
    "agent_type": "threat_hunter",
    "capabilities": ["network_analysis", "malware_detection"],
    "configuration": {
      "scan_interval": 300,
      "alert_threshold": 0.8
    },
    "description": "Primary threat hunting agent"
  }'
```

#### Monitor Agent Status
```bash
# Get real-time agent status
curl -X GET https://api.xorb.security/v1/agents/{agent_id}/status \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN"
```

#### Send Agent Commands
```bash
curl -X POST https://api.xorb.security/v1/agents/{agent_id}/commands \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "run_scan",
    "parameters": {
      "target": "192.168.1.0/24",
      "scan_type": "comprehensive"
    },
    "timeout_seconds": 300
  }'
```

### Task Orchestration

#### Submit Task
```bash
curl -X POST https://api.xorb.security/v1/orchestration/tasks \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Network Vulnerability Scan",
    "task_type": "vulnerability_scan",
    "priority": "high",
    "parameters": {
      "target": "10.0.0.0/8",
      "scope": "internal_network",
      "options": {
        "deep_scan": true,
        "compliance_check": true
      }
    },
    "orchestration_strategy": "ai_optimized",
    "description": "Comprehensive internal network security assessment"
  }'
```

#### Monitor Task Progress
```bash
# Get task details and progress
curl -X GET https://api.xorb.security/v1/orchestration/tasks/{task_id} \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN"
```

#### Get Orchestration Metrics
```bash
curl -X GET https://api.xorb.security/v1/orchestration/metrics \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN"
```

### Security Operations

#### List Threats
```bash
# Get recent threats
curl -X GET "https://api.xorb.security/v1/security/threats?severity=high&hours_back=24" \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN"
```

#### Create Threat Alert
```bash
curl -X POST https://api.xorb.security/v1/security/threats \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Suspicious Network Activity",
    "description": "Unusual outbound connections detected",
    "severity": "high",
    "category": "network_intrusion",
    "source_system": "network_monitor",
    "affected_hosts": ["ws-001", "srv-005"],
    "indicators": [
      {
        "indicator_type": "ip",
        "value": "203.0.113.45",
        "confidence": 0.85,
        "source": "firewall_logs",
        "first_seen": "2024-01-15T10:30:00Z",
        "last_seen": "2024-01-15T11:45:00Z",
        "tags": ["external", "suspicious"]
      }
    ]
  }'
```

#### Respond to Threat
```bash
curl -X POST https://api.xorb.security/v1/security/threats/{threat_id}/respond \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "actions": ["block_ip", "quarantine_host"],
    "parameters": {
      "ip": "203.0.113.45",
      "host": "ws-001"
    },
    "auto_execute": true,
    "notify_team": true
  }'
```

### Intelligence Integration

#### Request AI Decision
```bash
curl -X POST https://api.xorb.security/v1/intelligence/decisions \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "decision_type": "threat_classification",
    "context": {
      "scenario": "suspicious_network_activity",
      "available_data": {
        "indicators": 5,
        "severity_score": 0.8,
        "affected_systems": 2
      },
      "urgency_level": "high",
      "confidence_threshold": 0.7
    },
    "model_preferences": ["claude_agent"],
    "explanation_required": true
  }'
```

#### Provide Learning Feedback
```bash
curl -X POST https://api.xorb.security/v1/intelligence/feedback \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "decision_id": "decision_uuid_here",
    "outcome": "success",
    "actual_result": {
      "threat_contained": true,
      "false_positive": false,
      "response_time_minutes": 5
    },
    "effectiveness_score": 0.9,
    "lessons_learned": [
      "Quick response prevented lateral movement",
      "IP blocking was highly effective"
    ]
  }'
```

#### Get Orchestration Brain Status
```bash
# Monitor Qwen3 orchestration brain
curl -X GET https://api.xorb.security/v1/intelligence/models/qwen3_orchestrator/brain-status \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN"
```

## SDK Examples

### Python SDK

```python
import requests
import json
from datetime import datetime

class XORBClient:
    def __init__(self, base_url, cert_path, key_path, token=None):
        self.base_url = base_url
        self.cert = (cert_path, key_path)
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({'Authorization': f'Bearer {token}'})
    
    def authenticate(self, client_id, client_secret):
        """Get JWT token"""
        response = self.session.post(
            f"{self.base_url}/auth/token",
            cert=self.cert,
            json={
                "client_id": client_id,
                "client_secret": client_secret
            }
        )
        response.raise_for_status()
        data = response.json()
        self.token = data['access_token']
        self.session.headers.update({'Authorization': f'Bearer {self.token}'})
        return data
    
    def create_agent(self, name, agent_type, capabilities=None):
        """Create a new agent"""
        payload = {
            "name": name,
            "agent_type": agent_type,
            "capabilities": capabilities or []
        }
        response = self.session.post(
            f"{self.base_url}/agents",
            cert=self.cert,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def submit_task(self, name, task_type, priority="medium", **kwargs):
        """Submit orchestration task"""
        payload = {
            "name": name,
            "task_type": task_type,
            "priority": priority,
            **kwargs
        }
        response = self.session.post(
            f"{self.base_url}/orchestration/tasks",
            cert=self.cert,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_threats(self, severity=None, hours_back=24):
        """Get security threats"""
        params = {"hours_back": hours_back}
        if severity:
            params["severity"] = severity
        
        response = self.session.get(
            f"{self.base_url}/security/threats",
            cert=self.cert,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def request_ai_decision(self, decision_type, context):
        """Request AI-driven decision"""
        payload = {
            "decision_type": decision_type,
            "context": context
        }
        response = self.session.post(
            f"{self.base_url}/intelligence/decisions",
            cert=self.cert,
            json=payload
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = XORBClient(
    base_url="https://api.xorb.security/v1",
    cert_path="client.crt",
    key_path="client.key"
)

# Authenticate
client.authenticate("your_client_id", "your_client_secret")

# Create threat hunting agent
agent = client.create_agent(
    name="Network Threat Hunter",
    agent_type="threat_hunter",
    capabilities=["network_analysis", "malware_detection"]
)
print(f"Created agent: {agent['id']}")

# Submit vulnerability scan task
task = client.submit_task(
    name="Weekly Vulnerability Scan",
    task_type="vulnerability_scan",
    priority="medium",
    parameters={
        "target": "production_network",
        "options": {"comprehensive": True}
    }
)
print(f"Submitted task: {task['id']}")

# Get high-severity threats
threats = client.get_threats(severity="high")
print(f"Found {len(threats['threats'])} high-severity threats")

# Request AI decision for threat response
if threats['threats']:
    threat = threats['threats'][0]
    decision = client.request_ai_decision(
        decision_type="response_strategy",
        context={
            "scenario": "threat_response",
            "available_data": {
                "threat_id": threat['id'],
                "severity": threat['severity'],
                "category": threat['category']
            },
            "urgency_level": "high"
        }
    )
    print(f"AI recommendation: {decision['recommendation']} (confidence: {decision['confidence_score']})")
```

### TypeScript SDK

```typescript
import axios, { AxiosInstance } from 'axios';
import https from 'https';
import fs from 'fs';

interface XORBConfig {
  baseURL: string;
  certPath: string;
  keyPath: string;
  caCertPath?: string;
}

class XORBClient {
  private client: AxiosInstance;
  private token?: string;

  constructor(config: XORBConfig) {
    const httpsAgent = new https.Agent({
      cert: fs.readFileSync(config.certPath),
      key: fs.readFileSync(config.keyPath),
      ca: config.caCertPath ? fs.readFileSync(config.caCertPath) : undefined,
    });

    this.client = axios.create({
      baseURL: config.baseURL,
      httpsAgent,
      timeout: 30000,
    });

    // Add token to requests if available
    this.client.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`;
      }
      return config;
    });
  }

  async authenticate(clientId: string, clientSecret: string): Promise<any> {
    const response = await this.client.post('/auth/token', {
      client_id: clientId,
      client_secret: clientSecret,
    });

    this.token = response.data.access_token;
    return response.data;
  }

  async createAgent(agentData: {
    name: string;
    agent_type: string;
    capabilities?: string[];
    configuration?: Record<string, any>;
  }): Promise<any> {
    const response = await this.client.post('/agents', agentData);
    return response.data;
  }

  async listAgents(filters?: {
    status?: string;
    agent_type?: string;
    page?: number;
    per_page?: number;
  }): Promise<any> {
    const response = await this.client.get('/agents', { params: filters });
    return response.data;
  }

  async submitTask(taskData: {
    name: string;
    task_type: string;
    priority?: string;
    parameters?: Record<string, any>;
    description?: string;
  }): Promise<any> {
    const response = await this.client.post('/orchestration/tasks', taskData);
    return response.data;
  }

  async getThreats(filters?: {
    severity?: string;
    status?: string;
    hours_back?: number;
    limit?: number;
  }): Promise<any> {
    const response = await this.client.get('/security/threats', { params: filters });
    return response.data;
  }

  async requestAIDecision(decisionRequest: {
    decision_type: string;
    context: Record<string, any>;
    model_preferences?: string[];
  }): Promise<any> {
    const response = await this.client.post('/intelligence/decisions', decisionRequest);
    return response.data;
  }
}

// Usage example
const client = new XORBClient({
  baseURL: 'https://api.xorb.security/v1',
  certPath: './certs/client.crt',
  keyPath: './certs/client.key',
  caCertPath: './certs/ca.crt'
});

// Async wrapper for demonstration
(async () => {
  try {
    // Authenticate
    await client.authenticate('your_client_id', 'your_client_secret');
    
    // Create agent
    const agent = await client.createAgent({
      name: 'Security Analyst Bot',
      agent_type: 'security_analyst',
      capabilities: ['log_analysis', 'threat_intelligence']
    });
    
    console.log('Agent created:', agent.id);
    
    // Submit task
    const task = await client.submitTask({
      name: 'Daily Security Scan',
      task_type: 'security_assessment',
      priority: 'medium',
      parameters: {
        scope: 'all_systems',
        include_compliance: true
      }
    });
    
    console.log('Task submitted:', task.id);
    
  } catch (error) {
    console.error('API Error:', error.response?.data || error.message);
  }
})();
```

## Error Handling

### Standard Error Response Format

```json
{
  "code": "AUTHENTICATION_FAILED",
  "message": "Invalid JWT token",
  "details": {
    "error_type": "token_expired",
    "expires_at": "2024-01-15T12:00:00Z"
  },
  "request_id": "req_123456789",
  "timestamp": "2024-01-15T12:30:00Z"
}
```

### Common HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (invalid/missing auth)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `409` - Conflict (resource state conflict)
- `429` - Rate Limited
- `500` - Internal Server Error

### Rate Limiting

Rate limits are enforced per client and role:

- **Admin**: 10,000 requests/minute
- **Orchestrator**: 5,000 requests/minute
- **Agent**: 2,000 requests/minute
- **Analyst**: 1,000 requests/minute
- **Readonly**: 500 requests/minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 5000
X-RateLimit-Remaining: 4999
X-RateLimit-Reset: 1642248000
```

## Security Best Practices

### 1. Certificate Management

- Store certificates securely
- Use hardware security modules (HSM) when possible
- Rotate certificates regularly
- Implement certificate pinning

### 2. Token Security

- Store JWT tokens securely (not in browser localStorage)
- Implement automatic token refresh
- Use short-lived tokens with refresh tokens
- Revoke tokens on logout

### 3. Request Security

- Validate all input data
- Use HTTPS for all communications
- Implement request signing for critical operations
- Log all security-related API calls

### 4. Error Handling

- Don't expose sensitive information in error messages
- Implement proper retry logic with exponential backoff
- Log errors for monitoring and debugging
- Implement circuit breaker patterns

## Monitoring and Observability

### Health Checks

```bash
# System health check
curl -X GET https://api.xorb.security/v1/telemetry/health \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN"
```

### Metrics Collection

```bash
# Get system metrics
curl -X GET https://api.xorb.security/v1/telemetry/metrics \
  --cert client.crt --key client.key \
  -H "Authorization: Bearer $TOKEN"
```

### Audit Logging

All API calls are automatically logged with:
- Request ID for tracing
- User/client identification
- Timestamp and duration
- Request/response details (excluding sensitive data)
- Security events and decisions

## Support and Documentation

- **API Documentation**: Available at `/docs` endpoint
- **OpenAPI Spec**: Download from `/openapi.json`
- **Status Page**: Monitor system status
- **Support**: Contact platform team for integration support

For additional examples and advanced integration patterns, see the complete SDK documentation and example applications in the XORB developer portal.