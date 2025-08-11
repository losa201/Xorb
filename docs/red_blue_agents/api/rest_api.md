#  REST API Reference

The XORB Red/Blue Agent Framework provides a comprehensive REST API for managing missions, agents, and system operations. This API follows RESTful principles and uses JSON for data exchange.

##  🌐 Base URL

```
Production: https://api.xorb-security.com/v1
Staging:    https://staging-api.xorb-security.com/v1
Local:      http://localhost:8000/api/v1
```

##  🔐 Authentication

The API uses JWT bearer token authentication for all endpoints except public health checks.

###  Obtain Access Token

```http
POST /auth/login
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here"
}
```

###  Using Access Token

Include the token in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

###  Refresh Token

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "refresh_token_here"
}
```

##  📊 Response Format

All API responses follow a consistent format:

###  Success Response
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789",
    "version": "1.0.0"
  }
}
```

###  Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "targets",
        "message": "Field is required"
      }
    ]
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

###  HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

##  🎯 Mission Management

###  Create Mission

Create a new red/blue team mission.

```http
POST /missions
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Web Application Security Assessment",
  "description": "Comprehensive security testing of the main web application",
  "environment": "staging",
  "objectives": [
    "Perform reconnaissance of target systems",
    "Identify and exploit web vulnerabilities",
    "Test detection capabilities",
    "Establish persistence mechanisms"
  ],
  "targets": [
    {
      "type": "web_application",
      "host": "webapp.staging.company.com",
      "ports": [80, 443],
      "url": "https://webapp.staging.company.com",
      "metadata": {
        "technology_stack": "PHP/MySQL",
        "authentication": "form_based"
      }
    }
  ],
  "constraints": {
    "max_duration_seconds": 7200,
    "stealth_mode": true,
    "excluded_techniques": ["exploit.web_sqli"],
    "resource_limits": {
      "max_agents": 5,
      "max_sandboxes": 10
    }
  },
  "red_team_config": {
    "attack_intensity": "medium",
    "evasion_level": "high",
    "persistence_required": true
  },
  "blue_team_config": {
    "detection_mode": "active",
    "response_enabled": true,
    "hunting_enabled": true
  },
  "notifications": {
    "webhooks": ["https://your-webhook.com/missions"],
    "email": ["security-team@company.com"],
    "slack_channel": "#security-ops"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "mission_id": "mission_123e4567-e89b-12d3-a456-426614174000",
    "status": "pending",
    "created_at": "2024-01-15T10:30:00Z",
    "estimated_duration": 7200,
    "planned_agents": [
      {
        "agent_type": "red_recon",
        "estimated_tasks": 3
      },
      {
        "agent_type": "red_exploit",
        "estimated_tasks": 2
      },
      {
        "agent_type": "blue_detect",
        "estimated_tasks": 5
      }
    ]
  }
}
```

###  Start Mission

Start execution of a pending mission.

```http
POST /missions/{mission_id}/start
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "mission_id": "mission_123e4567-e89b-12d3-a456-426614174000",
    "status": "executing",
    "started_at": "2024-01-15T10:35:00Z",
    "agents_deployed": 3,
    "sandboxes_created": 3
  }
}
```

###  Get Mission Status

Retrieve detailed status of a mission.

```http
GET /missions/{mission_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "mission_id": "mission_123e4567-e89b-12d3-a456-426614174000",
    "name": "Web Application Security Assessment",
    "status": "executing",
    "progress": {
      "completion_percentage": 65,
      "current_phase": "exploitation",
      "estimated_completion": "2024-01-15T12:35:00Z"
    },
    "agents": [
      {
        "agent_id": "agent_red_recon_001",
        "agent_type": "red_recon",
        "status": "completed",
        "tasks_completed": 3,
        "tasks_total": 3,
        "success_rate": 100,
        "sandbox_id": "sandbox_001"
      },
      {
        "agent_id": "agent_red_exploit_002",
        "agent_type": "red_exploit",
        "status": "running",
        "tasks_completed": 1,
        "tasks_total": 2,
        "success_rate": 100,
        "current_task": "exploit.web_sqli",
        "sandbox_id": "sandbox_002"
      }
    ],
    "results": {
      "vulnerabilities_found": 3,
      "techniques_successful": 5,
      "detection_events": 2,
      "response_actions": 1
    },
    "telemetry": {
      "total_requests": 1247,
      "data_collected_mb": 15.2,
      "execution_time_seconds": 1850
    }
  }
}
```

###  List Missions

List missions with optional filtering.

```http
GET /missions?status=executing&environment=staging&limit=20&offset=0
Authorization: Bearer {token}
```

**Query Parameters:**
- `status` - Filter by mission status (pending, executing, completed, failed, cancelled)
- `environment` - Filter by environment (production, staging, development, cyber_range)
- `agent_type` - Filter by agent type
- `created_after` - ISO 8601 timestamp
- `created_before` - ISO 8601 timestamp
- `limit` - Number of results per page (default: 50, max: 100)
- `offset` - Number of results to skip

**Response:**
```json
{
  "success": true,
  "data": {
    "missions": [
      {
        "mission_id": "mission_123e4567-e89b-12d3-a456-426614174000",
        "name": "Web Application Security Assessment",
        "status": "executing",
        "environment": "staging",
        "created_at": "2024-01-15T10:30:00Z",
        "progress": 65,
        "agents_count": 3
      }
    ],
    "pagination": {
      "total": 1,
      "limit": 20,
      "offset": 0,
      "has_more": false
    }
  }
}
```

###  Stop Mission

Stop a running mission.

```http
POST /missions/{mission_id}/stop
Content-Type: application/json
Authorization: Bearer {token}

{
  "reason": "Emergency stop requested by user",
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "mission_id": "mission_123e4567-e89b-12d3-a456-426614174000",
    "status": "cancelled",
    "stopped_at": "2024-01-15T11:45:00Z",
    "agents_stopped": 3,
    "sandboxes_cleaned": 3
  }
}
```

###  Get Mission Results

Retrieve detailed results from a completed mission.

```http
GET /missions/{mission_id}/results?format=json
Authorization: Bearer {token}
```

**Query Parameters:**
- `format` - Response format (json, pdf, csv, xml)
- `include_raw_data` - Include raw telemetry data (true/false)
- `include_screenshots` - Include screenshot data (true/false)

**Response:**
```json
{
  "success": true,
  "data": {
    "mission_summary": {
      "mission_id": "mission_123e4567-e89b-12d3-a456-426614174000",
      "name": "Web Application Security Assessment",
      "duration_seconds": 7200,
      "completion_status": "completed",
      "success_rate": 85
    },
    "vulnerabilities": [
      {
        "id": "vuln_001",
        "type": "SQL Injection",
        "severity": "high",
        "cvss_score": 8.5,
        "cwe_id": "CWE-89",
        "location": "https://webapp.staging.company.com/login.php",
        "parameter": "username",
        "discovered_by": "agent_red_exploit_002",
        "discovery_time": "2024-01-15T11:15:00Z",
        "exploit_successful": true,
        "detection_bypassed": false,
        "response_triggered": true
      }
    ],
    "techniques_executed": [
      {
        "technique_id": "recon.port_scan",
        "agent_id": "agent_red_recon_001",
        "execution_time": "2024-01-15T10:35:00Z",
        "success": true,
        "duration_seconds": 45,
        "results": {
          "open_ports": [22, 80, 443],
          "services_identified": 3
        }
      }
    ],
    "detection_events": [
      {
        "event_id": "detect_001",
        "type": "anomalous_network_traffic",
        "severity": "medium",
        "detected_by": "agent_blue_detect_003",
        "detection_time": "2024-01-15T11:20:00Z",
        "source_ip": "172.20.0.5",
        "technique_detected": "exploit.web_sqli",
        "confidence": 0.89,
        "response_action": "alert_generated"
      }
    ],
    "response_actions": [
      {
        "action_id": "response_001",
        "type": "block_ip",
        "triggered_by": "detect_001",
        "executed_by": "agent_blue_respond_004",
        "execution_time": "2024-01-15T11:22:00Z",
        "success": true,
        "details": {
          "blocked_ip": "172.20.0.5",
          "duration_seconds": 3600
        }
      }
    ]
  }
}
```

##  🤖 Agent Management

###  List Agents

List available agents and their current status.

```http
GET /agents?type=red_team&status=available
Authorization: Bearer {token}
```

**Query Parameters:**
- `type` - Filter by agent type (red_team, blue_team)
- `status` - Filter by status (available, assigned, running, error)
- `capability` - Filter by technique capability
- `environment` - Filter by supported environment

**Response:**
```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "agent_type": "red_recon",
        "name": "Reconnaissance Agent",
        "status": "available",
        "capabilities": [
          "recon.port_scan",
          "recon.service_enum",
          "recon.web_crawl"
        ],
        "supported_environments": ["staging", "development", "cyber_range"],
        "resource_requirements": {
          "cpu_cores": 1.0,
          "memory_mb": 512,
          "disk_mb": 1024
        },
        "current_assignments": 0,
        "max_concurrent_missions": 3
      }
    ]
  }
}
```

###  Get Agent Details

Get detailed information about a specific agent type.

```http
GET /agents/{agent_type}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "agent_type": "red_exploit",
    "name": "Exploitation Agent",
    "description": "Specializes in initial access and privilege escalation techniques",
    "version": "2.1.0",
    "supported_categories": [
      "initial_access",
      "privilege_escalation",
      "credential_access"
    ],
    "techniques": [
      {
        "technique_id": "exploit.web_sqli",
        "name": "SQL Injection",
        "description": "Attempt SQL injection attacks on web applications",
        "mitre_id": "T1190",
        "risk_level": "high",
        "parameters": [
          {
            "name": "url",
            "type": "string",
            "required": true,
            "description": "Target URL with vulnerable parameter"
          },
          {
            "name": "parameter",
            "type": "string",
            "required": true,
            "description": "Vulnerable parameter name"
          }
        ]
      }
    ],
    "statistics": {
      "total_missions": 127,
      "success_rate": 78.5,
      "average_execution_time": 450,
      "last_updated": "2024-01-15T08:00:00Z"
    }
  }
}
```

###  Execute Agent Task

Execute a specific technique with an agent (for testing purposes).

```http
POST /agents/{agent_type}/execute
Content-Type: application/json
Authorization: Bearer {token}

{
  "technique_id": "recon.port_scan",
  "parameters": {
    "target": "scanme.nmap.org",
    "ports": "1-1000",
    "scan_type": "tcp"
  },
  "environment": "development",
  "timeout_seconds": 300
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
    "status": "completed",
    "execution_time": 67.5,
    "results": {
      "target": "scanme.nmap.org",
      "open_ports": [22, 80, 443, 9929, 31337],
      "scan_duration": 65.2,
      "total_ports_scanned": 1000
    },
    "telemetry": {
      "bytes_sent": 50000,
      "bytes_received": 15000,
      "packets_sent": 1000
    }
  }
}
```

##  📦 Sandbox Management

###  List Sandboxes

List active sandbox environments.

```http
GET /sandboxes?mission_id={mission_id}&status=running
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sandboxes": [
      {
        "sandbox_id": "sandbox_001",
        "mission_id": "mission_123e4567-e89b-12d3-a456-426614174000",
        "agent_type": "red_recon",
        "status": "running",
        "container_id": "c4f5e6d7e8f9",
        "container_name": "xorb-red-recon-sandbox_001",
        "created_at": "2024-01-15T10:35:00Z",
        "expires_at": "2024-01-15T12:35:00Z",
        "resource_usage": {
          "cpu_percent": 15.2,
          "memory_mb": 256,
          "disk_mb": 512,
          "network_rx_bytes": 1048576,
          "network_tx_bytes": 524288
        },
        "network": {
          "ip_address": "172.20.0.5",
          "ports": {
            "22": 32768
          },
          "isolation_mode": "bridge"
        }
      }
    ]
  }
}
```

###  Get Sandbox Details

Get detailed information about a specific sandbox.

```http
GET /sandboxes/{sandbox_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sandbox_id": "sandbox_001",
    "configuration": {
      "sandbox_type": "docker_sidecar",
      "image": "xorb/red-recon:latest",
      "resource_constraints": {
        "cpu_cores": 1.0,
        "memory_mb": 512,
        "disk_mb": 1024
      },
      "network_policy": {
        "isolation_mode": "bridge",
        "allowed_outbound": ["*.staging.company.com"],
        "blocked_outbound": ["*.production.company.com"]
      },
      "ttl_seconds": 7200
    },
    "runtime_info": {
      "container_id": "c4f5e6d7e8f9",
      "started_at": "2024-01-15T10:35:00Z",
      "last_activity": "2024-01-15T11:30:00Z",
      "process_count": 12,
      "active_connections": 3
    },
    "logs": [
      "2024-01-15T10:35:00Z INFO Agent initialized successfully",
      "2024-01-15T10:35:15Z INFO Starting port scan of target",
      "2024-01-15T10:36:22Z INFO Port scan completed, 5 open ports found"
    ]
  }
}
```

###  Stop Sandbox

Stop and remove a sandbox environment.

```http
POST /sandboxes/{sandbox_id}/stop
Content-Type: application/json
Authorization: Bearer {token}

{
  "force": false,
  "cleanup_data": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sandbox_id": "sandbox_001",
    "status": "stopped",
    "stopped_at": "2024-01-15T11:45:00Z",
    "cleanup_performed": true
  }
}
```

##  🎛️ Capability Management

###  List Techniques

List available techniques with filtering options.

```http
GET /capabilities/techniques?category=reconnaissance&environment=staging
Authorization: Bearer {token}
```

**Query Parameters:**
- `category` - Filter by MITRE ATT&CK category
- `environment` - Filter by allowed environment
- `risk_level` - Filter by risk level (low, medium, high, critical)
- `platform` - Filter by supported platform (linux, windows, macos)
- `search` - Search by name or description

**Response:**
```json
{
  "success": true,
  "data": {
    "techniques": [
      {
        "technique_id": "recon.port_scan",
        "name": "Port Scanning",
        "category": "reconnaissance",
        "description": "Discover open ports and services on target systems",
        "mitre_id": "T1046",
        "risk_level": "low",
        "stealth_level": "medium",
        "detection_difficulty": "medium",
        "platforms": ["linux", "windows", "macos"],
        "parameters": [
          {
            "name": "target",
            "type": "string",
            "required": true,
            "description": "Target IP address or hostname"
          }
        ],
        "dependencies": [],
        "allowed_environments": ["staging", "development", "cyber_range"]
      }
    ],
    "total": 45,
    "filtered": 12
  }
}
```

###  Get Technique Details

Get detailed information about a specific technique.

```http
GET /capabilities/techniques/{technique_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "technique_id": "exploit.web_sqli",
    "name": "SQL Injection",
    "category": "initial_access",
    "description": "Attempt SQL injection attacks on web applications",
    "mitre_id": "T1190",
    "risk_level": "high",
    "stealth_level": "medium",
    "detection_difficulty": "medium",
    "platforms": ["linux", "windows", "macos"],
    "parameters": [
      {
        "name": "url",
        "type": "string",
        "required": true,
        "description": "Target URL with vulnerable parameter",
        "constraints": {
          "pattern": "^https?://.*"
        }
      },
      {
        "name": "parameter",
        "type": "string",
        "required": true,
        "description": "Vulnerable parameter name"
      },
      {
        "name": "technique",
        "type": "string",
        "required": false,
        "default": "boolean",
        "description": "SQL injection technique",
        "constraints": {
          "choices": ["boolean", "union", "time", "error"]
        }
      }
    ],
    "dependencies": ["recon.web_crawl"],
    "environment_policies": {
      "production": "denied",
      "staging": "denied",
      "development": "allowed",
      "cyber_range": "allowed"
    },
    "statistics": {
      "success_rate": 72.5,
      "average_execution_time": 120,
      "total_executions": 1456,
      "last_30_days_executions": 89
    }
  }
}
```

###  Check Technique Permission

Check if a technique is allowed in a specific environment.

```http
GET /capabilities/techniques/{technique_id}/permission?environment=staging
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "technique_id": "exploit.web_sqli",
    "environment": "staging",
    "allowed": false,
    "reason": "Technique explicitly denied in staging environment",
    "alternative_techniques": [
      "recon.web_crawl",
      "recon.vulnerability_scan"
    ]
  }
}
```

###  List Environment Policies

List capability policies for different environments.

```http
GET /capabilities/environments
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "environments": [
      {
        "environment": "production",
        "description": "Production environment with strict security controls",
        "allowed_categories": [
          "detection",
          "analysis",
          "mitigation",
          "recovery"
        ],
        "denied_techniques": ["*"],
        "allowed_techniques": [
          "detect.network_anomaly",
          "detect.process_monitoring"
        ],
        "max_risk_level": "medium",
        "max_concurrent_agents": 5,
        "sandbox_constraints": {
          "network_isolation": true,
          "resource_limits": {
            "cpu_cores": 2,
            "memory_mb": 1024
          }
        }
      }
    ]
  }
}
```

##  📈 Telemetry & Analytics

###  Get Mission Analytics

Retrieve analytics data for missions.

```http
GET /analytics/missions?start_date=2024-01-01&end_date=2024-01-31
Authorization: Bearer {token}
```

**Query Parameters:**
- `start_date` - Start date (YYYY-MM-DD)
- `end_date` - End date (YYYY-MM-DD)
- `environment` - Filter by environment
- `agent_type` - Filter by agent type
- `aggregation` - Time aggregation (hour, day, week, month)

**Response:**
```json
{
  "success": true,
  "data": {
    "summary": {
      "total_missions": 156,
      "successful_missions": 134,
      "failed_missions": 22,
      "success_rate": 85.9,
      "average_duration": 3847,
      "total_vulnerabilities_found": 89,
      "total_detections": 234
    },
    "time_series": [
      {
        "date": "2024-01-15",
        "missions_started": 8,
        "missions_completed": 7,
        "success_rate": 87.5,
        "vulnerabilities_found": 4,
        "detections": 12
      }
    ],
    "top_techniques": [
      {
        "technique_id": "recon.port_scan",
        "executions": 145,
        "success_rate": 98.6,
        "average_duration": 67
      }
    ]
  }
}
```

###  Get Agent Performance

Retrieve performance metrics for agents.

```http
GET /analytics/agents/{agent_type}/performance?days=30
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "agent_type": "red_exploit",
    "performance_metrics": {
      "total_tasks": 234,
      "successful_tasks": 187,
      "success_rate": 79.9,
      "average_execution_time": 245,
      "error_rate": 5.1
    },
    "technique_performance": [
      {
        "technique_id": "exploit.web_sqli",
        "executions": 67,
        "success_rate": 73.1,
        "average_duration": 125,
        "error_types": [
          {
            "error": "Connection timeout",
            "count": 8
          }
        ]
      }
    ],
    "trending": {
      "success_rate_trend": "+2.3%",
      "execution_time_trend": "-5.7%",
      "error_rate_trend": "-1.2%"
    }
  }
}
```

###  Export Telemetry Data

Export raw telemetry data for analysis.

```http
POST /analytics/export
Content-Type: application/json
Authorization: Bearer {token}

{
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59Z",
  "data_types": ["mission_events", "agent_telemetry", "technique_executions"],
  "format": "csv",
  "filters": {
    "environment": ["staging", "development"],
    "agent_types": ["red_exploit", "blue_detect"]
  },
  "include_sensitive_data": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "export_id": "export_123e4567-e89b-12d3-a456-426614174000",
    "status": "processing",
    "estimated_completion": "2024-01-15T11:45:00Z",
    "download_url": null,
    "expires_at": "2024-01-22T11:45:00Z"
  }
}
```

##  🔧 System Management

###  System Health

Check overall system health.

```http
GET /health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T11:30:00Z",
    "version": "2.1.0",
    "uptime_seconds": 86400,
    "components": {
      "database": {
        "status": "healthy",
        "response_time_ms": 5,
        "connections": 12
      },
      "redis": {
        "status": "healthy",
        "response_time_ms": 2,
        "memory_usage_mb": 128
      },
      "scheduler": {
        "status": "healthy",
        "active_missions": 3,
        "pending_tasks": 7
      },
      "sandbox_orchestrator": {
        "status": "healthy",
        "active_sandboxes": 8,
        "available_resources": "85%"
      }
    }
  }
}
```

###  System Statistics

Get comprehensive system statistics.

```http
GET /system/stats
Authorization: Bearer {token}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "resource_usage": {
      "cpu_percent": 23.5,
      "memory_percent": 67.2,
      "disk_percent": 45.8,
      "network_rx_mbps": 5.2,
      "network_tx_mbps": 3.8
    },
    "mission_stats": {
      "total_missions": 1456,
      "active_missions": 8,
      "missions_today": 23,
      "average_duration": 3600
    },
    "agent_stats": {
      "total_agents": 8,
      "active_agents": 12,
      "agents_by_type": {
        "red_recon": 3,
        "red_exploit": 2,
        "blue_detect": 4,
        "blue_hunt": 3
      }
    },
    "sandbox_stats": {
      "total_sandboxes": 25,
      "active_sandboxes": 12,
      "resource_utilization": 68.5,
      "cleanup_events_today": 15
    }
  }
}
```

##  🔄 WebSocket API

For real-time updates, the framework provides WebSocket endpoints.

###  Connect to WebSocket

```javascript
const ws = new WebSocket('wss://api.xorb-security.com/v1/ws?token=your_jwt_token');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');

    // Subscribe to mission updates
    ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'missions',
        mission_id: 'mission_123e4567-e89b-12d3-a456-426614174000'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received update:', data);
};
```

###  WebSocket Message Types

####  Mission Status Updates
```json
{
  "type": "mission_update",
  "mission_id": "mission_123e4567-e89b-12d3-a456-426614174000",
  "status": "executing",
  "progress": 75,
  "agent_updates": [
    {
      "agent_id": "agent_red_exploit_002",
      "status": "running",
      "current_task": "exploit.web_sqli"
    }
  ]
}
```

####  Agent Events
```json
{
  "type": "agent_event",
  "agent_id": "agent_red_exploit_002",
  "event": "task_completed",
  "timestamp": "2024-01-15T11:45:00Z",
  "data": {
    "task_id": "task_123",
    "technique_id": "exploit.web_sqli",
    "success": true,
    "execution_time": 125
  }
}
```

##  📝 Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Authenticated Users**: 1000 requests per hour, 60 per minute
- **Premium Users**: 5000 requests per hour, 200 per minute
- **Enterprise Users**: Custom limits

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
X-RateLimit-Window: 3600
```

##  🧪 Testing & Examples

###  Python Example

```python
import requests
import json

#  Authentication
auth_response = requests.post('http://localhost:8000/api/v1/auth/login', json={
    'username': 'your-username',
    'password': 'your-password'
})
token = auth_response.json()['data']['access_token']

#  Create mission
mission_config = {
    'name': 'API Test Mission',
    'environment': 'development',
    'objectives': ['Test API integration'],
    'targets': [{'host': 'scanme.nmap.org', 'ports': [80, 443]}]
}

headers = {'Authorization': f'Bearer {token}'}
mission_response = requests.post(
    'http://localhost:8000/api/v1/missions',
    json=mission_config,
    headers=headers
)
mission_id = mission_response.json()['data']['mission_id']

#  Start mission
start_response = requests.post(
    f'http://localhost:8000/api/v1/missions/{mission_id}/start',
    headers=headers
)

print(f"Mission {mission_id} started successfully")
```

###  curl Examples

```bash
#  Get access token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}'

#  Create mission
curl -X POST http://localhost:8000/api/v1/missions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "Test Mission",
    "environment": "development",
    "objectives": ["Test curl integration"],
    "targets": [{"host": "scanme.nmap.org"}]
  }'

#  Check mission status
curl http://localhost:8000/api/v1/missions/MISSION_ID \
  -H "Authorization: Bearer YOUR_TOKEN"
```

##  🔍 Error Codes

###  Authentication Errors
- `AUTH_INVALID_CREDENTIALS` - Invalid username or password
- `AUTH_TOKEN_EXPIRED` - JWT token has expired
- `AUTH_TOKEN_INVALID` - Invalid JWT token format
- `AUTH_INSUFFICIENT_PERMISSIONS` - User lacks required permissions

###  Validation Errors
- `VALIDATION_REQUIRED_FIELD` - Required field is missing
- `VALIDATION_INVALID_FORMAT` - Field format is invalid
- `VALIDATION_CONSTRAINT_VIOLATION` - Field violates constraints
- `VALIDATION_DEPENDENCY_MISSING` - Required dependency not satisfied

###  Resource Errors
- `RESOURCE_NOT_FOUND` - Requested resource does not exist
- `RESOURCE_CONFLICT` - Resource conflict (e.g., duplicate name)
- `RESOURCE_QUOTA_EXCEEDED` - Resource quota exceeded
- `RESOURCE_UNAVAILABLE` - Resource temporarily unavailable

###  System Errors
- `SYSTEM_MAINTENANCE` - System under maintenance
- `SYSTEM_OVERLOADED` - System resources exhausted
- `SYSTEM_DATABASE_ERROR` - Database connection error
- `SYSTEM_INTERNAL_ERROR` - Unexpected internal error

---

For more information, see the [Python SDK documentation](./python_sdk.md) and [WebSocket API reference](./websocket_api.md).