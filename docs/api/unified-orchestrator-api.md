# Unified Orchestrator API Documentation

## Overview

The XORB Platform's Unified Orchestrator provides comprehensive service orchestration, workflow management, and system monitoring capabilities. This API consolidates service management, workflow execution, and operational monitoring into a single, powerful orchestration engine.

## Base URL
```
Production: https://api.verteidiq.com/orchestrator
Staging: https://staging-api.verteidiq.com/orchestrator
Development: http://localhost:8000/orchestrator
```

## Core Concepts

### Services
Services are managed components within the XORB platform. Each service has:
- **Service Definition**: Configuration and dependencies
- **Service Instance**: Runtime state and metrics
- **Health Monitoring**: Continuous health checks
- **Lifecycle Management**: Start, stop, restart operations

### Workflows
Workflows are automated sequences of tasks that can:
- **Execute in sequence or parallel**
- **Handle dependencies between tasks**
- **Retry failed operations**
- **Trigger on events or schedules**
- **Provide comprehensive monitoring**

### Task Executors
Task executors implement specific operation types:
- **Vulnerability Scanning**
- **Compliance Checking**
- **Threat Analysis**
- **Report Generation**
- **Notification Delivery**

---

## Endpoints

### Service Management

#### GET /orchestrator/services
List all registered services and their status.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "services": [
    {
      "service_id": "api-gateway",
      "name": "API Gateway Service",
      "service_type": "core",
      "status": "running",
      "start_time": "2024-01-15T10:00:00Z",
      "last_health_check": "2024-01-15T14:00:00Z",
      "restart_count": 0,
      "dependencies": ["database", "redis"],
      "resource_usage": {
        "cpu_percent": 15.2,
        "memory_mb": 256,
        "disk_io_mb": 12.5
      }
    }
  ],
  "summary": {
    "total_services": 12,
    "running_services": 10,
    "failed_services": 1,
    "maintenance_services": 1
  }
}
```

#### POST /orchestrator/services
Register a new service.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "service_id": "threat-analyzer",
  "name": "Threat Analysis Service",
  "service_type": "intelligence",
  "module_path": "xorb.intelligence.threat_analyzer",
  "class_name": "ThreatAnalyzer",
  "dependencies": ["database", "ml-models"],
  "config": {
    "max_concurrent_scans": 10,
    "timeout_minutes": 30
  },
  "health_check_url": "http://threat-analyzer:8001/health",
  "startup_timeout": 60,
  "restart_policy": "on-failure",
  "max_restarts": 3
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "service_id": "threat-analyzer",
  "message": "Service registered successfully"
}
```

#### POST /orchestrator/services/{service_id}/start
Start a specific service.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "success": true,
  "service_id": "threat-analyzer",
  "status": "running",
  "start_time": "2024-01-15T14:05:00Z",
  "message": "Service started successfully"
}
```

#### POST /orchestrator/services/{service_id}/stop
Stop a specific service.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "success": true,
  "service_id": "threat-analyzer",
  "status": "stopped",
  "stop_time": "2024-01-15T14:10:00Z",
  "message": "Service stopped successfully"
}
```

#### POST /orchestrator/services/{service_id}/restart
Restart a specific service.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "success": true,
  "service_id": "threat-analyzer",
  "status": "running",
  "restart_time": "2024-01-15T14:15:00Z",
  "restart_count": 1,
  "message": "Service restarted successfully"
}
```

#### GET /orchestrator/services/{service_id}/health
Get detailed health information for a service.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "service_id": "threat-analyzer",
  "status": "healthy",
  "last_check": "2024-01-15T14:00:00Z",
  "uptime_seconds": 3600,
  "health_checks": {
    "database_connection": "healthy",
    "external_api": "healthy",
    "memory_usage": "warning",
    "disk_space": "healthy"
  },
  "metrics": {
    "requests_per_minute": 45,
    "average_response_time_ms": 150,
    "error_rate_percent": 0.5,
    "queue_size": 3
  },
  "resource_usage": {
    "cpu_percent": 25.5,
    "memory_mb": 512,
    "memory_percent": 15.2,
    "disk_io_mb": 8.3,
    "network_io_mb": 2.1
  }
}
```

### Workflow Management

#### GET /orchestrator/workflows
List all workflow definitions.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `enabled`: Filter by enabled status (true/false)
- `type`: Filter by workflow type
- `limit`: Number of results (default: 100)
- `offset`: Pagination offset (default: 0)

**Response (200 OK):**
```json
{
  "workflows": [
    {
      "id": "vulnerability-scan-workflow",
      "name": "Vulnerability Scanning Workflow",
      "description": "Automated vulnerability scanning and reporting",
      "version": "1.2.0",
      "enabled": true,
      "created_at": "2024-01-10T09:00:00Z",
      "updated_at": "2024-01-15T12:00:00Z",
      "task_count": 5,
      "average_duration_minutes": 45,
      "success_rate_percent": 98.5,
      "last_execution": "2024-01-15T13:30:00Z",
      "triggers": [
        {
          "type": "scheduled",
          "schedule": "0 2 * * *",
          "description": "Daily at 2 AM"
        },
        {
          "type": "webhook",
          "url": "/webhooks/vulnerability-scan",
          "description": "Manual trigger"
        }
      ]
    }
  ],
  "pagination": {
    "total": 25,
    "limit": 100,
    "offset": 0,
    "has_more": false
  }
}
```

#### POST /orchestrator/workflows
Create a new workflow definition.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "id": "compliance-check-workflow",
  "name": "Compliance Check Workflow",
  "description": "Automated compliance verification workflow",
  "version": "1.0.0",
  "tasks": [
    {
      "id": "collect-data",
      "name": "Collect System Data",
      "task_type": "data_collection",
      "description": "Collect system configuration data",
      "parameters": {
        "data_sources": ["system_configs", "user_permissions", "network_configs"],
        "output_format": "json"
      },
      "dependencies": [],
      "timeout_minutes": 10,
      "retry_count": 3,
      "retry_delay_seconds": 30
    },
    {
      "id": "analyze-compliance",
      "name": "Analyze Compliance",
      "task_type": "compliance_check",
      "description": "Check data against compliance frameworks",
      "parameters": {
        "frameworks": ["SOC2", "ISO27001", "PCI-DSS"],
        "severity_threshold": "medium"
      },
      "dependencies": ["collect-data"],
      "timeout_minutes": 20,
      "retry_count": 2,
      "retry_delay_seconds": 60
    },
    {
      "id": "generate-report",
      "name": "Generate Compliance Report",
      "task_type": "report_generation",
      "description": "Generate detailed compliance report",
      "parameters": {
        "format": "pdf",
        "include_recommendations": true,
        "email_recipients": ["compliance@company.com"]
      },
      "dependencies": ["analyze-compliance"],
      "timeout_minutes": 5,
      "retry_count": 1,
      "retry_delay_seconds": 30
    }
  ],
  "triggers": [
    {
      "type": "scheduled",
      "schedule": "0 9 1 * *",
      "description": "Monthly on 1st at 9 AM"
    }
  ],
  "variables": {
    "notification_enabled": true,
    "report_retention_days": 90
  },
  "notifications": {
    "email": ["admin@company.com"],
    "slack": ["#compliance-team"],
    "webhook": ["https://company.com/webhooks/compliance"]
  },
  "sla_minutes": 60,
  "tags": ["compliance", "monthly", "automated"]
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "workflow_id": "compliance-check-workflow",
  "message": "Workflow created successfully"
}
```

#### POST /orchestrator/workflows/{workflow_id}/execute
Execute a workflow.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "trigger_data": {
    "initiated_by": "user@company.com",
    "priority": "high",
    "target_systems": ["prod-web-01", "prod-db-01"],
    "custom_parameters": {
      "scan_depth": "comprehensive",
      "include_experimental_checks": false
    }
  },
  "variables": {
    "notification_enabled": true,
    "custom_timeout_minutes": 120
  }
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "execution_id": "exec_1234567890abcdef",
  "workflow_id": "compliance-check-workflow",
  "status": "pending",
  "started_at": "2024-01-15T14:20:00Z",
  "estimated_duration_minutes": 60,
  "message": "Workflow execution started"
}
```

#### GET /orchestrator/workflows/executions
List workflow executions.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `workflow_id`: Filter by workflow ID
- `status`: Filter by status (pending, running, completed, failed, cancelled)
- `limit`: Number of results (default: 50)
- `offset`: Pagination offset (default: 0)
- `start_time`: Filter executions after this time (ISO 8601)
- `end_time`: Filter executions before this time (ISO 8601)

**Response (200 OK):**
```json
{
  "executions": [
    {
      "execution_id": "exec_1234567890abcdef",
      "workflow_id": "compliance-check-workflow",
      "workflow_name": "Compliance Check Workflow",
      "status": "running",
      "started_at": "2024-01-15T14:20:00Z",
      "estimated_completion": "2024-01-15T15:20:00Z",
      "triggered_by": "user@company.com",
      "progress": {
        "completed_tasks": 1,
        "total_tasks": 3,
        "current_task": "analyze-compliance",
        "percent_complete": 33
      },
      "task_results": {
        "collect-data": {
          "status": "completed",
          "started_at": "2024-01-15T14:20:00Z",
          "completed_at": "2024-01-15T14:25:00Z",
          "duration_seconds": 300,
          "result": {
            "data_collected": 1247,
            "sources_processed": 3,
            "output_size_mb": 2.4
          }
        }
      }
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

#### GET /orchestrator/workflows/executions/{execution_id}
Get detailed execution information.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "execution_id": "exec_1234567890abcdef",
  "workflow_id": "compliance-check-workflow",
  "workflow_name": "Compliance Check Workflow",
  "status": "completed",
  "started_at": "2024-01-15T14:20:00Z",
  "completed_at": "2024-01-15T15:15:00Z",
  "duration_seconds": 3300,
  "triggered_by": "user@company.com",
  "trigger_data": {
    "priority": "high",
    "target_systems": ["prod-web-01", "prod-db-01"]
  },
  "variables": {
    "notification_enabled": true,
    "custom_timeout_minutes": 120
  },
  "task_results": {
    "collect-data": {
      "status": "completed",
      "started_at": "2024-01-15T14:20:00Z",
      "completed_at": "2024-01-15T14:25:00Z",
      "duration_seconds": 300,
      "retry_count": 0,
      "result": {
        "data_collected": 1247,
        "sources_processed": 3,
        "output_size_mb": 2.4,
        "summary": "Successfully collected system data from all sources"
      }
    },
    "analyze-compliance": {
      "status": "completed",
      "started_at": "2024-01-15T14:25:00Z",
      "completed_at": "2024-01-15T15:10:00Z",
      "duration_seconds": 2700,
      "retry_count": 1,
      "result": {
        "frameworks_checked": ["SOC2", "ISO27001", "PCI-DSS"],
        "total_controls": 342,
        "passed_controls": 318,
        "failed_controls": 24,
        "compliance_score": 92.98,
        "critical_findings": 3,
        "high_findings": 8,
        "medium_findings": 13
      }
    },
    "generate-report": {
      "status": "completed",
      "started_at": "2024-01-15T15:10:00Z",
      "completed_at": "2024-01-15T15:15:00Z",
      "duration_seconds": 300,
      "retry_count": 0,
      "result": {
        "report_generated": true,
        "report_url": "https://reports.xorb.platform/compliance/20240115_141500.pdf",
        "report_size_mb": 12.7,
        "emails_sent": 2,
        "notifications_delivered": 3
      }
    }
  },
  "summary": {
    "success": true,
    "total_tasks": 3,
    "completed_tasks": 3,
    "failed_tasks": 0,
    "total_retries": 1,
    "compliance_score": 92.98,
    "critical_issues": 3
  }
}
```

#### POST /orchestrator/workflows/executions/{execution_id}/cancel
Cancel a running workflow execution.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "reason": "User requested cancellation",
  "force": false
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "execution_id": "exec_1234567890abcdef",
  "status": "cancelled",
  "cancelled_at": "2024-01-15T14:45:00Z",
  "message": "Workflow execution cancelled successfully"
}
```

### Task Executor Management

#### GET /orchestrator/task-executors
List all registered task executors.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "task_executors": [
    {
      "task_type": "vulnerability_scan",
      "executor_class": "VulnerabilityScanner",
      "health_status": "healthy",
      "capabilities": {
        "max_concurrent_tasks": 5,
        "supported_formats": ["nessus", "openvas", "qualys"],
        "average_execution_time_minutes": 30
      },
      "metrics": {
        "total_executions": 1247,
        "successful_executions": 1198,
        "failed_executions": 49,
        "success_rate_percent": 96.07,
        "average_duration_seconds": 1800
      },
      "last_execution": "2024-01-15T13:45:00Z"
    }
  ]
}
```

#### POST /orchestrator/task-executors
Register a new task executor.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "task_type": "custom_analysis",
  "executor_class": "CustomAnalysisExecutor",
  "module_path": "custom.analyzers.custom_executor",
  "capabilities": {
    "max_concurrent_tasks": 3,
    "supported_formats": ["json", "xml"],
    "requires_gpu": false,
    "memory_requirements_mb": 512
  },
  "configuration": {
    "timeout_minutes": 45,
    "retry_attempts": 2,
    "output_retention_days": 30
  }
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "task_type": "custom_analysis",
  "message": "Task executor registered successfully"
}
```

### System Monitoring

#### GET /orchestrator/metrics
Get orchestrator system metrics.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "timestamp": "2024-01-15T14:00:00Z",
  "orchestrator_status": "healthy",
  "system_metrics": {
    "total_services": 12,
    "running_services": 10,
    "failed_services": 1,
    "maintenance_services": 1,
    "total_workflows": 25,
    "active_workflows": 3,
    "completed_workflows_today": 47,
    "failed_workflows_today": 2,
    "average_task_duration_minutes": 23.5,
    "system_load_percent": 45.2,
    "memory_usage_percent": 67.8,
    "disk_usage_percent": 23.1
  },
  "performance_metrics": {
    "requests_per_minute": 150,
    "average_response_time_ms": 245,
    "error_rate_percent": 0.8,
    "queue_sizes": {
      "high_priority": 2,
      "medium_priority": 8,
      "low_priority": 15
    }
  },
  "resource_utilization": {
    "cpu_cores_used": 6.8,
    "cpu_cores_total": 16,
    "memory_gb_used": 21.7,
    "memory_gb_total": 32,
    "disk_gb_used": 147.3,
    "disk_gb_total": 512,
    "network_mbps_in": 45.2,
    "network_mbps_out": 38.7
  }
}
```

#### GET /orchestrator/health
Get orchestrator health status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "connection_pool": "ok"
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 3,
      "memory_usage_percent": 45.2
    },
    "temporal": {
      "status": "healthy",
      "response_time_ms": 8,
      "workflow_queue_size": 25
    },
    "service_mesh": {
      "status": "warning",
      "healthy_services": 10,
      "unhealthy_services": 1,
      "message": "One service in maintenance mode"
    }
  },
  "metrics": {
    "active_workflows": 3,
    "running_services": 10,
    "system_load": 45.2,
    "memory_usage": 67.8
  }
}
```

---

## Workflow Definition Schema

### Workflow Structure
```json
{
  "id": "string",
  "name": "string", 
  "description": "string",
  "version": "string",
  "enabled": true,
  "tasks": [
    {
      "id": "string",
      "name": "string",
      "task_type": "vulnerability_scan|compliance_check|threat_analysis|report_generation|notification|data_collection|remediation|approval|integration|service_fusion",
      "description": "string",
      "parameters": {},
      "dependencies": ["task_id"],
      "timeout_minutes": 30,
      "retry_count": 2,
      "retry_delay_seconds": 60,
      "condition": "optional_condition_expression",
      "on_success": ["next_task_id"],
      "on_failure": ["failure_handler_task_id"],
      "parallel_execution": false
    }
  ],
  "triggers": [
    {
      "type": "manual|scheduled|event_driven|api_trigger|webhook",
      "schedule": "cron_expression",
      "webhook_url": "string",
      "event_filters": {}
    }
  ],
  "variables": {},
  "notifications": {
    "email": ["email@example.com"],
    "slack": ["#channel"],
    "webhook": ["https://webhook.url"]
  },
  "sla_minutes": 60,
  "tags": ["tag1", "tag2"],
  "retry_policy": {
    "max_retries": 3,
    "backoff_strategy": "exponential",
    "retry_conditions": ["timeout", "network_error"]
  }
}
```

### Task Types and Parameters

#### vulnerability_scan
```json
{
  "task_type": "vulnerability_scan",
  "parameters": {
    "targets": ["192.168.1.0/24", "example.com"],
    "scan_type": "comprehensive|quick|deep",
    "scanner_engine": "nessus|openvas|qualys",
    "include_web_scan": true,
    "include_network_scan": true,
    "severity_threshold": "low|medium|high|critical",
    "output_format": "json|xml|pdf",
    "max_scan_time_minutes": 120
  }
}
```

#### compliance_check
```json
{
  "task_type": "compliance_check",
  "parameters": {
    "frameworks": ["SOC2", "ISO27001", "PCI-DSS", "HIPAA"],
    "target_systems": ["web-servers", "databases"],
    "check_types": ["configuration", "access_controls", "encryption"],
    "severity_threshold": "medium",
    "generate_evidence": true,
    "output_format": "json|pdf|csv"
  }
}
```

#### threat_analysis
```json
{
  "task_type": "threat_analysis",
  "parameters": {
    "data_sources": ["network_logs", "system_logs", "security_events"],
    "analysis_models": ["ml_anomaly", "rule_based", "behavioral"],
    "time_window_hours": 24,
    "confidence_threshold": 0.8,
    "include_iot_analysis": true,
    "correlation_rules": ["lateral_movement", "privilege_escalation"]
  }
}
```

---

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error_code": "WORKFLOW_EXECUTION_FAILED",
  "message": "Workflow execution failed due to task timeout",
  "details": {
    "execution_id": "exec_1234567890abcdef",
    "failed_task": "analyze-compliance",
    "failure_reason": "Task exceeded timeout of 30 minutes",
    "retry_available": true
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `SERVICE_NOT_FOUND` | Service ID not registered |
| `SERVICE_START_FAILED` | Failed to start service |
| `SERVICE_STOP_FAILED` | Failed to stop service |
| `DEPENDENCY_NOT_MET` | Service dependencies not satisfied |
| `WORKFLOW_NOT_FOUND` | Workflow ID not found |
| `WORKFLOW_EXECUTION_FAILED` | Workflow execution encountered error |
| `TASK_TIMEOUT` | Task exceeded timeout limit |
| `TASK_EXECUTOR_NOT_FOUND` | No executor registered for task type |
| `INVALID_WORKFLOW_DEFINITION` | Workflow definition validation failed |
| `INSUFFICIENT_RESOURCES` | Not enough system resources |
| `ORCHESTRATOR_UNAVAILABLE` | Orchestrator service unavailable |

---

## SDK Examples

### Python SDK
```python
from xorb_sdk import XORBOrchestratorClient

# Initialize client
orchestrator = XORBOrchestratorClient(
    base_url="https://api.verteidiq.com",
    access_token="your_access_token"
)

# List services
services = await orchestrator.services.list()
print(f"Running services: {services.summary.running_services}")

# Start a service
result = await orchestrator.services.start("threat-analyzer")
print(f"Service started: {result.success}")

# Execute workflow
execution = await orchestrator.workflows.execute(
    workflow_id="vulnerability-scan-workflow",
    trigger_data={
        "targets": ["192.168.1.0/24"],
        "priority": "high"
    }
)
print(f"Execution ID: {execution.execution_id}")

# Monitor execution
status = await orchestrator.workflows.get_execution(execution.execution_id)
print(f"Status: {status.status}, Progress: {status.progress.percent_complete}%")
```

### JavaScript SDK
```javascript
import { XORBOrchestratorClient } from '@xorb/orchestrator-sdk';

// Initialize client
const orchestrator = new XORBOrchestratorClient({
  baseURL: 'https://api.verteidiq.com',
  accessToken: 'your_access_token'
});

// List workflows
const workflows = await orchestrator.workflows.list();
console.log(`Total workflows: ${workflows.pagination.total}`);

// Execute workflow
const execution = await orchestrator.workflows.execute(
  'compliance-check-workflow',
  {
    triggerData: {
      priority: 'medium',
      targetSystems: ['prod-web-01']
    }
  }
);

// Poll for completion
const pollExecution = async (executionId) => {
  const status = await orchestrator.workflows.getExecution(executionId);
  
  if (status.status === 'completed') {
    console.log('Workflow completed successfully');
    console.log(`Compliance score: ${status.summary.compliance_score}`);
  } else if (status.status === 'failed') {
    console.log('Workflow failed');
  } else {
    setTimeout(() => pollExecution(executionId), 5000);
  }
};

pollExecution(execution.executionId);
```

---

## Monitoring and Observability

### Prometheus Metrics
```
# Service metrics
xorb_orchestrator_services_total
xorb_orchestrator_services_running
xorb_orchestrator_services_failed

# Workflow metrics  
xorb_orchestrator_workflows_total
xorb_orchestrator_workflows_active
xorb_orchestrator_workflow_duration_seconds
xorb_orchestrator_workflow_success_rate

# Task metrics
xorb_orchestrator_tasks_executed_total
xorb_orchestrator_task_duration_seconds
xorb_orchestrator_task_retry_count

# System metrics
xorb_orchestrator_cpu_usage_percent
xorb_orchestrator_memory_usage_percent
xorb_orchestrator_queue_size
```

### Grafana Dashboards
Pre-built dashboards available for:
- **Service Health Overview**
- **Workflow Execution Metrics**
- **Task Performance Analysis**
- **System Resource Utilization**
- **Error Rate and SLA Monitoring**

### Alert Rules
Recommended alert rules:
- Service down for > 5 minutes
- Workflow failure rate > 5%
- Task execution time > SLA
- System resource usage > 80%
- Queue size growing consistently

---

## Best Practices

### Workflow Design
1. **Keep tasks atomic and idempotent**
2. **Use meaningful task and workflow names**
3. **Set appropriate timeouts and retry policies**
4. **Design for failure - handle error conditions**
5. **Use dependencies to control execution order**
6. **Include health checks in long-running tasks**

### Service Management
1. **Define clear service dependencies**
2. **Implement proper health checks**
3. **Set resource limits and monitoring**
4. **Use graceful shutdown procedures**
5. **Monitor service metrics and logs**

### Performance Optimization
1. **Use parallel execution where possible**
2. **Optimize task executor resource usage**
3. **Implement proper caching strategies**
4. **Monitor and tune system resources**
5. **Use priority queues for critical workflows**

---

## Migration Guide

### From Legacy Orchestrators

1. **Update imports:**
   ```python
   # Old
   from api.app.infrastructure.service_orchestrator import ServiceOrchestrator
   
   # New
   from orchestrator.unified_orchestrator import UnifiedOrchestrator
   ```

2. **Update service registration:**
   ```python
   # Old
   service_orchestrator.register_service(service_config)
   
   # New
   unified_orchestrator.register_service(service_definition)
   ```

3. **Update workflow definitions:**
   - Convert to new JSON schema format
   - Update task type enumeration
   - Add new monitoring and retry capabilities

### Configuration Migration
Update environment variables and configuration:
```bash
# New unified configuration
ORCHESTRATOR_MAX_WORKERS=10
ORCHESTRATOR_QUEUE_SIZE=1000
ORCHESTRATOR_HEALTH_CHECK_INTERVAL=30
```

---

## Support and Resources

- **Documentation**: https://docs.verteidiq.com/orchestrator
- **API Reference**: https://api-docs.verteidiq.com/orchestrator
- **Workflow Examples**: https://github.com/verteidiq/workflow-examples
- **SDKs**: https://github.com/verteidiq/sdks
- **Support**: support@verteidiq.com