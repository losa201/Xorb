"""
Automated Workflow Orchestration Engine
Advanced workflow automation for security operations and compliance management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from abc import ABC, abstractmethod
from redis import asyncio as aioredis
import aiohttp
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
import yaml

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskType(Enum):
    VULNERABILITY_SCAN = "vulnerability_scan"
    COMPLIANCE_CHECK = "compliance_check"
    THREAT_ANALYSIS = "threat_analysis"
    REPORT_GENERATION = "report_generation"
    NOTIFICATION = "notification"
    DATA_COLLECTION = "data_collection"
    REMEDIATION = "remediation"
    APPROVAL = "approval"
    INTEGRATION = "integration"

class TriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    API_TRIGGER = "api_trigger"
    WEBHOOK = "webhook"

@dataclass
class WorkflowTask:
    id: str
    name: str
    task_type: TaskType
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    timeout_minutes: int
    retry_count: int
    retry_delay_seconds: int
    condition: Optional[str] = None
    on_success: Optional[List[str]] = None
    on_failure: Optional[List[str]] = None
    parallel_execution: bool = False

@dataclass
class WorkflowDefinition:
    id: str
    name: str
    description: str
    version: str
    tasks: List[WorkflowTask]
    triggers: List[Dict[str, Any]]
    variables: Dict[str, Any]
    notifications: Dict[str, List[str]]
    sla_minutes: Optional[int] = None
    tags: List[str] = None
    enabled: bool = True

@dataclass
class WorkflowExecution:
    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    triggered_by: str
    trigger_data: Dict[str, Any]
    task_results: Dict[str, Any]
    error_message: Optional[str] = None
    variables: Dict[str, Any] = None

class TaskExecutor(ABC):
    """Abstract base class for task executors"""
    
    @abstractmethod
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow task"""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate task parameters"""
        pass

class VulnerabilityScanExecutor(TaskExecutor):
    """Executor for vulnerability scanning tasks"""
    
    def __init__(self, scanner_service_url: str):
        self.scanner_service_url = scanner_service_url
        
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vulnerability scan"""
        try:
            params = task.parameters
            target = params.get('target')
            scan_type = params.get('scan_type', 'comprehensive')
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    'target': target,
                    'scan_type': scan_type,
                    'options': params.get('options', {}),
                    'workflow_execution_id': context['execution_id']
                }
                
                async with session.post(f"{self.scanner_service_url}/api/v1/scans", 
                                      json=payload) as response:
                    if response.status == 200:
                        scan_data = await response.json()
                        scan_id = scan_data['scan_id']
                        
                        # Poll for completion
                        return await self._wait_for_scan_completion(session, scan_id)
                    else:
                        raise Exception(f"Failed to start scan: {response.status}")
                        
        except Exception as e:
            logger.error(f"Vulnerability scan failed: {e}")
            raise
            
    async def _wait_for_scan_completion(self, session: aiohttp.ClientSession, scan_id: str) -> Dict[str, Any]:
        """Wait for scan completion"""
        max_attempts = 120  # 2 hours with 1-minute intervals
        attempt = 0
        
        while attempt < max_attempts:
            try:
                async with session.get(f"{self.scanner_service_url}/api/v1/scans/{scan_id}/status") as response:
                    if response.status == 200:
                        status_data = await response.json()
                        
                        if status_data['status'] == 'completed':
                            # Get full results
                            async with session.get(f"{self.scanner_service_url}/api/v1/scans/{scan_id}/results") as results_response:
                                if results_response.status == 200:
                                    results = await results_response.json()
                                    return {
                                        'scan_id': scan_id,
                                        'status': 'completed',
                                        'results': results,
                                        'vulnerabilities_found': len(results.get('vulnerabilities', [])),
                                        'severity_breakdown': results.get('severity_breakdown', {}),
                                        'scan_duration': results.get('duration_seconds', 0)
                                    }
                                    
                        elif status_data['status'] == 'failed':
                            raise Exception(f"Scan failed: {status_data.get('error', 'Unknown error')}")
                            
            except Exception as e:
                logger.warning(f"Error checking scan status: {e}")
                
            await asyncio.sleep(60)  # Check every minute
            attempt += 1
            
        raise Exception(f"Scan timeout after {max_attempts} minutes")
        
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate scan parameters"""
        required_fields = ['target']
        return all(field in parameters for field in required_fields)

class ComplianceCheckExecutor(TaskExecutor):
    """Executor for compliance checking tasks"""
    
    def __init__(self, compliance_service_url: str):
        self.compliance_service_url = compliance_service_url
        
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compliance check"""
        try:
            params = task.parameters
            framework = params.get('framework')  # GDPR, NIS2, SOC2, etc.
            scope = params.get('scope', 'full')
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    'framework': framework,
                    'scope': scope,
                    'assets': params.get('assets', []),
                    'workflow_execution_id': context['execution_id']
                }
                
                async with session.post(f"{self.compliance_service_url}/api/v1/compliance/check", 
                                      json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'framework': framework,
                            'compliance_score': result.get('score', 0),
                            'passed_controls': result.get('passed_controls', []),
                            'failed_controls': result.get('failed_controls', []),
                            'recommendations': result.get('recommendations', []),
                            'risk_level': result.get('risk_level', 'unknown')
                        }
                    else:
                        raise Exception(f"Compliance check failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            raise
            
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate compliance check parameters"""
        required_fields = ['framework']
        return all(field in parameters for field in required_fields)

class NotificationExecutor(TaskExecutor):
    """Executor for notification tasks"""
    
    def __init__(self, notification_service_url: str):
        self.notification_service_url = notification_service_url
        
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification task"""
        try:
            params = task.parameters
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    'recipients': params.get('recipients', []),
                    'subject': params.get('subject', ''),
                    'message': self._render_message(params.get('template', ''), context),
                    'channels': params.get('channels', ['email']),
                    'priority': params.get('priority', 'normal'),
                    'attachments': params.get('attachments', [])
                }
                
                async with session.post(f"{self.notification_service_url}/api/v1/notifications/send", 
                                      json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'notification_id': result.get('id'),
                            'status': 'sent',
                            'recipients_count': len(payload['recipients']),
                            'channels_used': payload['channels']
                        }
                    else:
                        raise Exception(f"Notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            raise
            
    def _render_message(self, template: str, context: Dict[str, Any]) -> str:
        """Render message template with context variables"""
        try:
            # Simple template rendering - in production, use Jinja2 or similar
            message = template
            for key, value in context.get('variables', {}).items():
                message = message.replace(f"{{{key}}}", str(value))
            return message
        except Exception:
            return template
            
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate notification parameters"""
        required_fields = ['recipients', 'subject', 'template']
        return all(field in parameters for field in required_fields)

class WorkflowOrchestrator:
    """Main workflow orchestration engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.executors: Dict[TaskType, TaskExecutor] = {}
        self.redis_client = None
        self.temporal_client = None
        self.running = False
        
    async def initialize(self):
        """Initialize the orchestrator"""
        logger.info("Initializing Workflow Orchestrator...")
        
        # Initialize Redis for state management
        redis_url = self.config.get('redis_url', 'redis://localhost:6379')
        self.redis_client = await aioredis.from_url(redis_url)
        
        # Initialize Temporal client
        temporal_host = self.config.get('temporal_host', 'localhost:7233')
        self.temporal_client = await Client.connect(temporal_host)
        
        # Initialize task executors
        await self._initialize_executors()
        
        # Load workflow definitions
        await self._load_workflows()
        
        # Start scheduler
        asyncio.create_task(self._scheduler_loop())
        
        self.running = True
        logger.info("Workflow Orchestrator initialized successfully")
        
    async def _initialize_executors(self):
        """Initialize task executors"""
        service_urls = self.config.get('service_urls', {})
        
        # Initialize executors
        self.executors[TaskType.VULNERABILITY_SCAN] = VulnerabilityScanExecutor(
            service_urls.get('scanner', 'http://localhost:8001')
        )
        self.executors[TaskType.COMPLIANCE_CHECK] = ComplianceCheckExecutor(
            service_urls.get('compliance', 'http://localhost:8002')
        )
        self.executors[TaskType.NOTIFICATION] = NotificationExecutor(
            service_urls.get('notifications', 'http://localhost:8003')
        )
        
    async def _load_workflows(self):
        """Load workflow definitions from configuration"""
        workflows_dir = self.config.get('workflows_dir', './workflows')
        
        # Load default workflows
        default_workflows = self._get_default_workflows()
        for workflow in default_workflows:
            self.workflows[workflow.id] = workflow
            
        logger.info(f"Loaded {len(self.workflows)} workflow definitions")
        
    def _get_default_workflows(self) -> List[WorkflowDefinition]:
        """Get default workflow definitions"""
        return [
            WorkflowDefinition(
                id="daily_security_scan",
                name="Daily Security Scan",
                description="Comprehensive daily security assessment",
                version="1.0.0",
                tasks=[
                    WorkflowTask(
                        id="vulnerability_scan",
                        name="Vulnerability Scan",
                        task_type=TaskType.VULNERABILITY_SCAN,
                        description="Scan all critical assets for vulnerabilities",
                        parameters={
                            "target": "all_critical_assets",
                            "scan_type": "comprehensive",
                            "options": {"deep_scan": True}
                        },
                        dependencies=[],
                        timeout_minutes=60,
                        retry_count=2,
                        retry_delay_seconds=300
                    ),
                    WorkflowTask(
                        id="compliance_check",
                        name="Compliance Check",
                        task_type=TaskType.COMPLIANCE_CHECK,
                        description="Check compliance against regulatory frameworks",
                        parameters={
                            "framework": "GDPR",
                            "scope": "full"
                        },
                        dependencies=["vulnerability_scan"],
                        timeout_minutes=30,
                        retry_count=1,
                        retry_delay_seconds=60
                    ),
                    WorkflowTask(
                        id="security_report",
                        name="Generate Security Report",
                        task_type=TaskType.REPORT_GENERATION,
                        description="Generate comprehensive security report",
                        parameters={
                            "report_type": "daily_security",
                            "include_recommendations": True
                        },
                        dependencies=["vulnerability_scan", "compliance_check"],
                        timeout_minutes=15,
                        retry_count=1,
                        retry_delay_seconds=30
                    ),
                    WorkflowTask(
                        id="notify_security_team",
                        name="Notify Security Team",
                        task_type=TaskType.NOTIFICATION,
                        description="Send report to security team",
                        parameters={
                            "recipients": ["security-team@company.com"],
                            "subject": "Daily Security Report - {date}",
                            "template": "Daily security scan completed. {vulnerabilities_found} vulnerabilities found. Compliance score: {compliance_score}%.",
                            "channels": ["email", "slack"]
                        },
                        dependencies=["security_report"],
                        timeout_minutes=5,
                        retry_count=3,
                        retry_delay_seconds=60
                    )
                ],
                triggers=[
                    {
                        "type": "scheduled",
                        "schedule": "0 6 * * *",  # Daily at 6 AM
                        "timezone": "UTC"
                    }
                ],
                variables={
                    "date": "${current_date}",
                    "environment": "production"
                },
                notifications={
                    "on_success": ["security-team@company.com"],
                    "on_failure": ["security-team@company.com", "ops-team@company.com"]
                },
                sla_minutes=120,
                tags=["security", "daily", "automated"]
            ),
            
            WorkflowDefinition(
                id="incident_response",
                name="Security Incident Response",
                description="Automated response to security incidents",
                version="1.0.0",
                tasks=[
                    WorkflowTask(
                        id="threat_analysis",
                        name="Threat Analysis",
                        task_type=TaskType.THREAT_ANALYSIS,
                        description="Analyze the security threat",
                        parameters={
                            "incident_data": "${incident_data}",
                            "analysis_type": "comprehensive"
                        },
                        dependencies=[],
                        timeout_minutes=10,
                        retry_count=2,
                        retry_delay_seconds=30
                    ),
                    WorkflowTask(
                        id="immediate_notification",
                        name="Immediate Alert",
                        task_type=TaskType.NOTIFICATION,
                        description="Immediately notify security team",
                        parameters={
                            "recipients": ["security-oncall@company.com"],
                            "subject": "SECURITY INCIDENT ALERT - {incident_type}",
                            "template": "Security incident detected: {incident_description}. Severity: {severity}. Immediate attention required.",
                            "channels": ["email", "sms", "slack"],
                            "priority": "high"
                        },
                        dependencies=[],
                        timeout_minutes=2,
                        retry_count=5,
                        retry_delay_seconds=10
                    ),
                    WorkflowTask(
                        id="automated_containment",
                        name="Automated Containment",
                        task_type=TaskType.REMEDIATION,
                        description="Attempt automated threat containment",
                        parameters={
                            "containment_actions": ["isolate_affected_systems", "block_malicious_ips"],
                            "approval_required": False
                        },
                        dependencies=["threat_analysis"],
                        timeout_minutes=15,
                        retry_count=1,
                        retry_delay_seconds=60,
                        condition="threat_severity >= 'high'"
                    )
                ],
                triggers=[
                    {
                        "type": "event_driven",
                        "event_source": "security_monitoring",
                        "event_type": "incident_detected"
                    },
                    {
                        "type": "webhook",
                        "webhook_path": "/api/webhooks/incident"
                    }
                ],
                variables={
                    "incident_data": "${trigger_data}",
                    "environment": "production"
                },
                notifications={
                    "on_success": ["security-team@company.com"],
                    "on_failure": ["security-team@company.com", "management@company.com"]
                },
                sla_minutes=30,
                tags=["security", "incident", "urgent"]
            )
        ]
        
    async def create_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """Create a new workflow definition"""
        try:
            # Validate workflow
            await self._validate_workflow(workflow_def)
            
            # Store workflow
            self.workflows[workflow_def.id] = workflow_def
            
            # Persist to Redis
            await self.redis_client.set(
                f"workflow_def:{workflow_def.id}",
                json.dumps(asdict(workflow_def), default=str),
                ex=86400 * 30  # 30 days
            )
            
            logger.info(f"Created workflow: {workflow_def.id}")
            return workflow_def.id
            
        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_def.id}: {e}")
            raise
            
    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any] = None, 
                             triggered_by: str = "manual") -> str:
        """Execute a workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
                
            workflow_def = self.workflows[workflow_id]
            
            if not workflow_def.enabled:
                raise ValueError(f"Workflow {workflow_id} is disabled")
                
            execution_id = str(uuid.uuid4())
            
            # Create execution record
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                started_at=datetime.now(),
                completed_at=None,
                triggered_by=triggered_by,
                trigger_data=trigger_data or {},
                task_results={},
                variables=workflow_def.variables.copy()
            )
            
            self.executions[execution_id] = execution
            
            # Start execution asynchronously
            asyncio.create_task(self._execute_workflow_tasks(execution, workflow_def))
            
            logger.info(f"Started workflow execution: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            raise
            
    async def _execute_workflow_tasks(self, execution: WorkflowExecution, 
                                    workflow_def: WorkflowDefinition):
        """Execute workflow tasks"""
        try:
            execution.status = WorkflowStatus.RUNNING
            
            # Build task dependency graph
            task_graph = self._build_task_graph(workflow_def.tasks)
            
            # Execute tasks in topological order
            completed_tasks = set()
            
            while len(completed_tasks) < len(workflow_def.tasks):
                # Find tasks ready to execute
                ready_tasks = []
                for task in workflow_def.tasks:
                    if (task.id not in completed_tasks and 
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                        
                if not ready_tasks:
                    raise Exception("Circular dependency detected in workflow tasks")
                    
                # Execute ready tasks (parallel if allowed)
                parallel_tasks = [t for t in ready_tasks if t.parallel_execution]
                serial_tasks = [t for t in ready_tasks if not t.parallel_execution]
                
                # Execute parallel tasks
                if parallel_tasks:
                    await asyncio.gather(*[
                        self._execute_task(task, execution) 
                        for task in parallel_tasks
                    ])
                    completed_tasks.update(task.id for task in parallel_tasks)
                    
                # Execute serial tasks
                for task in serial_tasks:
                    await self._execute_task(task, execution)
                    completed_tasks.add(task.id)
                    
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            # Send success notifications
            await self._send_workflow_notifications(execution, workflow_def, 'success')
            
            logger.info(f"Workflow execution completed: {execution.id}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
            # Send failure notifications
            await self._send_workflow_notifications(execution, workflow_def, 'failure')
            
            logger.error(f"Workflow execution failed: {execution.id} - {e}")
            
    async def _execute_task(self, task: WorkflowTask, execution: WorkflowExecution):
        """Execute a single task"""
        task_start_time = datetime.now()
        
        try:
            logger.info(f"Executing task: {task.id} in workflow {execution.id}")
            
            # Check condition if specified
            if task.condition and not self._evaluate_condition(task.condition, execution):
                logger.info(f"Task {task.id} skipped due to condition: {task.condition}")
                execution.task_results[task.id] = {
                    'status': 'skipped',
                    'reason': 'condition_not_met',
                    'condition': task.condition
                }
                return
                
            # Get executor for task type
            if task.task_type not in self.executors:
                raise Exception(f"No executor found for task type: {task.task_type}")
                
            executor = self.executors[task.task_type]
            
            # Validate parameters
            if not executor.validate_parameters(task.parameters):
                raise Exception(f"Invalid parameters for task {task.id}")
                
            # Execute with retries
            retry_count = 0
            while retry_count <= task.retry_count:
                try:
                    # Create execution context
                    context = {
                        'execution_id': execution.id,
                        'workflow_id': execution.workflow_id,
                        'task_id': task.id,
                        'variables': execution.variables,
                        'trigger_data': execution.trigger_data,
                        'previous_results': execution.task_results
                    }
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        executor.execute(task, context),
                        timeout=task.timeout_minutes * 60
                    )
                    
                    # Store successful result
                    execution.task_results[task.id] = {
                        'status': 'completed',
                        'result': result,
                        'duration_seconds': (datetime.now() - task_start_time).total_seconds(),
                        'retry_count': retry_count
                    }
                    
                    # Update execution variables with task results
                    if isinstance(result, dict):
                        for key, value in result.items():
                            execution.variables[f"{task.id}_{key}"] = value
                            
                    logger.info(f"Task {task.id} completed successfully")
                    return
                    
                except asyncio.TimeoutError:
                    error_msg = f"Task {task.id} timed out after {task.timeout_minutes} minutes"
                    logger.warning(f"{error_msg} (attempt {retry_count + 1})")
                    
                    if retry_count >= task.retry_count:
                        raise Exception(error_msg)
                        
                except Exception as e:
                    error_msg = f"Task {task.id} failed: {str(e)}"
                    logger.warning(f"{error_msg} (attempt {retry_count + 1})")
                    
                    if retry_count >= task.retry_count:
                        raise Exception(error_msg)
                        
                retry_count += 1
                if retry_count <= task.retry_count:
                    await asyncio.sleep(task.retry_delay_seconds)
                    
        except Exception as e:
            # Store failed result
            execution.task_results[task.id] = {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': (datetime.now() - task_start_time).total_seconds(),
                'retry_count': retry_count
            }
            
            # Execute failure actions if specified
            if task.on_failure:
                for action in task.on_failure:
                    await self._execute_failure_action(action, execution, task)
                    
            raise
            
    def _build_task_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        for task in tasks:
            graph[task.id] = task.dependencies
        return graph
        
    def _evaluate_condition(self, condition: str, execution: WorkflowExecution) -> bool:
        """Evaluate task condition"""
        try:
            # Simple condition evaluation - in production, use a proper expression evaluator
            # For now, support basic comparisons
            context = {
                'variables': execution.variables,
                'trigger_data': execution.trigger_data,
                'task_results': execution.task_results
            }
            
            # Very basic evaluation - extend as needed
            if 'threat_severity' in condition:
                severity = execution.trigger_data.get('severity', 'low')
                if '>=' in condition:
                    target_severity = condition.split('>=')[1].strip().strip("'\"")
                    severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                    return severity_levels.get(severity, 1) >= severity_levels.get(target_severity, 1)
                    
            return True  # Default to true for unknown conditions
            
        except Exception:
            return True  # Default to true on evaluation error
            
    async def _execute_failure_action(self, action: str, execution: WorkflowExecution, 
                                    failed_task: WorkflowTask):
        """Execute failure action"""
        try:
            # Simple failure actions - extend as needed
            if action == 'notify_admin':
                # Send admin notification
                pass
            elif action == 'rollback':
                # Execute rollback procedures
                pass
        except Exception as e:
            logger.error(f"Failed to execute failure action {action}: {e}")
            
    async def _send_workflow_notifications(self, execution: WorkflowExecution, 
                                         workflow_def: WorkflowDefinition, 
                                         event_type: str):
        """Send workflow notifications"""
        try:
            notification_config = workflow_def.notifications.get(f'on_{event_type}', [])
            
            if notification_config and self.executors.get(TaskType.NOTIFICATION):
                notification_task = WorkflowTask(
                    id=f"notification_{event_type}",
                    name=f"Workflow {event_type.title()} Notification",
                    task_type=TaskType.NOTIFICATION,
                    description=f"Send {event_type} notification",
                    parameters={
                        'recipients': notification_config,
                        'subject': f"Workflow {workflow_def.name} - {event_type.title()}",
                        'template': f"Workflow {workflow_def.name} has {event_type}ed. Execution ID: {execution.id}",
                        'channels': ['email']
                    },
                    dependencies=[],
                    timeout_minutes=5,
                    retry_count=2,
                    retry_delay_seconds=30
                )
                
                context = {
                    'execution_id': execution.id,
                    'workflow_id': execution.workflow_id,
                    'variables': execution.variables
                }
                
                await self.executors[TaskType.NOTIFICATION].execute(notification_task, context)
                
        except Exception as e:
            logger.error(f"Failed to send workflow notifications: {e}")
            
    async def _scheduler_loop(self):
        """Main scheduler loop for handling scheduled workflows"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check for scheduled workflows
                for workflow_id, workflow_def in self.workflows.items():
                    if not workflow_def.enabled:
                        continue
                        
                    for trigger in workflow_def.triggers:
                        if trigger.get('type') == 'scheduled':
                            if await self._should_trigger_scheduled_workflow(workflow_def, trigger, current_time):
                                await self.execute_workflow(workflow_id, {}, 'scheduler')
                                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                
            await asyncio.sleep(60)  # Check every minute
            
    async def _should_trigger_scheduled_workflow(self, workflow_def: WorkflowDefinition, 
                                               trigger: Dict[str, Any], 
                                               current_time: datetime) -> bool:
        """Check if scheduled workflow should be triggered"""
        try:
            # Simple cron-like scheduling - in production, use proper cron library
            schedule = trigger.get('schedule', '')
            
            # For now, just handle daily schedules
            if schedule.startswith('0 6 * * *'):  # Daily at 6 AM
                last_run_key = f"last_run:{workflow_def.id}"
                last_run = await self.redis_client.get(last_run_key)
                
                if last_run:
                    last_run_date = datetime.fromisoformat(last_run.decode())
                    if (current_time - last_run_date).days < 1:
                        return False
                        
                # Check if it's around 6 AM
                if current_time.hour == 6 and current_time.minute < 5:
                    await self.redis_client.set(last_run_key, current_time.isoformat(), ex=86400)
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking scheduled trigger: {e}")
            return False
            
    async def _validate_workflow(self, workflow_def: WorkflowDefinition):
        """Validate workflow definition"""
        # Check for circular dependencies
        task_ids = {task.id for task in workflow_def.tasks}
        
        for task in workflow_def.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep}")
                    
        # Check for cycles (simplified check)
        visited = set()
        for task in workflow_def.tasks:
            if await self._has_circular_dependency(task, workflow_def.tasks, visited):
                raise ValueError(f"Circular dependency detected involving task {task.id}")
                
    async def _has_circular_dependency(self, task: WorkflowTask, all_tasks: List[WorkflowTask], 
                                     visited: set, path: set = None) -> bool:
        """Check for circular dependencies"""
        if path is None:
            path = set()
            
        if task.id in path:
            return True
            
        if task.id in visited:
            return False
            
        visited.add(task.id)
        path.add(task.id)
        
        task_map = {t.id: t for t in all_tasks}
        
        for dep_id in task.dependencies:
            if dep_id in task_map:
                dep_task = task_map[dep_id]
                if await self._has_circular_dependency(dep_task, all_tasks, visited, path):
                    return True
                    
        path.remove(task.id)
        return False
        
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status"""
        return self.executions.get(execution_id)
        
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.now()
                return True
        return False
        
    async def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        executions = [e for e in self.executions.values() if e.workflow_id == workflow_id]
        
        if not executions:
            return {}
            
        total_executions = len(executions)
        successful_executions = len([e for e in executions if e.status == WorkflowStatus.COMPLETED])
        failed_executions = len([e for e in executions if e.status == WorkflowStatus.FAILED])
        
        # Calculate average duration for completed executions
        completed_executions = [e for e in executions if e.status == WorkflowStatus.COMPLETED]
        avg_duration = 0
        if completed_executions:
            durations = [(e.completed_at - e.started_at).total_seconds() for e in completed_executions]
            avg_duration = sum(durations) / len(durations)
            
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'average_duration_seconds': avg_duration,
            'last_execution': max(executions, key=lambda e: e.started_at).started_at if executions else None
        }
        
    async def shutdown(self):
        """Shutdown the orchestrator"""
        self.running = False
        
        if self.redis_client:
            await self.redis_client.close()
            
        if self.temporal_client:
            await self.temporal_client.close()
            
        logger.info("Workflow Orchestrator shutdown complete")

# Factory function
def create_workflow_orchestrator(config: Dict[str, Any]) -> WorkflowOrchestrator:
    """Create and configure workflow orchestrator"""
    default_config = {
        'redis_url': 'redis://localhost:6379',
        'temporal_host': 'localhost:7233',
        'service_urls': {
            'scanner': 'http://localhost:8001',
            'compliance': 'http://localhost:8002',
            'notifications': 'http://localhost:8003'
        },
        'workflows_dir': './workflows'
    }
    
    final_config = {**default_config, **config}
    return WorkflowOrchestrator(final_config)