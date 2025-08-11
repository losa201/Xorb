"""
Advanced Security Orchestration Implementation
Sophisticated security automation, workflow orchestration, and incident response
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from pathlib import Path

from ..domain.entities import User, Organization
from .interfaces import SecurityOrchestrationService
from .base_service import XORBService, ServiceType


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    SCHEDULED = "scheduled"


class ActionType(Enum):
    """Types of security actions"""
    SCAN = "scan"
    BLOCK = "block"
    ISOLATE = "isolate"
    NOTIFY = "notify"
    ANALYZE = "analyze"
    REMEDIATE = "remediate"
    INVESTIGATE = "investigate"
    REPORT = "report"
    MONITOR = "monitor"
    BACKUP = "backup"


class TriggerType(Enum):
    """Types of workflow triggers"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    THRESHOLD = "threshold"
    API = "api"
    WEBHOOK = "webhook"


@dataclass
class WorkflowAction:
    """Individual workflow action"""
    id: str
    name: str
    action_type: ActionType
    parameters: Dict[str, Any]
    dependencies: List[str]
    timeout_seconds: int
    retry_count: int
    on_failure: str  # "stop", "continue", "retry"
    condition: Optional[str] = None  # Conditional execution
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "action_type": self.action_type.value,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "on_failure": self.on_failure,
            "condition": self.condition
        }


@dataclass
class WorkflowTrigger:
    """Workflow trigger configuration"""
    trigger_type: TriggerType
    configuration: Dict[str, Any]
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": self.trigger_type.value,
            "configuration": self.configuration,
            "enabled": self.enabled
        }


@dataclass
class SecurityWorkflow:
    """Security automation workflow"""
    id: str
    name: str
    description: str
    version: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    status: WorkflowStatus
    triggers: List[WorkflowTrigger]
    actions: List[WorkflowAction]
    variables: Dict[str, Any]
    tags: List[str]
    priority: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "triggers": [trigger.to_dict() for trigger in self.triggers],
            "actions": [action.to_dict() for action in self.actions],
            "variables": self.variables,
            "tags": self.tags,
            "priority": self.priority
        }


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    triggered_by: str
    trigger_data: Dict[str, Any]
    execution_context: Dict[str, Any]
    action_results: Dict[str, Any]
    error_message: Optional[str]
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "triggered_by": self.triggered_by,
            "trigger_data": self.trigger_data,
            "execution_context": self.execution_context,
            "action_results": self.action_results,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds
        }


class SecurityActionExecutor:
    """Executes individual security actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Action handlers
        self._action_handlers = {
            ActionType.SCAN: self._execute_scan_action,
            ActionType.BLOCK: self._execute_block_action,
            ActionType.ISOLATE: self._execute_isolate_action,
            ActionType.NOTIFY: self._execute_notify_action,
            ActionType.ANALYZE: self._execute_analyze_action,
            ActionType.REMEDIATE: self._execute_remediate_action,
            ActionType.INVESTIGATE: self._execute_investigate_action,
            ActionType.REPORT: self._execute_report_action,
            ActionType.MONITOR: self._execute_monitor_action,
            ActionType.BACKUP: self._execute_backup_action
        }
    
    async def execute_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a security action"""
        try:
            start_time = time.time()
            
            # Check if action should be executed based on condition
            if action.condition and not self._evaluate_condition(action.condition, context):
                return {
                    "status": "skipped",
                    "reason": "Condition not met",
                    "duration": 0.0
                }
            
            # Get action handler
            handler = self._action_handlers.get(action.action_type)
            if not handler:
                raise ValueError(f"No handler for action type: {action.action_type}")
            
            # Execute action with retry logic
            result = await self._execute_with_retry(handler, action, context)
            
            duration = time.time() - start_time
            result["duration"] = duration
            
            self.logger.info(f"Action {action.name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing action {action.name}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _execute_with_retry(self, handler: Callable, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with retry logic"""
        last_error = None
        
        for attempt in range(action.retry_count + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying action {action.name}, attempt {attempt + 1}")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(action, context),
                    timeout=action.timeout_seconds
                )
                
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Action timed out after {action.timeout_seconds} seconds"
                self.logger.warning(f"Action {action.name} timed out")
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Action {action.name} failed: {str(e)}")
                
                if attempt < action.retry_count:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        raise Exception(f"Action failed after {action.retry_count + 1} attempts: {last_error}")
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate action condition"""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            # Format: "variable operator value" (e.g., "threat_score > 0.7")
            
            if ">" in condition:
                var, value = condition.split(">")
                var_value = context.get(var.strip(), 0)
                return float(var_value) > float(value.strip())
            
            elif "<" in condition:
                var, value = condition.split("<")
                var_value = context.get(var.strip(), 0)
                return float(var_value) < float(value.strip())
            
            elif "==" in condition:
                var, value = condition.split("==")
                var_value = context.get(var.strip(), "")
                return str(var_value).strip() == value.strip().strip('"\'')
            
            elif "!=" in condition:
                var, value = condition.split("!=")
                var_value = context.get(var.strip(), "")
                return str(var_value).strip() != value.strip().strip('"\'')
            
            else:
                # Boolean variable
                return bool(context.get(condition.strip(), False))
                
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {str(e)}")
            return False
    
    async def _execute_scan_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security scan action"""
        parameters = action.parameters
        scan_type = parameters.get("scan_type", "comprehensive")
        targets = parameters.get("targets", [])
        
        # In production, integrate with actual PTaaS service
        self.logger.info(f"Executing {scan_type} scan on {len(targets)} targets")
        
        # Simulate scan execution
        await asyncio.sleep(2)  # Simulate scan time
        
        return {
            "status": "completed",
            "scan_id": str(uuid.uuid4()),
            "targets_scanned": len(targets),
            "vulnerabilities_found": 3,  # Mock result
            "scan_type": scan_type
        }
    
    async def _execute_block_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute block action (IP, domain, etc.)"""
        parameters = action.parameters
        indicators = parameters.get("indicators", [])
        block_type = parameters.get("type", "ip")
        
        self.logger.info(f"Blocking {len(indicators)} {block_type} indicators")
        
        # Simulate blocking
        await asyncio.sleep(0.5)
        
        return {
            "status": "completed",
            "blocked_indicators": indicators,
            "block_type": block_type,
            "rules_created": len(indicators)
        }
    
    async def _execute_isolate_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system isolation action"""
        parameters = action.parameters
        systems = parameters.get("systems", [])
        isolation_type = parameters.get("type", "network")
        
        self.logger.info(f"Isolating {len(systems)} systems ({isolation_type})")
        
        # Simulate isolation
        await asyncio.sleep(1)
        
        return {
            "status": "completed",
            "isolated_systems": systems,
            "isolation_type": isolation_type,
            "isolation_time": datetime.utcnow().isoformat()
        }
    
    async def _execute_notify_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification action"""
        parameters = action.parameters
        recipients = parameters.get("recipients", [])
        message = parameters.get("message", "Security alert")
        channel = parameters.get("channel", "email")
        
        self.logger.info(f"Sending {channel} notifications to {len(recipients)} recipients")
        
        # Simulate notification
        await asyncio.sleep(0.3)
        
        return {
            "status": "completed",
            "notifications_sent": len(recipients),
            "channel": channel,
            "message_id": str(uuid.uuid4())
        }
    
    async def _execute_analyze_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis action"""
        parameters = action.parameters
        data_source = parameters.get("data_source", "logs")
        analysis_type = parameters.get("type", "threat_analysis")
        
        self.logger.info(f"Performing {analysis_type} on {data_source}")
        
        # Simulate analysis
        await asyncio.sleep(3)
        
        return {
            "status": "completed",
            "analysis_type": analysis_type,
            "data_source": data_source,
            "threat_score": 0.7,  # Mock result
            "indicators_found": 5,
            "analysis_id": str(uuid.uuid4())
        }
    
    async def _execute_remediate_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute remediation action"""
        parameters = action.parameters
        remediation_type = parameters.get("type", "patch")
        targets = parameters.get("targets", [])
        
        self.logger.info(f"Executing {remediation_type} remediation on {len(targets)} targets")
        
        # Simulate remediation
        await asyncio.sleep(5)
        
        return {
            "status": "completed",
            "remediation_type": remediation_type,
            "targets_remediated": len(targets),
            "success_rate": 0.95  # Mock result
        }
    
    async def _execute_investigate_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute investigation action"""
        parameters = action.parameters
        investigation_type = parameters.get("type", "forensic")
        scope = parameters.get("scope", [])
        
        self.logger.info(f"Starting {investigation_type} investigation")
        
        # Simulate investigation
        await asyncio.sleep(4)
        
        return {
            "status": "completed",
            "investigation_type": investigation_type,
            "investigation_id": str(uuid.uuid4()),
            "evidence_collected": 15,  # Mock result
            "timeline_events": 23
        }
    
    async def _execute_report_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report generation action"""
        parameters = action.parameters
        report_type = parameters.get("type", "incident")
        format_type = parameters.get("format", "pdf")
        
        self.logger.info(f"Generating {report_type} report in {format_type} format")
        
        # Simulate report generation
        await asyncio.sleep(2)
        
        return {
            "status": "completed",
            "report_type": report_type,
            "format": format_type,
            "report_id": str(uuid.uuid4()),
            "report_url": f"/reports/{uuid.uuid4()}.{format_type}"
        }
    
    async def _execute_monitor_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute monitoring action"""
        parameters = action.parameters
        monitor_type = parameters.get("type", "continuous")
        duration = parameters.get("duration_hours", 24)
        
        self.logger.info(f"Setting up {monitor_type} monitoring for {duration} hours")
        
        return {
            "status": "completed",
            "monitor_type": monitor_type,
            "monitor_id": str(uuid.uuid4()),
            "duration_hours": duration,
            "monitoring_started": datetime.utcnow().isoformat()
        }
    
    async def _execute_backup_action(self, action: WorkflowAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute backup action"""
        parameters = action.parameters
        backup_type = parameters.get("type", "incremental")
        systems = parameters.get("systems", [])
        
        self.logger.info(f"Creating {backup_type} backup of {len(systems)} systems")
        
        # Simulate backup
        await asyncio.sleep(3)
        
        return {
            "status": "completed",
            "backup_type": backup_type,
            "backup_id": str(uuid.uuid4()),
            "systems_backed_up": len(systems),
            "backup_size_gb": 150.5  # Mock result
        }


class WorkflowEngine:
    """Core workflow execution engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.executor = SecurityActionExecutor()
        
        # Active executions
        self._active_executions: Dict[str, WorkflowExecution] = {}
        
        # Workflow scheduler
        self._scheduler_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    async def execute_workflow(self, workflow: SecurityWorkflow, trigger_data: Dict[str, Any], user: User) -> WorkflowExecution:
        """Execute a security workflow"""
        try:
            execution_id = str(uuid.uuid4())
            
            # Create execution instance
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow.id,
                status=WorkflowStatus.PENDING,
                started_at=datetime.utcnow(),
                completed_at=None,
                triggered_by=user.username,
                trigger_data=trigger_data,
                execution_context=workflow.variables.copy(),
                action_results={},
                error_message=None,
                duration_seconds=0.0
            )
            
            # Store execution
            self._active_executions[execution_id] = execution
            
            # Start execution asynchronously
            asyncio.create_task(self._execute_workflow_async(execution, workflow))
            
            self.logger.info(f"Started workflow execution {execution_id} for workflow {workflow.name}")
            return execution
            
        except Exception as e:
            self.logger.error(f"Error starting workflow execution: {str(e)}")
            raise
    
    async def _execute_workflow_async(self, execution: WorkflowExecution, workflow: SecurityWorkflow):
        """Execute workflow asynchronously"""
        try:
            execution.status = WorkflowStatus.RUNNING
            start_time = time.time()
            
            # Build action dependency graph
            action_graph = self._build_action_graph(workflow.actions)
            
            # Execute actions in dependency order
            executed_actions = set()
            
            while len(executed_actions) < len(workflow.actions):
                # Find actions ready to execute
                ready_actions = self._find_ready_actions(action_graph, executed_actions)
                
                if not ready_actions:
                    # Check for circular dependencies or missing dependencies
                    remaining_actions = [a for a in workflow.actions if a.id not in executed_actions]
                    if remaining_actions:
                        raise Exception("Circular dependency or missing dependency detected")
                    break
                
                # Execute ready actions in parallel
                action_tasks = []
                for action in ready_actions:
                    task = asyncio.create_task(
                        self.executor.execute_action(action, execution.execution_context)
                    )
                    action_tasks.append((action, task))
                
                # Wait for all actions to complete
                for action, task in action_tasks:
                    try:
                        result = await task
                        execution.action_results[action.id] = result
                        executed_actions.add(action.id)
                        
                        # Update execution context with action results
                        if result.get("status") == "completed":
                            self._update_execution_context(execution, action, result)
                        
                        # Handle action failure
                        elif result.get("status") == "failed" and action.on_failure == "stop":
                            raise Exception(f"Action {action.name} failed and configured to stop workflow")
                        
                    except Exception as e:
                        self.logger.error(f"Action {action.name} failed: {str(e)}")
                        execution.action_results[action.id] = {
                            "status": "failed",
                            "error": str(e)
                        }
                        
                        if action.on_failure == "stop":
                            raise
                        else:
                            executed_actions.add(action.id)  # Continue execution
            
            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = time.time() - start_time
            
            self.logger.info(f"Workflow execution {execution.id} completed in {execution.duration_seconds:.2f}s")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = time.time() - start_time
            
            self.logger.error(f"Workflow execution {execution.id} failed: {str(e)}")
    
    def _build_action_graph(self, actions: List[WorkflowAction]) -> Dict[str, List[str]]:
        """Build action dependency graph"""
        graph = {}
        
        for action in actions:
            graph[action.id] = action.dependencies
        
        return graph
    
    def _find_ready_actions(self, graph: Dict[str, List[str]], executed_actions: set) -> List[WorkflowAction]:
        """Find actions ready to execute (all dependencies satisfied)"""
        ready_action_ids = []
        
        for action_id, dependencies in graph.items():
            if action_id not in executed_actions:
                # Check if all dependencies are satisfied
                if all(dep in executed_actions for dep in dependencies):
                    ready_action_ids.append(action_id)
        
        # Return actual action objects
        return [action for action in self._workflow_actions if action.id in ready_action_ids]
    
    def _update_execution_context(self, execution: WorkflowExecution, action: WorkflowAction, result: Dict[str, Any]):
        """Update execution context with action results"""
        # Add action results to context for use by subsequent actions
        execution.execution_context[f"{action.id}_result"] = result
        
        # Extract specific values based on action type
        if action.action_type == ActionType.SCAN:
            execution.execution_context["last_scan_id"] = result.get("scan_id")
            execution.execution_context["vulnerabilities_found"] = result.get("vulnerabilities_found", 0)
        
        elif action.action_type == ActionType.ANALYZE:
            execution.execution_context["threat_score"] = result.get("threat_score", 0.0)
            execution.execution_context["indicators_found"] = result.get("indicators_found", 0)
        
        elif action.action_type == ActionType.INVESTIGATE:
            execution.execution_context["investigation_id"] = result.get("investigation_id")
            execution.execution_context["evidence_collected"] = result.get("evidence_collected", 0)
    
    async def start_scheduler(self):
        """Start workflow scheduler for scheduled workflows"""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info("Workflow scheduler started")
    
    async def stop_scheduler(self):
        """Stop workflow scheduler"""
        self._scheduler_running = False
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Workflow scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._scheduler_running:
            try:
                # Check for scheduled workflows (placeholder)
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {str(e)}")
                await asyncio.sleep(60)


class AdvancedSecurityOrchestrationImplementation(SecurityOrchestrationService, XORBService):
    """Advanced security orchestration and automation service"""
    
    def __init__(self):
        super().__init__(service_type=ServiceType.ORCHESTRATION)
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.workflow_engine = WorkflowEngine()
        
        # Workflow storage
        self._workflows: Dict[str, SecurityWorkflow] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        
        # Initialize built-in workflows
        self._initialize_builtin_workflows()
    
    def _initialize_builtin_workflows(self):
        """Initialize built-in security workflows"""
        try:
            # Incident Response Workflow
            incident_response = SecurityWorkflow(
                id="incident_response_workflow",
                name="Automated Incident Response",
                description="Comprehensive incident response workflow with isolation, analysis, and notification",
                version="1.0",
                created_by="system",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=WorkflowStatus.PENDING,
                triggers=[
                    WorkflowTrigger(
                        trigger_type=TriggerType.THRESHOLD,
                        configuration={"metric": "threat_score", "threshold": 0.8}
                    )
                ],
                actions=[
                    WorkflowAction(
                        id="isolate_system",
                        name="Isolate Affected Systems",
                        action_type=ActionType.ISOLATE,
                        parameters={"systems": ["${affected_systems}"], "type": "network"},
                        dependencies=[],
                        timeout_seconds=300,
                        retry_count=2,
                        on_failure="continue"
                    ),
                    WorkflowAction(
                        id="analyze_threat",
                        name="Analyze Threat Indicators",
                        action_type=ActionType.ANALYZE,
                        parameters={"data_source": "logs", "type": "threat_analysis"},
                        dependencies=["isolate_system"],
                        timeout_seconds=600,
                        retry_count=1,
                        on_failure="continue"
                    ),
                    WorkflowAction(
                        id="notify_team",
                        name="Notify Security Team",
                        action_type=ActionType.NOTIFY,
                        parameters={
                            "recipients": ["security-team@company.com"],
                            "message": "Critical security incident detected and containment initiated",
                            "channel": "email"
                        },
                        dependencies=["analyze_threat"],
                        timeout_seconds=60,
                        retry_count=3,
                        on_failure="continue"
                    ),
                    WorkflowAction(
                        id="generate_report",
                        name="Generate Incident Report",
                        action_type=ActionType.REPORT,
                        parameters={"type": "incident", "format": "pdf"},
                        dependencies=["analyze_threat"],
                        timeout_seconds=300,
                        retry_count=1,
                        on_failure="continue"
                    )
                ],
                variables={"severity_threshold": 0.8},
                tags=["incident_response", "automated"],
                priority=1
            )
            
            # Vulnerability Management Workflow
            vulnerability_management = SecurityWorkflow(
                id="vulnerability_management_workflow",
                name="Automated Vulnerability Management",
                description="Scheduled vulnerability scanning and remediation workflow",
                version="1.0",
                created_by="system",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=WorkflowStatus.PENDING,
                triggers=[
                    WorkflowTrigger(
                        trigger_type=TriggerType.SCHEDULED,
                        configuration={"schedule": "0 2 * * 1"}  # Weekly Monday 2 AM
                    )
                ],
                actions=[
                    WorkflowAction(
                        id="comprehensive_scan",
                        name="Comprehensive Vulnerability Scan",
                        action_type=ActionType.SCAN,
                        parameters={
                            "scan_type": "comprehensive",
                            "targets": ["${scan_targets}"]
                        },
                        dependencies=[],
                        timeout_seconds=3600,
                        retry_count=1,
                        on_failure="stop"
                    ),
                    WorkflowAction(
                        id="analyze_vulnerabilities",
                        name="Analyze Scan Results",
                        action_type=ActionType.ANALYZE,
                        parameters={"type": "vulnerability_analysis"},
                        dependencies=["comprehensive_scan"],
                        timeout_seconds=600,
                        retry_count=1,
                        on_failure="continue"
                    ),
                    WorkflowAction(
                        id="remediate_critical",
                        name="Remediate Critical Vulnerabilities",
                        action_type=ActionType.REMEDIATE,
                        parameters={"type": "patch", "priority": "critical"},
                        dependencies=["analyze_vulnerabilities"],
                        timeout_seconds=1800,
                        retry_count=2,
                        on_failure="continue",
                        condition="vulnerabilities_found > 0"
                    ),
                    WorkflowAction(
                        id="backup_before_patch",
                        name="Create System Backup",
                        action_type=ActionType.BACKUP,
                        parameters={"type": "incremental"},
                        dependencies=["analyze_vulnerabilities"],
                        timeout_seconds=900,
                        retry_count=1,
                        on_failure="continue"
                    ),
                    WorkflowAction(
                        id="notify_results",
                        name="Send Vulnerability Report",
                        action_type=ActionType.NOTIFY,
                        parameters={
                            "recipients": ["security-team@company.com"],
                            "message": "Weekly vulnerability scan completed",
                            "channel": "email"
                        },
                        dependencies=["remediate_critical", "backup_before_patch"],
                        timeout_seconds=60,
                        retry_count=2,
                        on_failure="continue"
                    )
                ],
                variables={"scan_targets": ["production_network"]},
                tags=["vulnerability_management", "scheduled"],
                priority=2
            )
            
            # Threat Hunting Workflow
            threat_hunting = SecurityWorkflow(
                id="threat_hunting_workflow",
                name="Proactive Threat Hunting",
                description="Automated threat hunting and investigation workflow",
                version="1.0",
                created_by="system",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=WorkflowStatus.PENDING,
                triggers=[
                    WorkflowTrigger(
                        trigger_type=TriggerType.SCHEDULED,
                        configuration={"schedule": "0 */6 * * *"}  # Every 6 hours
                    )
                ],
                actions=[
                    WorkflowAction(
                        id="analyze_logs",
                        name="Analyze Security Logs",
                        action_type=ActionType.ANALYZE,
                        parameters={"data_source": "security_logs", "type": "anomaly_detection"},
                        dependencies=[],
                        timeout_seconds=1200,
                        retry_count=1,
                        on_failure="continue"
                    ),
                    WorkflowAction(
                        id="investigate_anomalies",
                        name="Investigate Detected Anomalies",
                        action_type=ActionType.INVESTIGATE,
                        parameters={"type": "automated", "scope": ["network", "endpoints"]},
                        dependencies=["analyze_logs"],
                        timeout_seconds=1800,
                        retry_count=1,
                        on_failure="continue",
                        condition="indicators_found > 5"
                    ),
                    WorkflowAction(
                        id="correlate_threats",
                        name="Correlate Threat Intelligence",
                        action_type=ActionType.ANALYZE,
                        parameters={"type": "threat_correlation"},
                        dependencies=["investigate_anomalies"],
                        timeout_seconds=600,
                        retry_count=1,
                        on_failure="continue"
                    ),
                    WorkflowAction(
                        id="generate_hunting_report",
                        name="Generate Threat Hunting Report",
                        action_type=ActionType.REPORT,
                        parameters={"type": "threat_hunting", "format": "json"},
                        dependencies=["correlate_threats"],
                        timeout_seconds=300,
                        retry_count=1,
                        on_failure="continue"
                    )
                ],
                variables={"anomaly_threshold": 0.7},
                tags=["threat_hunting", "proactive"],
                priority=3
            )
            
            # Store built-in workflows
            self._workflows[incident_response.id] = incident_response
            self._workflows[vulnerability_management.id] = vulnerability_management
            self._workflows[threat_hunting.id] = threat_hunting
            
            self.logger.info(f"Initialized {len(self._workflows)} built-in workflows")
            
        except Exception as e:
            self.logger.error(f"Error initializing built-in workflows: {str(e)}")
    
    async def create_workflow(
        self,
        workflow_definition: Dict[str, Any],
        user: User,
        org: Organization
    ) -> Dict[str, Any]:
        """Create security automation workflow"""
        try:
            workflow_id = str(uuid.uuid4())
            
            # Parse workflow definition
            workflow = self._parse_workflow_definition(workflow_definition, workflow_id, user.username)
            
            # Validate workflow
            validation_result = self._validate_workflow(workflow)
            if not validation_result["valid"]:
                return {
                    "error": "Workflow validation failed",
                    "validation_errors": validation_result["errors"]
                }
            
            # Store workflow
            self._workflows[workflow_id] = workflow
            
            self.logger.info(f"Created workflow {workflow.name} ({workflow_id}) by user {user.username}")
            
            return {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "status": "created",
                "validation": validation_result,
                "created_at": workflow.created_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating workflow: {str(e)}")
            return {"error": str(e)}
    
    async def execute_workflow(
        self,
        workflow_id: str,
        parameters: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """Execute a security workflow"""
        try:
            if workflow_id not in self._workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self._workflows[workflow_id]
            
            # Merge parameters with workflow variables
            trigger_data = {**workflow.variables, **parameters}
            
            # Execute workflow
            execution = await self.workflow_engine.execute_workflow(workflow, trigger_data, user)
            
            # Store execution
            self._executions[execution.id] = execution
            
            return {
                "execution_id": execution.id,
                "workflow_id": workflow_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "triggered_by": execution.triggered_by
            }
            
        except Exception as e:
            self.logger.error(f"Error executing workflow: {str(e)}")
            return {"error": str(e)}
    
    async def get_workflow_status(
        self,
        execution_id: str,
        user: User
    ) -> Dict[str, Any]:
        """Get status of workflow execution"""
        try:
            if execution_id not in self._executions:
                # Check in workflow engine's active executions
                if execution_id in self.workflow_engine._active_executions:
                    execution = self.workflow_engine._active_executions[execution_id]
                else:
                    return {"error": f"Execution {execution_id} not found"}
            else:
                execution = self._executions[execution_id]
            
            # Calculate progress
            progress = self._calculate_execution_progress(execution)
            
            return {
                "execution_id": execution_id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "progress": progress,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": execution.duration_seconds,
                "triggered_by": execution.triggered_by,
                "action_results": execution.action_results,
                "error_message": execution.error_message
            }
            
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {str(e)}")
            return {"error": str(e)}
    
    async def schedule_recurring_scan(
        self,
        targets: List[str],
        schedule: str,
        scan_config: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """Schedule recurring security scans"""
        try:
            # Create workflow definition for recurring scan
            workflow_definition = {
                "name": f"Recurring Scan - {', '.join(targets[:3])}{'...' if len(targets) > 3 else ''}",
                "description": f"Scheduled security scan for {len(targets)} targets",
                "triggers": [
                    {
                        "trigger_type": "scheduled",
                        "configuration": {"schedule": schedule}
                    }
                ],
                "actions": [
                    {
                        "id": "scheduled_scan",
                        "name": "Execute Scheduled Scan",
                        "action_type": "scan",
                        "parameters": {
                            "targets": targets,
                            "scan_type": scan_config.get("scan_type", "comprehensive"),
                            **scan_config
                        },
                        "dependencies": [],
                        "timeout_seconds": 3600,
                        "retry_count": 1,
                        "on_failure": "continue"
                    },
                    {
                        "id": "send_results",
                        "name": "Send Scan Results",
                        "action_type": "notify",
                        "parameters": {
                            "recipients": [user.email],
                            "message": f"Scheduled scan completed for {len(targets)} targets",
                            "channel": "email"
                        },
                        "dependencies": ["scheduled_scan"],
                        "timeout_seconds": 60,
                        "retry_count": 2,
                        "on_failure": "continue"
                    }
                ],
                "variables": {"scan_targets": targets},
                "tags": ["scheduled", "recurring", "scan"]
            }
            
            # Create workflow
            result = await self.create_workflow(workflow_definition, user, None)
            
            if "error" not in result:
                result["schedule"] = schedule
                result["targets"] = targets
                result["scan_config"] = scan_config
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error scheduling recurring scan: {str(e)}")
            return {"error": str(e)}
    
    def _parse_workflow_definition(self, definition: Dict[str, Any], workflow_id: str, created_by: str) -> SecurityWorkflow:
        """Parse workflow definition into SecurityWorkflow object"""
        # Parse triggers
        triggers = []
        for trigger_def in definition.get("triggers", []):
            trigger = WorkflowTrigger(
                trigger_type=TriggerType(trigger_def["trigger_type"]),
                configuration=trigger_def["configuration"],
                enabled=trigger_def.get("enabled", True)
            )
            triggers.append(trigger)
        
        # Parse actions
        actions = []
        for action_def in definition.get("actions", []):
            action = WorkflowAction(
                id=action_def["id"],
                name=action_def["name"],
                action_type=ActionType(action_def["action_type"]),
                parameters=action_def["parameters"],
                dependencies=action_def.get("dependencies", []),
                timeout_seconds=action_def.get("timeout_seconds", 300),
                retry_count=action_def.get("retry_count", 1),
                on_failure=action_def.get("on_failure", "stop"),
                condition=action_def.get("condition")
            )
            actions.append(action)
        
        # Create workflow
        workflow = SecurityWorkflow(
            id=workflow_id,
            name=definition["name"],
            description=definition.get("description", ""),
            version=definition.get("version", "1.0"),
            created_by=created_by,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            status=WorkflowStatus.PENDING,
            triggers=triggers,
            actions=actions,
            variables=definition.get("variables", {}),
            tags=definition.get("tags", []),
            priority=definition.get("priority", 5)
        )
        
        # Store actions reference for dependency resolution
        self._workflow_actions = actions
        
        return workflow
    
    def _validate_workflow(self, workflow: SecurityWorkflow) -> Dict[str, Any]:
        """Validate workflow definition"""
        errors = []
        
        # Check for duplicate action IDs
        action_ids = [action.id for action in workflow.actions]
        if len(action_ids) != len(set(action_ids)):
            errors.append("Duplicate action IDs found")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(workflow.actions):
            errors.append("Circular dependencies detected")
        
        # Check for invalid dependencies
        for action in workflow.actions:
            for dep in action.dependencies:
                if dep not in action_ids:
                    errors.append(f"Action {action.id} depends on non-existent action {dep}")
        
        # Check trigger configurations
        for trigger in workflow.triggers:
            if trigger.trigger_type == TriggerType.SCHEDULED:
                if "schedule" not in trigger.configuration:
                    errors.append("Scheduled trigger missing schedule configuration")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _has_circular_dependencies(self, actions: List[WorkflowAction]) -> bool:
        """Check for circular dependencies in workflow actions"""
        # Build dependency graph
        graph = {}
        for action in actions:
            graph[action.id] = action.dependencies
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for action_id in graph:
            if action_id not in visited:
                if has_cycle(action_id):
                    return True
        
        return False
    
    def _calculate_execution_progress(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Calculate workflow execution progress"""
        if execution.status == WorkflowStatus.PENDING:
            return {"percentage": 0, "current_phase": "Initializing"}
        
        elif execution.status == WorkflowStatus.RUNNING:
            # Calculate based on completed actions
            workflow = self._workflows.get(execution.workflow_id)
            if not workflow:
                return {"percentage": 50, "current_phase": "Running"}
            
            total_actions = len(workflow.actions)
            completed_actions = len([r for r in execution.action_results.values() 
                                  if r.get("status") in ["completed", "skipped"]])
            
            percentage = (completed_actions / total_actions * 100) if total_actions > 0 else 0
            
            # Determine current phase
            current_phase = "Executing Actions"
            if execution.action_results:
                last_action_id = list(execution.action_results.keys())[-1]
                last_action = next((a for a in workflow.actions if a.id == last_action_id), None)
                if last_action:
                    current_phase = f"Executing: {last_action.name}"
            
            return {
                "percentage": min(95, percentage),  # Cap at 95% until complete
                "current_phase": current_phase,
                "completed_actions": completed_actions,
                "total_actions": total_actions
            }
        
        elif execution.status == WorkflowStatus.COMPLETED:
            return {"percentage": 100, "current_phase": "Completed"}
        
        elif execution.status == WorkflowStatus.FAILED:
            return {"percentage": 0, "current_phase": "Failed"}
        
        else:
            return {"percentage": 0, "current_phase": execution.status.value.title()}
    
    async def get_available_workflows(self, user: User) -> List[Dict[str, Any]]:
        """Get list of available workflows"""
        workflows = []
        
        for workflow in self._workflows.values():
            workflows.append({
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version,
                "status": workflow.status.value,
                "tags": workflow.tags,
                "priority": workflow.priority,
                "action_count": len(workflow.actions),
                "trigger_count": len(workflow.triggers),
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat()
            })
        
        # Sort by priority and name
        workflows.sort(key=lambda w: (w["priority"], w["name"]))
        
        return workflows
    
    async def start_services(self):
        """Start orchestration services"""
        await self.workflow_engine.start_scheduler()
        self.logger.info("Security orchestration services started")
    
    async def stop_services(self):
        """Stop orchestration services"""
        await self.workflow_engine.stop_scheduler()
        self.logger.info("Security orchestration services stopped")