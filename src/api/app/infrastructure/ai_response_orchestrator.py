"""AI-Powered Automated Response Orchestrator with Decision Trees"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Set
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from copy import deepcopy

from .database import get_async_session
from .observability import get_metrics_collector, add_trace_context
from ..services.intelligence_service import IntelligenceService
from ..services.interfaces import JobService

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ALERT = "alert"
    INVESTIGATE = "investigate"
    MITIGATE = "mitigate"
    ESCALATE = "escalate"
    REMEDIATE = "remediate"
    MONITOR = "monitor"
    NO_ACTION = "no_action"


class TriggerType(Enum):
    THREAT_DETECTED = "threat_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    INCIDENT_CREATED = "incident_created"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    MANUAL_TRIGGER = "manual_trigger"
    SCHEDULED_CHECK = "scheduled_check"
    EXTERNAL_ALERT = "external_alert"


class DecisionCriteria(Enum):
    SEVERITY_LEVEL = "severity_level"
    CONFIDENCE_SCORE = "confidence_score"
    ASSET_CRITICALITY = "asset_criticality"
    BUSINESS_HOURS = "business_hours"
    USER_BEHAVIOR = "user_behavior"
    THREAT_TYPE = "threat_type"
    GEOGRAPHIC_LOCATION = "geographic_location"
    HISTORICAL_PATTERNS = "historical_patterns"


class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class DecisionNode:
    node_id: str
    name: str
    criteria: DecisionCriteria
    operator: str  # eq, gt, lt, in, contains, etc.
    value: Any
    true_path: Optional[str] = None  # Node ID for true condition
    false_path: Optional[str] = None  # Node ID for false condition
    confidence_threshold: float = 0.0
    description: str = ""


@dataclass
class ActionNode:
    node_id: str
    name: str
    action_type: ResponseType
    parameters: Dict[str, Any]
    timeout_seconds: int = 300
    retry_attempts: int = 3
    prerequisite_actions: List[str] = None
    parallel_execution: bool = False
    description: str = ""


@dataclass
class ResponseWorkflow:
    workflow_id: str
    name: str
    description: str
    trigger_types: List[TriggerType]
    decision_tree: Dict[str, Union[DecisionNode, ActionNode]]
    root_node: str
    tenant_id: UUID
    created_by: str
    created_at: datetime
    updated_at: datetime
    enabled: bool = True
    priority: int = 1  # Higher numbers = higher priority
    tags: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    trigger_event: Dict[str, Any]
    status: WorkflowStatus
    current_node: Optional[str]
    execution_path: List[str]
    actions_executed: List[Dict[str, Any]]
    tenant_id: UUID
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    context_data: Dict[str, Any] = None


@dataclass
class TriggerEvent:
    event_id: str
    event_type: TriggerType
    severity: str
    confidence: float
    source: str
    timestamp: datetime
    tenant_id: UUID
    data: Dict[str, Any]
    context: Dict[str, Any] = None


class AIResponseOrchestrator:
    """Automated response orchestrator with AI-powered decision trees"""

    def __init__(self, intelligence_service: IntelligenceService, job_service: JobService):
        self.intelligence_service = intelligence_service
        self.job_service = job_service
        self.metrics = get_metrics_collector()

        # Workflow registry and execution tracking
        self.workflows: Dict[str, ResponseWorkflow] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}

        # Event queue for processing triggers
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing_workers: List[asyncio.Task] = []

        # Response action handlers
        self.action_handlers: Dict[ResponseType, Callable] = {
            ResponseType.BLOCK: self._handle_block_action,
            ResponseType.QUARANTINE: self._handle_quarantine_action,
            ResponseType.ALERT: self._handle_alert_action,
            ResponseType.INVESTIGATE: self._handle_investigate_action,
            ResponseType.MITIGATE: self._handle_mitigate_action,
            ResponseType.ESCALATE: self._handle_escalate_action,
            ResponseType.REMEDIATE: self._handle_remediate_action,
            ResponseType.MONITOR: self._handle_monitor_action,
            ResponseType.NO_ACTION: self._handle_no_action
        }

        # Built-in workflows
        self._initialize_default_workflows()

    async def initialize(self):
        """Initialize the response orchestrator"""
        logger.info("Initializing AI Response Orchestrator...")

        # Start event processing workers
        for i in range(3):  # 3 concurrent workers
            worker = asyncio.create_task(self._event_processor(f"worker-{i}"))
            self.processing_workers.append(worker)

        # Load workflows from database
        await self._load_workflows_from_database()

        logger.info(f"AI Response Orchestrator initialized with {len(self.workflows)} workflows")

    async def trigger_response(self, event: TriggerEvent) -> List[str]:
        """Trigger automated response for an event"""
        try:
            # Find matching workflows
            matching_workflows = self._find_matching_workflows(event)

            if not matching_workflows:
                logger.debug(f"No matching workflows for event {event.event_id}")
                return []

            # Sort by priority
            matching_workflows.sort(key=lambda w: w.priority, reverse=True)

            execution_ids = []

            # Execute workflows
            for workflow in matching_workflows:
                try:
                    execution_id = await self._execute_workflow(workflow, event)
                    execution_ids.append(execution_id)

                except Exception as e:
                    logger.error(f"Failed to execute workflow {workflow.workflow_id}: {e}")

            return execution_ids

        except Exception as e:
            logger.error(f"Failed to trigger response for event {event.event_id}: {e}")
            return []

    async def create_workflow(self, workflow: ResponseWorkflow) -> str:
        """Create a new response workflow"""
        try:
            # Validate workflow
            await self._validate_workflow(workflow)

            # Store workflow
            self.workflows[workflow.workflow_id] = workflow
            await self._store_workflow_in_database(workflow)

            logger.info(f"Created workflow {workflow.workflow_id}: {workflow.name}")
            return workflow.workflow_id

        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise

    async def update_workflow(self, workflow: ResponseWorkflow) -> bool:
        """Update an existing workflow"""
        try:
            if workflow.workflow_id not in self.workflows:
                return False

            # Validate workflow
            await self._validate_workflow(workflow)

            # Update workflow
            workflow.updated_at = datetime.utcnow()
            self.workflows[workflow.workflow_id] = workflow
            await self._update_workflow_in_database(workflow)

            logger.info(f"Updated workflow {workflow.workflow_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update workflow {workflow.workflow_id}: {e}")
            return False

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow"""
        try:
            if workflow_id not in self.workflows:
                return False

            # Check for active executions
            active_executions = [
                ex for ex in self.active_executions.values()
                if ex.workflow_id == workflow_id
            ]

            if active_executions:
                logger.warning(f"Cannot delete workflow {workflow_id}: {len(active_executions)} active executions")
                return False

            # Delete workflow
            del self.workflows[workflow_id]
            await self._delete_workflow_from_database(workflow_id)

            logger.info(f"Deleted workflow {workflow_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False

    async def get_workflow_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status"""
        return self.active_executions.get(execution_id)

    async def cancel_workflow_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution"""
        try:
            if execution_id not in self.active_executions:
                return False

            execution = self.active_executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.utcnow()

            logger.info(f"Cancelled workflow execution {execution_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False

    async def get_workflow_metrics(self, workflow_id: str, days: int = 7) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        try:
            # This would query database for execution history
            # For now, return mock metrics
            return {
                "total_executions": 45,
                "successful_executions": 42,
                "failed_executions": 3,
                "avg_execution_time_ms": 2150,
                "trigger_breakdown": {
                    "threat_detected": 20,
                    "anomaly_detected": 15,
                    "manual_trigger": 10
                },
                "action_breakdown": {
                    "block": 18,
                    "alert": 45,
                    "investigate": 22,
                    "escalate": 3
                }
            }

        except Exception as e:
            logger.error(f"Failed to get metrics for workflow {workflow_id}: {e}")
            return {}

    # Private methods

    def _find_matching_workflows(self, event: TriggerEvent) -> List[ResponseWorkflow]:
        """Find workflows that match the trigger event"""
        matching_workflows = []

        for workflow in self.workflows.values():
            if not workflow.enabled:
                continue

            # Check if workflow handles this trigger type
            if event.event_type in workflow.trigger_types:
                # Additional filtering could be added here
                matching_workflows.append(workflow)

        return matching_workflows

    async def _execute_workflow(self, workflow: ResponseWorkflow, event: TriggerEvent) -> str:
        """Execute a response workflow"""
        execution_id = str(uuid4())

        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow.workflow_id,
            trigger_event=asdict(event),
            status=WorkflowStatus.IN_PROGRESS,
            current_node=workflow.root_node,
            execution_path=[],
            actions_executed=[],
            tenant_id=workflow.tenant_id,
            started_at=datetime.utcnow(),
            context_data=deepcopy(event.data)
        )

        self.active_executions[execution_id] = execution

        # Add to processing queue
        await self.event_queue.put((workflow, execution, event))

        return execution_id

    async def _event_processor(self, worker_name: str):
        """Background worker for processing workflow executions"""
        logger.info(f"Started event processor: {worker_name}")

        while True:
            try:
                # Get next workflow execution
                workflow, execution, event = await self.event_queue.get()

                logger.info(f"Worker {worker_name} processing execution {execution.execution_id}")

                # Process the workflow
                success = await self._process_workflow_execution(workflow, execution, event)

                # Update execution status
                execution.status = WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED
                execution.completed_at = datetime.utcnow()

                # Store execution results
                await self._store_execution_results(execution)

                # Record metrics
                duration = (execution.completed_at - execution.started_at).total_seconds()
                self.metrics.record_job_execution(
                    f"workflow_execution_{workflow.name.replace(' ', '_').lower()}",
                    duration,
                    success
                )

                logger.info(f"Workflow execution {execution.execution_id} completed: {success}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processor {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

    async def _process_workflow_execution(
        self,
        workflow: ResponseWorkflow,
        execution: WorkflowExecution,
        event: TriggerEvent
    ) -> bool:
        """Process workflow execution through decision tree"""
        try:
            current_node_id = workflow.root_node
            max_iterations = 100  # Prevent infinite loops
            iterations = 0

            while current_node_id and iterations < max_iterations:
                iterations += 1

                if current_node_id not in workflow.decision_tree:
                    execution.error_message = f"Node {current_node_id} not found in decision tree"
                    return False

                node = workflow.decision_tree[current_node_id]
                execution.execution_path.append(current_node_id)
                execution.current_node = current_node_id

                if isinstance(node, DecisionNode):
                    # Process decision node
                    next_node_id = await self._evaluate_decision_node(node, execution, event)
                    current_node_id = next_node_id

                elif isinstance(node, ActionNode):
                    # Execute action node
                    success = await self._execute_action_node(node, execution, event)

                    if not success and not node.parallel_execution:
                        # Stop execution on failure unless parallel
                        execution.error_message = f"Action {node.name} failed"
                        return False

                    # No explicit next node for actions - workflow ends
                    current_node_id = None

                else:
                    execution.error_message = f"Unknown node type: {type(node)}"
                    return False

            if iterations >= max_iterations:
                execution.error_message = "Workflow exceeded maximum iterations"
                return False

            return True

        except Exception as e:
            execution.error_message = str(e)
            logger.error(f"Workflow execution failed: {e}")
            return False

    async def _evaluate_decision_node(
        self,
        node: DecisionNode,
        execution: WorkflowExecution,
        event: TriggerEvent
    ) -> Optional[str]:
        """Evaluate a decision node and return next node ID"""
        try:
            # Get value to evaluate
            if node.criteria == DecisionCriteria.SEVERITY_LEVEL:
                actual_value = event.data.get('severity', 'low')
            elif node.criteria == DecisionCriteria.CONFIDENCE_SCORE:
                actual_value = event.confidence
            elif node.criteria == DecisionCriteria.THREAT_TYPE:
                actual_value = event.data.get('threat_type', 'unknown')
            elif node.criteria == DecisionCriteria.BUSINESS_HOURS:
                current_hour = datetime.now().hour
                actual_value = 9 <= current_hour <= 17  # Simple business hours
            elif node.criteria == DecisionCriteria.ASSET_CRITICALITY:
                actual_value = event.data.get('asset_criticality', 'low')
            else:
                # Use AI intelligence service for complex decisions
                decision_result = await self._get_ai_decision(node, execution, event)
                actual_value = decision_result.get('value', False)

            # Evaluate condition
            result = self._evaluate_condition(actual_value, node.operator, node.value)

            # Add decision context
            execution.context_data[f"decision_{node.node_id}"] = {
                "criteria": node.criteria.value,
                "actual_value": actual_value,
                "expected_value": node.value,
                "operator": node.operator,
                "result": result
            }

            # Return appropriate path
            return node.true_path if result else node.false_path

        except Exception as e:
            logger.error(f"Failed to evaluate decision node {node.node_id}: {e}")
            return node.false_path  # Default to false path on error

    def _evaluate_condition(self, actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate a condition based on operator"""
        try:
            if operator == "eq":
                return actual == expected
            elif operator == "neq":
                return actual != expected
            elif operator == "gt":
                return float(actual) > float(expected)
            elif operator == "gte":
                return float(actual) >= float(expected)
            elif operator == "lt":
                return float(actual) < float(expected)
            elif operator == "lte":
                return float(actual) <= float(expected)
            elif operator == "in":
                return actual in expected
            elif operator == "not_in":
                return actual not in expected
            elif operator == "contains":
                return str(expected).lower() in str(actual).lower()
            elif operator == "startswith":
                return str(actual).startswith(str(expected))
            elif operator == "endswith":
                return str(actual).endswith(str(expected))
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False

    async def _get_ai_decision(
        self,
        node: DecisionNode,
        execution: WorkflowExecution,
        event: TriggerEvent
    ) -> Dict[str, Any]:
        """Use AI intelligence service for complex decision making"""
        try:
            # Create decision request for intelligence service
            from ..routers.intelligence import DecisionRequest, DecisionType, DecisionContext

            decision_request = DecisionRequest(
                decision_type=DecisionType.RESPONSE_STRATEGY,
                context=DecisionContext(
                    scenario=f"Automated response decision for {node.criteria.value}",
                    available_data=execution.context_data,
                    constraints=[],
                    historical_context=[],
                    urgency_level=event.severity,
                    confidence_threshold=node.confidence_threshold
                ),
                model_preferences=[]
            )

            # Get AI decision
            decision_response = await self.intelligence_service.process_decision_request(
                decision_request,
                execution.tenant_id
            )

            # Parse decision for boolean result
            recommendation = decision_response.recommendation.lower()

            if "yes" in recommendation or "true" in recommendation or "proceed" in recommendation:
                return {"value": True, "reasoning": decision_response.reasoning}
            elif "no" in recommendation or "false" in recommendation or "stop" in recommendation:
                return {"value": False, "reasoning": decision_response.reasoning}
            else:
                # Use confidence score
                return {
                    "value": decision_response.confidence_score >= node.confidence_threshold,
                    "reasoning": decision_response.reasoning
                }

        except Exception as e:
            logger.error(f"AI decision failed for node {node.node_id}: {e}")
            return {"value": False, "reasoning": [f"AI decision failed: {str(e)}"]}

    async def _execute_action_node(
        self,
        node: ActionNode,
        execution: WorkflowExecution,
        event: TriggerEvent
    ) -> bool:
        """Execute an action node"""
        try:
            start_time = datetime.utcnow()

            # Get action handler
            handler = self.action_handlers.get(node.action_type)
            if not handler:
                logger.error(f"No handler for action type: {node.action_type}")
                return False

            # Execute action with retries
            for attempt in range(node.retry_attempts + 1):
                try:
                    result = await asyncio.wait_for(
                        handler(node, execution, event),
                        timeout=node.timeout_seconds
                    )

                    # Record action execution
                    action_record = {
                        "node_id": node.node_id,
                        "action_type": node.action_type.value,
                        "attempt": attempt + 1,
                        "result": result,
                        "executed_at": start_time.isoformat(),
                        "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                    }

                    execution.actions_executed.append(action_record)

                    if result.get("success", False):
                        logger.info(f"Action {node.name} executed successfully")
                        return True
                    elif attempt < node.retry_attempts:
                        logger.warning(f"Action {node.name} failed, retrying ({attempt + 1}/{node.retry_attempts})")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

                except asyncio.TimeoutError:
                    logger.error(f"Action {node.name} timed out (attempt {attempt + 1})")
                    if attempt >= node.retry_attempts:
                        return False
                except Exception as e:
                    logger.error(f"Action {node.name} failed: {e} (attempt {attempt + 1})")
                    if attempt >= node.retry_attempts:
                        return False

            return False

        except Exception as e:
            logger.error(f"Failed to execute action node {node.node_id}: {e}")
            return False

    # Action handlers

    async def _handle_block_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle block/isolation action"""
        try:
            target = node.parameters.get("target", "unknown")
            block_type = node.parameters.get("type", "ip")  # ip, domain, user, etc.
            duration = node.parameters.get("duration_minutes", 60)

            # Implement actual blocking logic here
            # For now, simulate blocking
            await asyncio.sleep(0.1)  # Simulate processing time

            logger.info(f"Blocked {block_type} {target} for {duration} minutes")

            return {
                "success": True,
                "action": "block",
                "target": target,
                "type": block_type,
                "duration_minutes": duration,
                "message": f"Successfully blocked {target}"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_quarantine_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle quarantine action"""
        try:
            target = node.parameters.get("target", "unknown")
            quarantine_location = node.parameters.get("location", "/quarantine")

            # Implement actual quarantine logic
            await asyncio.sleep(0.1)

            logger.info(f"Quarantined {target} to {quarantine_location}")

            return {
                "success": True,
                "action": "quarantine",
                "target": target,
                "location": quarantine_location,
                "message": f"Successfully quarantined {target}"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_alert_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle alert/notification action"""
        try:
            recipients = node.parameters.get("recipients", [])
            alert_type = node.parameters.get("type", "email")
            message = node.parameters.get("message", f"Security alert: {event.event_type.value}")

            # Send alerts to recipients
            for recipient in recipients:
                # Implement actual alerting (email, Slack, webhook, etc.)
                logger.info(f"Sent {alert_type} alert to {recipient}: {message}")

            return {
                "success": True,
                "action": "alert",
                "recipients": recipients,
                "alert_type": alert_type,
                "message": message
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_investigate_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle investigation action"""
        try:
            investigation_type = node.parameters.get("type", "automated")
            collect_evidence = node.parameters.get("collect_evidence", True)

            # Create investigation job
            job_payload = {
                "investigation_type": investigation_type,
                "trigger_event": asdict(event),
                "collect_evidence": collect_evidence,
                "execution_id": execution.execution_id
            }

            # Schedule investigation job
            job_result = await self.job_service.schedule_job({
                "job_type": "INVESTIGATION",
                "payload": job_payload,
                "tenant_id": execution.tenant_id,
                "priority": "HIGH"
            })

            return {
                "success": True,
                "action": "investigate",
                "job_id": job_result.get("job_id"),
                "investigation_type": investigation_type
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_mitigate_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle mitigation action"""
        try:
            mitigation_steps = node.parameters.get("steps", [])

            # Execute mitigation steps
            completed_steps = []
            for step in mitigation_steps:
                # Implement actual mitigation logic
                logger.info(f"Executing mitigation step: {step}")
                completed_steps.append(step)

            return {
                "success": True,
                "action": "mitigate",
                "completed_steps": completed_steps,
                "total_steps": len(mitigation_steps)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_escalate_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle escalation action"""
        try:
            escalation_level = node.parameters.get("level", "tier_2")
            notify_users = node.parameters.get("notify_users", [])

            # Create escalation ticket/incident
            escalation_data = {
                "level": escalation_level,
                "trigger_event": asdict(event),
                "workflow_execution": execution.execution_id,
                "notify_users": notify_users
            }

            # Notify escalation recipients
            for user in notify_users:
                logger.info(f"Escalated to {user} (level: {escalation_level})")

            return {
                "success": True,
                "action": "escalate",
                "level": escalation_level,
                "notified_users": notify_users
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_remediate_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle remediation action"""
        try:
            remediation_script = node.parameters.get("script")
            auto_approve = node.parameters.get("auto_approve", False)

            if not auto_approve:
                # Queue for manual approval
                logger.info(f"Remediation queued for approval: {remediation_script}")
                return {
                    "success": True,
                    "action": "remediate",
                    "status": "pending_approval",
                    "script": remediation_script
                }

            # Execute remediation
            logger.info(f"Executing remediation: {remediation_script}")

            return {
                "success": True,
                "action": "remediate",
                "status": "executed",
                "script": remediation_script
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_monitor_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle enhanced monitoring action"""
        try:
            monitor_duration = node.parameters.get("duration_hours", 24)
            monitor_targets = node.parameters.get("targets", [])

            # Enable enhanced monitoring
            for target in monitor_targets:
                logger.info(f"Enabled enhanced monitoring for {target} (duration: {monitor_duration}h)")

            return {
                "success": True,
                "action": "monitor",
                "targets": monitor_targets,
                "duration_hours": monitor_duration
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_no_action(self, node: ActionNode, execution: WorkflowExecution, event: TriggerEvent) -> Dict[str, Any]:
        """Handle no action (logging only)"""
        logger.info(f"No action required for event {event.event_id}")
        return {
            "success": True,
            "action": "no_action",
            "message": "Event logged, no action required"
        }

    def _initialize_default_workflows(self):
        """Initialize built-in default workflows"""
        # High severity threat response workflow
        high_severity_workflow = ResponseWorkflow(
            workflow_id="default_high_severity_response",
            name="High Severity Threat Response",
            description="Automated response for high severity security threats",
            trigger_types=[TriggerType.THREAT_DETECTED, TriggerType.ANOMALY_DETECTED],
            decision_tree={
                "root": DecisionNode(
                    node_id="root",
                    name="Check Severity",
                    criteria=DecisionCriteria.SEVERITY_LEVEL,
                    operator="in",
                    value=["high", "critical"],
                    true_path="check_confidence",
                    false_path="low_severity_action"
                ),
                "check_confidence": DecisionNode(
                    node_id="check_confidence",
                    name="Check Confidence",
                    criteria=DecisionCriteria.CONFIDENCE_SCORE,
                    operator="gte",
                    value=0.8,
                    true_path="immediate_response",
                    false_path="investigate_action"
                ),
                "immediate_response": ActionNode(
                    node_id="immediate_response",
                    name="Immediate Block and Alert",
                    action_type=ResponseType.BLOCK,
                    parameters={
                        "target": "source_ip",
                        "type": "ip",
                        "duration_minutes": 120
                    }
                ),
                "investigate_action": ActionNode(
                    node_id="investigate_action",
                    name="Automated Investigation",
                    action_type=ResponseType.INVESTIGATE,
                    parameters={
                        "type": "automated",
                        "collect_evidence": True
                    }
                ),
                "low_severity_action": ActionNode(
                    node_id="low_severity_action",
                    name="Monitor and Log",
                    action_type=ResponseType.MONITOR,
                    parameters={
                        "duration_hours": 4,
                        "targets": ["source_ip"]
                    }
                )
            },
            root_node="root",
            tenant_id=UUID("00000000-0000-0000-0000-000000000000"),  # System tenant
            created_by="system",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            priority=5,
            tags=["default", "high_severity", "threat_response"]
        )

        self.workflows[high_severity_workflow.workflow_id] = high_severity_workflow

    async def _validate_workflow(self, workflow: ResponseWorkflow):
        """Validate workflow structure"""
        # Check that root node exists
        if workflow.root_node not in workflow.decision_tree:
            raise ValueError(f"Root node {workflow.root_node} not found in decision tree")

        # Validate all node references
        for node_id, node in workflow.decision_tree.items():
            if isinstance(node, DecisionNode):
                if node.true_path and node.true_path not in workflow.decision_tree:
                    raise ValueError(f"True path {node.true_path} not found for node {node_id}")
                if node.false_path and node.false_path not in workflow.decision_tree:
                    raise ValueError(f"False path {node.false_path} not found for node {node_id}")

        # Check for cycles (basic check)
        try:
            # Build graph and check for cycles
            graph = nx.DiGraph()
            for node_id, node in workflow.decision_tree.items():
                graph.add_node(node_id)
                if isinstance(node, DecisionNode):
                    if node.true_path:
                        graph.add_edge(node_id, node.true_path)
                    if node.false_path:
                        graph.add_edge(node_id, node.false_path)

            if not nx.is_directed_acyclic_graph(graph):
                raise ValueError("Workflow contains cycles")

        except ImportError:
            # NetworkX not available, skip cycle check
            pass

    async def _store_workflow_in_database(self, workflow: ResponseWorkflow):
        """Store workflow in database"""
        try:
            # Would implement database storage
            logger.debug(f"Stored workflow {workflow.workflow_id} in database")

        except Exception as e:
            logger.error(f"Failed to store workflow in database: {e}")

    async def _update_workflow_in_database(self, workflow: ResponseWorkflow):
        """Update workflow in database"""
        try:
            # Would implement database update
            logger.debug(f"Updated workflow {workflow.workflow_id} in database")

        except Exception as e:
            logger.error(f"Failed to update workflow in database: {e}")

    async def _delete_workflow_from_database(self, workflow_id: str):
        """Delete workflow from database"""
        try:
            # Would implement database deletion
            logger.debug(f"Deleted workflow {workflow_id} from database")

        except Exception as e:
            logger.error(f"Failed to delete workflow from database: {e}")

    async def _load_workflows_from_database(self):
        """Load workflows from database"""
        try:
            # Would implement database loading
            logger.debug("Loaded workflows from database")

        except Exception as e:
            logger.error(f"Failed to load workflows from database: {e}")

    async def _store_execution_results(self, execution: WorkflowExecution):
        """Store execution results in database"""
        try:
            # Would implement database storage
            logger.debug(f"Stored execution results for {execution.execution_id}")

        except Exception as e:
            logger.error(f"Failed to store execution results: {e}")

    async def shutdown(self):
        """Shutdown the response orchestrator"""
        logger.info("Shutting down AI Response Orchestrator...")

        # Cancel processing workers
        for worker in self.processing_workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.processing_workers, return_exceptions=True)

        logger.info("AI Response Orchestrator shutdown complete")


# Global response orchestrator instance
_response_orchestrator: Optional[AIResponseOrchestrator] = None


async def get_response_orchestrator(
    intelligence_service: IntelligenceService,
    job_service: JobService
) -> AIResponseOrchestrator:
    """Get global response orchestrator instance"""
    global _response_orchestrator

    if _response_orchestrator is None:
        _response_orchestrator = AIResponseOrchestrator(intelligence_service, job_service)
        await _response_orchestrator.initialize()

    return _response_orchestrator
