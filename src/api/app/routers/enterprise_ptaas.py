"""
Enterprise PTaaS Router - Production-ready penetration testing endpoints
Advanced security testing orchestration with enterprise features
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from uuid import UUID
import asyncio
import logging
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from ..services.ptaas_orchestrator_service import get_orchestration_service, PTaaSOrchestrationService
from ..services.ptaas_scanner_service import get_scanner_service, SecurityScannerService
from ..domain.tenant_entities import ScanTarget, ScanResult
from ..middleware.audit_logging import audit_log
from ..middleware.rate_limiting import rate_limit
from ..auth.dependencies import get_current_user, require_permissions

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(
    prefix="/api/v1/enterprise/ptaas",
    tags=["Enterprise PTaaS"],
    dependencies=[Depends(security)]
)

# Request/Response Models
class EnterpriseWorkflowRequest(BaseModel):
    """Enterprise workflow execution request"""
    workflow_id: str = Field(..., description="Workflow ID to execute")
    targets: List[Dict[str, Any]] = Field(..., description="Scan targets")
    priority: str = Field(default="medium", description="Execution priority (low/medium/high/critical)")
    compliance_framework: Optional[str] = Field(None, description="Compliance framework to validate against")
    stealth_mode: bool = Field(False, description="Enable stealth scanning mode")
    notifications: Optional[List[str]] = Field(None, description="Notification channels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ThreatSimulationRequest(BaseModel):
    """Threat simulation request"""
    simulation_type: str = Field(..., description="Type of threat simulation")
    targets: List[Dict[str, Any]] = Field(..., description="Target systems")
    attack_vectors: List[str] = Field(..., description="Attack vectors to simulate")
    stealth_level: str = Field(default="medium", description="Stealth level (low/medium/high)")
    duration_hours: int = Field(default=24, description="Simulation duration in hours")
    mitre_techniques: Optional[List[str]] = Field(None, description="Specific MITRE ATT&CK techniques")

class ComplianceAssessmentRequest(BaseModel):
    """Compliance assessment request"""
    framework: str = Field(..., description="Compliance framework (PCI-DSS, HIPAA, SOX, etc.)")
    scope: Dict[str, Any] = Field(..., description="Assessment scope")
    targets: List[Dict[str, Any]] = Field(..., description="Systems to assess")
    evidence_collection: bool = Field(True, description="Collect compliance evidence")
    remediation_planning: bool = Field(True, description="Generate remediation plans")

class WorkflowExecutionResponse(BaseModel):
    """Workflow execution response"""
    execution_id: str
    workflow_id: str
    status: str
    started_at: datetime
    estimated_completion: Optional[datetime]
    progress: float
    tasks_total: int
    tasks_completed: int
    current_phase: str

class EnterpriseMetricsResponse(BaseModel):
    """Enterprise metrics response"""
    service_info: Dict[str, Any]
    workflow_metrics: Dict[str, Any]
    resource_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    security_metrics: Dict[str, Any]

# Enterprise PTaaS Endpoints

@router.post("/workflows/execute", response_model=WorkflowExecutionResponse)
@rate_limit(requests_per_minute=10)
@audit_log("ptaas_workflow_execute")
async def execute_enterprise_workflow(
    request: EnterpriseWorkflowRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    orchestrator: PTaaSOrchestrationService = Depends(get_orchestration_service)
):
    """Execute enterprise security workflow with advanced orchestration"""
    
    try:
        # Validate user permissions
        await require_permissions(current_user, ["ptaas:execute", "enterprise:access"])
        
        # Convert request targets to ScanTarget objects
        scan_targets = []
        for target_data in request.targets:
            target = ScanTarget(
                host=target_data["host"],
                ports=target_data.get("ports", [80, 443]),
                scan_profile=target_data.get("scan_profile", "comprehensive")
            )
            scan_targets.append(target)
        
        # Execute workflow with enterprise features
        execution_id = await orchestrator.execute_workflow(
            workflow_id=request.workflow_id,
            targets=scan_targets,
            triggered_by=f"user:{current_user['username']}",
            tenant_id=current_user.get("tenant_id")
        )
        
        # Get execution details
        execution = await orchestrator.get_execution_status(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Get workflow details for task counting
        workflows = await orchestrator.list_workflows()
        workflow = next((w for w in workflows if w.id == request.workflow_id), None)
        tasks_total = len(workflow.tasks) if workflow else 0
        
        # Schedule notifications if configured
        if request.notifications:
            background_tasks.add_task(
                _schedule_notifications,
                execution_id,
                request.notifications,
                current_user
            )
        
        return WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id=request.workflow_id,
            status=execution.status.value,
            started_at=execution.started_at,
            estimated_completion=_estimate_completion(execution, tasks_total),
            progress=execution.progress,
            tasks_total=tasks_total,
            tasks_completed=len(execution.task_results),
            current_phase=_get_current_phase(execution)
        )
        
    except Exception as e:
        logger.error(f"Enterprise workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/threat-simulation")
@rate_limit(requests_per_minute=5)
@audit_log("ptaas_threat_simulation")
async def execute_threat_simulation(
    request: ThreatSimulationRequest,
    current_user: dict = Depends(get_current_user),
    orchestrator: PTaaSOrchestrationService = Depends(get_orchestration_service)
):
    """Execute advanced threat simulation with MITRE ATT&CK framework"""
    
    try:
        # Validate high-privilege operation
        await require_permissions(current_user, ["ptaas:execute", "threat_simulation:access", "enterprise:access"])
        
        # Convert targets
        scan_targets = []
        for target_data in request.targets:
            target = ScanTarget(
                host=target_data["host"],
                ports=target_data.get("ports", []),
                scan_profile="threat_simulation"
            )
            scan_targets.append(target)
        
        # Get threat simulation workflow
        workflows = await orchestrator.list_workflows()
        simulation_workflow = next((w for w in workflows if w.workflow_type.value == "threat_simulation"), None)
        
        if not simulation_workflow:
            raise HTTPException(status_code=404, detail="Threat simulation workflow not found")
        
        # Register threat simulation hooks
        await orchestrator.register_hook("pre_execution", _threat_simulation_pre_hook)
        await orchestrator.register_hook("post_execution", _threat_simulation_post_hook)
        
        # Execute simulation
        execution_id = await orchestrator.execute_workflow(
            workflow_id=simulation_workflow.id,
            targets=scan_targets,
            triggered_by=f"threat_simulation:{current_user['username']}",
            tenant_id=current_user.get("tenant_id")
        )
        
        return {
            "simulation_id": execution_id,
            "simulation_type": request.simulation_type,
            "status": "initiated",
            "estimated_duration_hours": request.duration_hours,
            "attack_vectors": request.attack_vectors,
            "stealth_level": request.stealth_level,
            "monitoring_url": f"/api/v1/enterprise/ptaas/executions/{execution_id}"
        }
        
    except Exception as e:
        logger.error(f"Threat simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compliance-assessment")
@rate_limit(requests_per_minute=3)
@audit_log("ptaas_compliance_assessment")
async def execute_compliance_assessment(
    request: ComplianceAssessmentRequest,
    current_user: dict = Depends(get_current_user),
    orchestrator: PTaaSOrchestrationService = Depends(get_orchestration_service)
):
    """Execute comprehensive compliance assessment"""
    
    try:
        # Validate compliance assessment permissions
        await require_permissions(current_user, ["ptaas:execute", "compliance:assess", "enterprise:access"])
        
        # Get compliance workflow based on framework
        workflows = await orchestrator.list_workflows()
        compliance_workflow = None
        
        for workflow in workflows:
            if (workflow.workflow_type.value == "compliance_scan" and 
                workflow.compliance_framework and 
                workflow.compliance_framework.value.replace("_", "-") == request.framework):
                compliance_workflow = workflow
                break
        
        if not compliance_workflow:
            raise HTTPException(
                status_code=404, 
                detail=f"Compliance workflow for {request.framework} not found"
            )
        
        # Convert targets
        scan_targets = []
        for target_data in request.targets:
            target = ScanTarget(
                host=target_data["host"],
                ports=target_data.get("ports", []),
                scan_profile="compliance"
            )
            scan_targets.append(target)
        
        # Execute compliance assessment
        execution_id = await orchestrator.execute_workflow(
            workflow_id=compliance_workflow.id,
            targets=scan_targets,
            triggered_by=f"compliance:{current_user['username']}",
            tenant_id=current_user.get("tenant_id")
        )
        
        return {
            "assessment_id": execution_id,
            "framework": request.framework,
            "status": "initiated",
            "scope": request.scope,
            "evidence_collection": request.evidence_collection,
            "remediation_planning": request.remediation_planning,
            "monitoring_url": f"/api/v1/enterprise/ptaas/executions/{execution_id}"
        }
        
    except Exception as e:
        logger.error(f"Compliance assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions/{execution_id}")
async def get_execution_status(
    execution_id: str,
    include_details: bool = Query(False, description="Include detailed task results"),
    current_user: dict = Depends(get_current_user),
    orchestrator: PTaaSOrchestrationService = Depends(get_orchestration_service)
):
    """Get detailed execution status with enterprise metrics"""
    
    try:
        await require_permissions(current_user, ["ptaas:read", "enterprise:access"])
        
        execution = await orchestrator.get_execution_status(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Get workflow details
        workflows = await orchestrator.list_workflows()
        workflow = next((w for w in workflows if w.id == execution.workflow_id), None)
        
        response = {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "workflow_name": workflow.name if workflow else "Unknown",
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "triggered_by": execution.triggered_by,
            "progress": execution.progress,
            "compliance_score": execution.compliance_score,
            "findings_summary": execution.findings_summary,
            "error_message": execution.error_message
        }
        
        if include_details:
            response["task_results"] = execution.task_results
            response["scan_results"] = [result.__dict__ if hasattr(result, '__dict__') else result 
                                     for result in execution.scan_results]
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get execution status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=EnterpriseMetricsResponse)
async def get_enterprise_metrics(
    current_user: dict = Depends(get_current_user),
    orchestrator: PTaaSOrchestrationService = Depends(get_orchestration_service),
    scanner: SecurityScannerService = Depends(get_scanner_service)
):
    """Get comprehensive enterprise PTaaS metrics"""
    
    try:
        await require_permissions(current_user, ["ptaas:read", "metrics:view", "enterprise:access"])
        
        # Get orchestrator metrics
        orchestrator_metrics = await orchestrator.get_metrics()
        
        # Get scanner metrics
        scanner_health = await scanner.health_check()
        
        # Calculate security metrics
        executions = await orchestrator.list_executions()
        recent_executions = [e for e in executions if e.started_at > datetime.utcnow() - timedelta(days=7)]
        
        security_metrics = {
            "total_scans_last_7_days": len(recent_executions),
            "vulnerabilities_found_last_7_days": sum(
                len(e.findings_summary.get("vulnerabilities", [])) for e in recent_executions 
                if e.findings_summary
            ),
            "critical_issues_last_7_days": sum(
                e.findings_summary.get("severity_breakdown", {}).get("critical", 0) for e in recent_executions
                if e.findings_summary
            ),
            "compliance_assessments_last_7_days": len([
                e for e in recent_executions if "compliance" in e.triggered_by.lower()
            ]),
            "threat_simulations_last_7_days": len([
                e for e in recent_executions if "threat_simulation" in e.triggered_by.lower()
            ]),
            "average_scan_duration_minutes": sum(
                (e.completed_at - e.started_at).total_seconds() / 60 for e in recent_executions
                if e.completed_at
            ) / len([e for e in recent_executions if e.completed_at]) if recent_executions else 0
        }
        
        # Add scanner availability
        orchestrator_metrics["security_metrics"] = security_metrics
        orchestrator_metrics["scanner_health"] = {
            "status": scanner_health.status.value,
            "available_scanners": scanner_health.checks.get("available_scanners", 0),
            "message": scanner_health.message
        }
        
        return EnterpriseMetricsResponse(**orchestrator_metrics)
        
    except Exception as e:
        logger.error(f"Failed to get enterprise metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/executions/{execution_id}")
@audit_log("ptaas_execution_cancel")
async def cancel_execution(
    execution_id: str,
    current_user: dict = Depends(get_current_user),
    orchestrator: PTaaSOrchestrationService = Depends(get_orchestration_service)
):
    """Cancel running execution"""
    
    try:
        await require_permissions(current_user, ["ptaas:execute", "enterprise:access"])
        
        success = await orchestrator.cancel_execution(execution_id)
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")
        
        return {"message": "Execution cancelled successfully", "execution_id": execution_id}
        
    except Exception as e:
        logger.error(f"Failed to cancel execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def list_enterprise_workflows(
    current_user: dict = Depends(get_current_user),
    orchestrator: PTaaSOrchestrationService = Depends(get_orchestration_service)
):
    """List available enterprise workflows"""
    
    try:
        await require_permissions(current_user, ["ptaas:read", "enterprise:access"])
        
        workflows = await orchestrator.list_workflows()
        
        return {
            "workflows": [
                {
                    "id": w.id,
                    "name": w.name,
                    "type": w.workflow_type.value,
                    "description": w.description,
                    "compliance_framework": w.compliance_framework.value if w.compliance_framework else None,
                    "task_count": len(w.tasks),
                    "enabled": w.enabled
                }
                for w in workflows
            ],
            "total_workflows": len(workflows)
        }
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper Functions

async def _schedule_notifications(execution_id: str, channels: List[str], user: dict):
    """Schedule notifications for workflow completion"""
    try:
        # This would integrate with notification systems (Slack, email, etc.)
        logger.info(f"Scheduling notifications for execution {execution_id} to channels: {channels}")
        # Implementation would depend on notification infrastructure
    except Exception as e:
        logger.error(f"Failed to schedule notifications: {e}")

def _estimate_completion(execution, tasks_total: int) -> Optional[datetime]:
    """Estimate completion time based on progress"""
    if execution.progress == 0 or tasks_total == 0:
        return None
    
    elapsed = (datetime.utcnow() - execution.started_at).total_seconds()
    remaining_time = (elapsed / execution.progress) * (1 - execution.progress)
    
    return datetime.utcnow() + timedelta(seconds=remaining_time)

def _get_current_phase(execution) -> str:
    """Get current execution phase"""
    if execution.status.value == "running":
        completed_tasks = len(execution.task_results)
        if completed_tasks == 0:
            return "initialization"
        elif execution.progress < 0.3:
            return "discovery"
        elif execution.progress < 0.7:
            return "analysis"
        elif execution.progress < 0.9:
            return "validation"
        else:
            return "reporting"
    return execution.status.value

async def _threat_simulation_pre_hook(context: Dict[str, Any]):
    """Pre-execution hook for threat simulations"""
    logger.info(f"Starting threat simulation: {context['execution_id']}")
    # Additional threat simulation setup

async def _threat_simulation_post_hook(context: Dict[str, Any]):
    """Post-execution hook for threat simulations"""
    logger.info(f"Completed threat simulation: {context['execution_id']}")
    # Additional threat simulation cleanup