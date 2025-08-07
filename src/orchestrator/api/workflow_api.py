from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from src.orchestrator.core.base_orchestrator import WorkflowOrchestrator, WorkflowDefinition, WorkflowExecution, WorkflowStatus

# Create Pydantic models for API requests and responses
class WorkflowTaskCreate(BaseModel):
    id: str
    name: str
    task_type: str
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

class WorkflowCreate(BaseModel):
    id: str
    name: str
    description: str
    version: str
    tasks: List[WorkflowTaskCreate]
    triggers: List[Dict[str, Any]]
    variables: Dict[str, Any]
    notifications: Dict[str, List[str]]
    sla_minutes: Optional[int] = None
    tags: List[str] = []
    enabled: bool = True

class WorkflowExecutionResponse(BaseModel):
    id: str
    workflow_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    triggered_by: str
    error_message: Optional[str] = None

class WorkflowTaskResponse(BaseModel):
    id: str
    name: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str] = None

class WorkflowDetailsResponse(BaseModel):
    workflow_id: str
    name: str
    description: str
    status: str
    executions: List[WorkflowExecutionResponse]
    tasks: List[WorkflowTaskResponse]

# Create API router
router = APIRouter(
    prefix="/api/v1/workflows",
    tags=["workflows"],
)

# Dependency to get orchestrator instance
async def get_orchestrator() -> WorkflowOrchestrator:
    # In production, this would come from a dependency injection container
    return WorkflowOrchestrator.get_instance()

@router.post("", status_code=201)
async def create_workflow(
    workflow: WorkflowCreate,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Create a new workflow definition"""
    try:
        workflow_def = WorkflowDefinition(**workflow.dict())
        workflow_id = await orchestrator.create_workflow(workflow_def)
        return {"workflow_id": workflow_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    trigger_data: Dict[str, Any] = None,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Execute a workflow"""
    try:
        execution_id = await orchestrator.execute_workflow(workflow_id, trigger_data)
        return {"execution_id": execution_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{workflow_id}")
async def get_workflow_details(
    workflow_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
) -> WorkflowDetailsResponse:
    """Get detailed information about a workflow"""
    try:
        workflow_def = await orchestrator.get_workflow(workflow_id)
        executions = await orchestrator.get_workflow_executions(workflow_id)
        
        # Convert executions to response format
        execution_responses = [
            WorkflowExecutionResponse(**execution.__dict__)
            for execution in executions
        ]
        
        # Convert tasks to response format
        task_responses = []
        for task in workflow_def.tasks:
            # Get task status from latest execution
            latest_execution = execution_responses[-1] if execution_responses else None
            task_status = "unknown"
            error_message = None
            
            if latest_execution and latest_execution.id in latest_execution.task_results:
                task_result = latest_execution.task_results[latest_execution.id]
                task_status = task_result.get('status', 'unknown')
                error_message = task_result.get('error')
            
            task_responses.append(WorkflowTaskResponse(
                id=task.id,
                name=task.name,
                status=task_status,
                started_at=datetime.now(),  # Would be populated from execution data
                completed_at=datetime.now() if task_status == 'completed' else None,
                error_message=error_message
            ))
        
        return WorkflowDetailsResponse(
            workflow_id=workflow_def.id,
            name=workflow_def.name,
            description=workflow_def.description,
            status="active",  # Would be determined by workflow state
            executions=execution_responses,
            tasks=task_responses
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/executions/{execution_id}")
async def get_execution_status(
    execution_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
) -> WorkflowExecutionResponse:
    """Get workflow execution status"""
    try:
        execution = await orchestrator.get_execution_status(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return WorkflowExecutionResponse(**execution.__dict__)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(
    execution_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Cancel a running workflow execution"""
    try:
        success = await orchestrator.cancel_execution(execution_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to cancel execution")
        return {"status": "cancelled"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("")
async def list_workflows(
    enabled: bool = True,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """List all available workflows"""
    try:
        workflows = await orchestrator.list_workflows(enabled)
        return {
            "workflows": [
                {
                    "id": wf.id,
                    "name": wf.name,
                    "description": wf.description,
                    "status": "active" if wf.enabled else "disabled",
                    "tags": wf.tags
                } for wf in workflows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{workflow_id}/metrics")
async def get_workflow_metrics(
    workflow_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get workflow execution metrics"""
    try:
        metrics = await orchestrator.get_workflow_metrics(workflow_id)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{workflow_id}/enable")
async def enable_workflow(
    workflow_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Enable a workflow"""
    try:
        success = await orchestrator.enable_workflow(workflow_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to enable workflow")
        return {"status": "enabled"}

@router.post("/{workflow_id}/disable")
async def disable_workflow(
    workflow_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Disable a workflow"""
    try:
        success = await orchestrator.disable_workflow(workflow_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to disable workflow")
        return {"status": "disabled"}