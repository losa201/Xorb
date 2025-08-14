
from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from ..container import get_container
from ..services.interfaces import DiscoveryService
from ..domain.exceptions import DomainException
from ..dependencies import get_current_user, get_current_organization

router = APIRouter()


class StartDiscoveryRequest(BaseModel):
    domain: str = Field(..., description="Domain to discover")


class DiscoveryWorkflowResponse(BaseModel):
    id: str
    domain: str
    status: str
    workflow_id: str
    created_at: str
    completed_at: str = None
    findings: List[dict] = None


@router.post("/discover", response_model=dict)
async def start_discovery(
    request: StartDiscoveryRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    current_org=Depends(get_current_organization)
):
    """Start a new discovery workflow"""
    try:
        container = get_container()
        discovery_service = container.get(DiscoveryService)

        workflow = await discovery_service.start_discovery(
            domain=request.domain,
            user=current_user,
            org=current_org
        )

        return {
            "status": "workflow_started",
            "workflow_id": workflow.workflow_id,
            "id": str(workflow.id),
            "domain": workflow.domain,
            "created_at": workflow.created_at.isoformat()
        }

    except DomainException as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/discover/{workflow_id}", response_model=DiscoveryWorkflowResponse)
async def get_discovery_results(
    workflow_id: str,
    current_user=Depends(get_current_user)
):
    """Retrieve the results of a completed discovery workflow"""
    try:
        container = get_container()
        discovery_service = container.get(DiscoveryService)

        workflow = await discovery_service.get_discovery_results(
            workflow_id=workflow_id,
            user=current_user
        )

        return DiscoveryWorkflowResponse(
            id=str(workflow.id),
            domain=workflow.domain,
            status=workflow.status,
            workflow_id=workflow.workflow_id,
            created_at=workflow.created_at.isoformat(),
            completed_at=workflow.completed_at.isoformat() if workflow.completed_at else None,
            findings=workflow.findings or []
        )

    except DomainException as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail=str(e))
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
