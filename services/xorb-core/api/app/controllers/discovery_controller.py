"""
Discovery controller - Handles discovery workflow HTTP requests
"""

from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from ..container import get_container
from ..services.interfaces import DiscoveryService
from ..domain.exceptions import DomainException
from ..domain.entities import User, Organization
from .base import BaseController


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


class DiscoveryController(BaseController):
    """Discovery controller"""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup discovery routes"""
        
        @self.router.post("/discover", response_model=dict)
        async def start_discovery_endpoint(
            request: StartDiscoveryRequest,
            background_tasks: BackgroundTasks,
            current_user: User = Depends(get_current_user),
            current_org: Organization = Depends(get_current_organization)
        ):
            return await self.start_discovery(request, background_tasks, current_user, current_org)
        
        @self.router.get("/discover/{workflow_id}", response_model=DiscoveryWorkflowResponse)
        async def get_discovery_results_endpoint(
            workflow_id: str,
            current_user: User = Depends(get_current_user)
        ):
            return await self.get_discovery_results(workflow_id, current_user)
        
        @self.router.get("/discover", response_model=List[DiscoveryWorkflowResponse])
        async def list_user_workflows_endpoint(
            limit: int = 50,
            offset: int = 0,
            current_user: User = Depends(get_current_user)
        ):
            return await self.list_user_workflows(current_user, limit, offset)
        
        @self.router.delete("/discover/{workflow_id}")
        async def cancel_workflow_endpoint(
            workflow_id: str,
            current_user: User = Depends(get_current_user)
        ):
            return await self.cancel_workflow(workflow_id, current_user)
    
    async def start_discovery(
        self,
        request: StartDiscoveryRequest,
        background_tasks: BackgroundTasks,
        current_user: User,
        current_org: Organization
    ) -> dict:
        """Start a new discovery workflow"""
        
        try:
            container = get_container()
            discovery_service = container.get(DiscoveryService)
            
            # Start discovery workflow
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
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    async def get_discovery_results(
        self,
        workflow_id: str,
        current_user: User
    ) -> DiscoveryWorkflowResponse:
        """Get results from discovery workflow"""
        
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
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    async def list_user_workflows(
        self,
        current_user: User,
        limit: int = 50,
        offset: int = 0
    ) -> List[DiscoveryWorkflowResponse]:
        """Get workflows for current user"""
        
        try:
            container = get_container()
            discovery_service = container.get(DiscoveryService)
            
            workflows = await discovery_service.get_user_workflows(
                user=current_user,
                limit=limit,
                offset=offset
            )
            
            return [
                DiscoveryWorkflowResponse(
                    id=str(workflow.id),
                    domain=workflow.domain,
                    status=workflow.status,
                    workflow_id=workflow.workflow_id,
                    created_at=workflow.created_at.isoformat(),
                    completed_at=workflow.completed_at.isoformat() if workflow.completed_at else None,
                    findings=workflow.findings or []
                )
                for workflow in workflows
            ]
            
        except DomainException as e:
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    async def cancel_workflow(
        self,
        workflow_id: str,
        current_user: User
    ) -> dict:
        """Cancel a running workflow"""
        
        try:
            container = get_container()
            discovery_service = container.get(DiscoveryService)
            
            success = await discovery_service.cancel_workflow(
                workflow_id=workflow_id,
                user=current_user
            )
            
            return {
                "message": "Workflow cancelled successfully" if success else "Failed to cancel workflow",
                "cancelled": success,
                "workflow_id": workflow_id
            }
            
        except DomainException as e:
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )


def get_current_user() -> User:
    """Dependency to get current user - simplified for this example"""
    from ..domain.entities import User
    # In a real implementation, this would extract and validate the user from the JWT token
    return User.create(username="admin", email="admin@xorb.com", roles=["admin"])


def get_current_organization() -> Organization:
    """Dependency to get current organization - simplified for this example"""
    from ..domain.entities import Organization
    # In a real implementation, this would be determined from the user context or request headers
    return Organization.create(name="Default Organization", plan_type="Enterprise")


# Create controller instance and export router
discovery_controller = DiscoveryController()
router = discovery_controller.router