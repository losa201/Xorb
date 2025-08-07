"""
Discovery service implementation
"""

from typing import List, Optional
from uuid import UUID

# Optional temporal dependency
try:
    from temporalio.client import Client
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    Client = None

from ..domain.entities import User, Organization, DiscoveryWorkflow
from ..domain.exceptions import (
    WorkflowExecutionFailed, ValidationError, ResourceNotFound
)
from ..domain.repositories import DiscoveryRepository
from ..domain.value_objects import Domain
from .interfaces import DiscoveryService


class DiscoveryServiceImpl(DiscoveryService):
    """Implementation of discovery service"""
    
    def __init__(
        self,
        discovery_repository: DiscoveryRepository,
        temporal_url: str = "temporal:7233",
        task_queue: str = "xorb-task-queue"
    ):
        self.discovery_repository = discovery_repository
        self.temporal_url = temporal_url
        self.task_queue = task_queue
    
    async def start_discovery(
        self,
        domain: str,
        user: User,
        org: Organization
    ) -> DiscoveryWorkflow:
        """Start a new discovery workflow"""
        
        # Validate domain
        try:
            domain_obj = Domain(domain)
        except ValueError as e:
            raise ValidationError(str(e))
        
        try:
            # Check if Temporal is available
            if not TEMPORAL_AVAILABLE:
                # Simulate workflow creation for testing
                workflow_id = f"discovery-{domain}-{user.id}"
                workflow = DiscoveryWorkflow.create(
                    domain=domain,
                    user_id=user.id,
                    org_id=org.id,
                    workflow_id=workflow_id
                )
                await self.discovery_repository.save_workflow(workflow)
                return workflow
            
            # Connect to Temporal
            client = await Client.connect(self.temporal_url)
            
            # Start workflow
            # Import actual workflow definition
            from temporalio.workflow import Workflow
            from xorb_core.workflows.discovery import DiscoveryWorkflow
            
            # Generate unique workflow ID
            workflow_id = f"discovery-{domain}-{user.id}"
            
            # Start actual workflow with Temporal
            handle = await client.start_workflow(
                DiscoveryWorkflow.run,
                domain,
                id=workflow_id,
                task_queue=self.task_queue,
            )
            workflow = DiscoveryWorkflow.create(
                domain=domain,
                user_id=user.id,
                org_id=org.id,
                workflow_id=workflow_id
            )
            
            # Save to repository
            await self.discovery_repository.save_workflow(workflow)
            
            return workflow
            
        except Exception as e:
            raise WorkflowExecutionFailed(
                f"Failed to start discovery workflow: {str(e)}",
                workflow_id=workflow_id
            )
    
    async def get_discovery_results(
        self,
        workflow_id: str,
        user: User
    ) -> Optional[DiscoveryWorkflow]:
        """Get results from discovery workflow"""
        
        # Get workflow from repository
        workflow = await self.discovery_repository.get_by_workflow_id(workflow_id)
        
        if not workflow:
            raise ResourceNotFound(f"Workflow not found: {workflow_id}")
        
        # Check if user has access to this workflow
        if workflow.user_id != user.id and not user.has_role("admin"):
            raise ResourceNotFound(f"Workflow not found: {workflow_id}")
        
        # If workflow is still running, check status with Temporal
        if workflow.status in ["started", "running"] and TEMPORAL_AVAILABLE:
            try:
                client = await Client.connect(self.temporal_url)
                handle = client.get_workflow_handle(workflow_id)
                
                # Check if workflow is complete
                try:
                    results = await handle.result()
                    
                    # Update workflow with results
                    workflow.mark_completed(results)
                    await self.discovery_repository.update_workflow(workflow)
                    
                except Exception:
                    # Workflow still running or failed
                    pass
                    
            except Exception as e:
                # Could not connect to Temporal or other error
                # Log the error but return the workflow as-is
                pass
        
        return workflow
    
    async def get_user_workflows(
        self,
        user: User,
        limit: int = 50,
        offset: int = 0
    ) -> List[DiscoveryWorkflow]:
        """Get workflows for user"""
        
        if limit > 100:
            raise ValidationError("Maximum limit is 100")
        
        if offset < 0:
            raise ValidationError("Offset cannot be negative")
        
        return await self.discovery_repository.get_user_workflows(
            user_id=user.id,
            limit=limit,
            offset=offset
        )
    
    async def cancel_workflow(
        self,
        workflow_id: str,
        user: User
    ) -> bool:
        """Cancel a running workflow"""
        
        workflow = await self.discovery_repository.get_by_workflow_id(workflow_id)
        
        if not workflow:
            raise ResourceNotFound(f"Workflow not found: {workflow_id}")
        
        # Check if user has access to this workflow
        if workflow.user_id != user.id and not user.has_role("admin"):
            raise ResourceNotFound(f"Workflow not found: {workflow_id}")
        
        if workflow.status not in ["started", "running"]:
            raise ValidationError("Workflow is not running")
        
        try:
            if not TEMPORAL_AVAILABLE:
                # Simulate cancellation for testing
                workflow.mark_failed()  # Or create a mark_cancelled method
                await self.discovery_repository.update_workflow(workflow)
                return True
            
            # Connect to Temporal and cancel workflow
            client = await Client.connect(self.temporal_url)
            handle = client.get_workflow_handle(workflow_id)
            await handle.cancel()
            
            # Update workflow status
            workflow.mark_failed()  # Or create a mark_cancelled method
            await self.discovery_repository.update_workflow(workflow)
            
            return True
            
        except Exception as e:
            raise WorkflowExecutionFailed(
                f"Failed to cancel workflow: {str(e)}",
                workflow_id=workflow_id
            )