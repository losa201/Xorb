from fastapi import APIRouter, BackgroundTasks
from temporalio.client import Client
from typing import List

from xorb_core.models.agents import Finding
from services.worker.app.workflows import DiscoveryWorkflow

router = APIRouter()

@router.post("/discover", response_model=dict)
async def start_discovery(domain: str, background_tasks: BackgroundTasks):
    """
    Kick off a new discovery workflow.
    """
    client = await Client.connect("temporal:7233")
    handle = await client.start_workflow(
        DiscoveryWorkflow.run,
        domain,
        id=f"discovery-{domain}",
        task_queue="xorb-task-queue",
    )
    return {"status": "workflow_started", "workflow_id": handle.id}

@router.get("/discover/{workflow_id}", response_model=List[Finding])
async def get_discovery_results(workflow_id: str):
    """
    Retrieve the results of a completed discovery workflow.
    """
    client = await Client.connect("temporal:7233")
    handle = client.get_workflow_handle(workflow_id)
    results = await handle.result()
    return results
