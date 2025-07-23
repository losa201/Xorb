
import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

# Import the registry to ensure agents are discovered on startup
from xorb_core.agent_registry import agent_registry
from services.worker.activities import run_agent
from services.worker.workflows import DynamicScanWorkflow


async def main():
    # Ensure agents are loaded
    agent_registry._discover_agents()
    print(f"Loaded {len(agent_registry.get_all_agents())} agents.")

    client = await Client.connect("temporal:7233", namespace="default")

    # Set up a worker
    # The single run_agent activity can execute any agent.
    # The DynamicScanWorkflow orchestrates them.
    worker = Worker(
        client,
        task_queue="xorb-task-queue",
        workflows=[DynamicScanWorkflow],
        activities=[run_agent],
    )

    print("Starting worker...")
    await worker.run()
    print("Worker finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down worker...")
