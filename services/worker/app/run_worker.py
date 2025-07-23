import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from .activities import enumerate_subdomains_activity, resolve_dns_activity
from .workflows import DiscoveryWorkflow

async def main():
    client = await Client.connect("temporal:7233")
    worker = Worker(
        client,
        task_queue="xorb-task-queue",
        workflows=[DiscoveryWorkflow],
        activities=[enumerate_subdomains_activity, resolve_dns_activity],
    )
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
