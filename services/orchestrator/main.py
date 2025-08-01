import asyncio
import json
from pathlib import Path

from temporalio.client import Client
from temporalio.exceptions import WorkflowAlreadyStartedError

# Try multiple import paths for compatibility
try:
    from packages.xorb_core.xorb_core.models import DiscoveryTarget
except ImportError:
    try:
        from xorb_core.models.agents import DiscoveryTarget
    except ImportError:
        # Fallback DiscoveryTarget definition
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class DiscoveryTarget:
            target_type: str
            value: str
            scope: str | None = None
            metadata: dict[str, Any] | None = None

try:
    from services.worker.workflows import DynamicScanWorkflow
except ImportError:
    # Create a fallback workflow class
    from temporalio import workflow

    @workflow.defn
    class DynamicScanWorkflow:
        @workflow.run
        async def run(self, target):
            print(f"Processing target: {target}")
            return {"status": "completed", "target": target}

# Path to the target ingestion file
TARGETS_FILE = Path(__file__).parent.parent.parent / "targets.json"

# A simple in-memory set to track processed targets to avoid duplicates
PROCESSED_TARGETS: set[str] = set()

async def main():
    """The main loop for the autonomous orchestrator."""
    print("Starting autonomous orchestrator...")

    try:
        client = await Client.connect("temporal:7233", namespace="default")
    except Exception as e:
        print(f"Error connecting to Temporal: {e}. Is it running?")
        return

    print(f"Watching for new targets in: {TARGETS_FILE}")

    while True:
        try:
            with open(TARGETS_FILE) as f:
                targets_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file doesn't exist or is empty/corrupt, wait and try again
            await asyncio.sleep(10)
            continue

        for target_dict in targets_data:
            try:
                target = DiscoveryTarget(**target_dict)
                target_id = f"{target.target_type}::{target.value}"

                if target_id not in PROCESSED_TARGETS:
                    print(f"New target found: {target.value} ({target.target_type})")
                    try:
                        # Use the target_id as a unique workflow ID to prevent duplicates
                        await client.start_workflow(
                            DynamicScanWorkflow.run,
                            args=[target],
                            id=target_id,
                            task_queue="xorb-task-queue",
                        )
                        print(f"Successfully started workflow for: {target_id}")
                        PROCESSED_TARGETS.add(target_id)
                    except WorkflowAlreadyStartedError:
                        print(f"Workflow for {target_id} already running. Adding to processed list.")
                        PROCESSED_TARGETS.add(target_id)
                    except Exception as e:
                        print(f"Error starting workflow for {target_id}: {e}")

            except Exception as e:
                # Handle malformed target entries
                print(f"Skipping malformed target: {target_dict}. Error: {e}")

        # Poll for new targets every 10 seconds
        await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down orchestrator...")
