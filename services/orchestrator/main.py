import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Set

from temporalio.client import Client
from temporalio.exceptions import WorkflowAlreadyStartedError

from xorb_core.models.agents import DiscoveryTarget
from services.worker.workflows import DynamicScanWorkflow

# Path to the target ingestion file
TARGETS_FILE = Path(__file__).parent.parent.parent / "targets.json"

# A simple in-memory set to track processed targets to avoid duplicates
PROCESSED_TARGETS: Set[str] = set()

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
            with open(TARGETS_FILE, 'r') as f:
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