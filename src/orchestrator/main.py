import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from temporalio.client import Client
from temporalio.exceptions import WorkflowAlreadyStartedError, RPCError, ApplicationError
from temporalio.common import RetryPolicy

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
    # Import proper workflow implementation
    from xorb_core.workflows.discovery import DynamicScanWorkflow  # noqa: F401

# Path to the target ingestion file
TARGETS_FILE = Path(__file__).parent.parent.parent / "targets.json"

# A simple in-memory set to track processed targets to avoid duplicates
PROCESSED_TARGETS: set[str] = set()

# Error handling configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 30  # seconds
ERROR_THRESHOLD = 5  # number of errors before circuit breaker trips
ERROR_WINDOW = 60  # seconds

# Circuit breaker state
CIRCUIT_BREAKER_STATE = {
    "tripped": False,
    "trip_time": 0,
    "error_count": 0,
    "error_window_start": 0,
}

# Workflow priority configuration
WORKFLOW_PRIORITY = {
    "high": {"retry_policy": RetryPolicy(maximum_attempts=5, initial_interval=1, backoff_coefficient=2, maximum_interval=30)},
    "medium": {"retry_policy": RetryPolicy(maximum_attempts=3, initial_interval=2, backoff_coefficient=2, maximum_interval=60)},
    "low": {"retry_policy": RetryPolicy(maximum_attempts=2, initial_interval=5, backoff_coefficient=2, maximum_interval=120)},
}

async def reset_circuit_breaker():
    """Reset the circuit breaker after the error window has passed."""
    while True:
        current_time = time.time()
        if CIRCUIT_BREAKER_STATE["tripped"] and (current_time - CIRCUIT_BREAKER_STATE["trip_time"]) > ERROR_WINDOW:
            print("Circuit breaker automatically resetting after error window")
            CIRCUIT_BREAKER_STATE["tripped"] = False
            CIRCUIT_BREAKER_STATE["error_count"] = 0
            CIRCUIT_BREAKER_STATE["error_window_start"] = current_time
        await asyncio.sleep(10)  # Check every 10 seconds

async def handle_workflow_errors(target_id: str, error: Exception, attempt: int, max_attempts: int) -> bool:
    """Handle workflow errors with retry logic and circuit breaker.
    
    Returns:
        bool: True if retry should be attempted, False if circuit breaker is tripped
    """
    current_time = time.time()
    
    # Update error tracking
    if not CIRCUIT_BREAKER_STATE["tripped"]:
        # Reset window if it's expired
        if current_time - CIRCUIT_BREAKER_STATE["error_window_start"] > ERROR_WINDOW:
            CIRCUIT_BREAKER_STATE["error_count"] = 0
            CIRCUIT_BREAKER_STATE["error_window_start"] = current_time
            
        CIRCUIT_BREAKER_STATE["error_count"] += 1
        
    print(f"Error handling workflow {target_id}: {str(error)}. Attempt {attempt}/{max_attempts}")
    
    # Check if circuit breaker should trip
    if CIRCUIT_BREAKER_STATE["error_count"] >= ERROR_THRESHOLD:
        CIRCUIT_BREAKER_STATE["tripped"] = True
        CIRCUIT_BREAKER_STATE["trip_time"] = current_time
        print(f"Circuit breaker tripped. Too many errors in {ERROR_WINDOW} seconds.")
        return False
    
    # If tripped, don't retry
    if CIRCUIT_BREAKER_STATE["tripped"]:
        print("Circuit breaker is tripped. No further attempts will be made.")
        return False
    
    # Calculate exponential backoff delay
    delay = min(INITIAL_RETRY_DELAY * (2 ** (attempt - 1)), MAX_RETRY_DELAY)
    print(f"Retrying in {delay} seconds...")
    await asyncio.sleep(delay)
    return True

async def start_workflow_with_retries(client: Client, target, target_id: str, max_attempts: int = 3):
    """Start a workflow with retry logic and circuit breaker."""
    for attempt in range(1, max_attempts + 1):
        try:
            # Use the target_id as a unique workflow ID to prevent duplicates
            await client.start_workflow(
                DynamicScanWorkflow.run,
                args=[target],
                id=target_id,
                task_queue="xorb-task-queue",
                retry_policy=WORKFLOW_PRIORITY["medium"]["retry_policy"]
            )
            print(f"Successfully started workflow for: {target_id}")
            PROCESSED_TARGETS.add(target_id)
            return True
            
        except WorkflowAlreadyStartedError:
            print(f"Workflow for {target_id} already running. Adding to processed list.")
            PROCESSED_TARGETS.add(target_id)
            return True
            
        except RPCError as e:
            print(f"RPC error starting workflow for {target_id}: {e}")
            if not await handle_workflow_errors(target_id, e, attempt, max_attempts):
                return False
                
        except ApplicationError as e:
            print(f"Application error starting workflow for {target_id}: {e}")
            if not await handle_workflow_errors(target_id, e, attempt, max_attempts):
                return False
                
        except Exception as e:
            print(f"Unexpected error starting workflow for {target_id}: {e}")
            if not await handle_workflow_errors(target_id, e, attempt, max_attempts):
                return False
                
    return False

async def main():
    """The main loop for the autonomous orchestrator."""
    print("Starting autonomous orchestrator...")

    try:
        client = await Client.connect("temporal:7233", namespace="default")
    except Exception as e:
        print(f"Error connecting to Temporal: {e}. Is it running?")
        return

    print(f"Watching for new targets in: {TARGETS_FILE}")
    
    # Start the circuit breaker reset task
    asyncio.create_task(reset_circuit_breaker())

    while True:
        try:
            with open(TARGETS_FILE) as f:
                targets_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # If the file doesn't exist or is empty/corrupt, wait and try again
            print(f"Error reading targets file: {e}")
            await asyncio.sleep(10)
            continue

        for target_dict in targets_data:
            try:
                target = DiscoveryTarget(**target_dict)
                target_id = f"{target.target_type}::{target.value}"

                if target_id not in PROCESSED_TARGETS:
                    print(f"New target found: {target.value} ({target.target_type})")
                    success = await start_workflow_with_retries(client, target, target_id)
                    if not success:
                        print(f"Failed to start workflow for {target_id} after {MAX_RETRIES} attempts")
                        # Add to processed to avoid infinite retries
                        PROCESSED_TARGETS.add(target_id)

            except Exception as e:
                # Handle malformed target entries
                print(f"Skipping malformed target: {target_dict}. Error: {e}")

        # Poll for new targets every 10 seconds
        await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")

# Path to the target ingestion file
TARGETS_FILE = Path(__file__).parent.parent.parent / "targets.json"
# Current workflow version
current_workflow_version = os.getenv("WORKFLOW_VERSION", "v2")

# A simple in-memory set to track processed targets to avoid duplicates
PROCESSED_TARGETS: Dict[str, str] = {}  # {target_id: workflow_version}

# Workflow version mapping
WORKFLOW_VERSIONS = {
    "v1": "DynamicScanWorkflow_v1",
    "v2": "DynamicScanWorkflow_v2",
}

# Migration tracking
MIGRATION_STATS = {
    "v1_to_v2": 0,
    "v2_to_v1": 0,
    "current_version": current_workflow_version,
}

async def get_appropriate_workflow(client: Client, target_id: str, current_version: str):
    """
    Determine which workflow version to use for a target, considering:
    1. If the target is already processed, use the version it was processed with
    2. Otherwise, use the current version
    """
    # Check if this target has been processed before
    if target_id in PROCESSED_TARGETS:
        return PROCESSED_TARGETS[target_id]
    
    # For new targets, use the current version
    return current_version

async def migrate_workflow(client: Client, target_id: str, old_version: str, new_version: str):
    """
    Migrate a workflow from old_version to new_version
    This could involve:
    1. Getting the current state of the workflow
    2. Starting a new workflow with the new version
    3. Preserving state between versions
    4. Completing the old workflow
    """
    try:
        # Get the current workflow execution
        old_workflow = await client.get_workflow_handle(target_id)
        
        # Get the current workflow state
        old_workflow_info = await old_workflow.describe()
        
        # Start new workflow with new version
        new_workflow_id = f"{target_id}_migrated_{new_version}"
        
        # Get the workflow class based on version
        workflow_class = get_workflow_class(new_version)
        
        # Start the new workflow with the same input
        new_workflow = await client.start_workflow(
            workflow_class.run,
            args=[old_workflow_info.last_result],  # Use the last result from old workflow
            id=new_workflow_id,
            task_queue="xorb-task-queue",
        )
        
        # Complete the old workflow with a migration message
        await old_workflow.cancel()
        
        # Update migration stats
        migration_key = f"{old_version}_to_{new_version}"
        if migration_key in MIGRATION_STATS:
            MIGRATION_STATS[migration_key] += 1
        else:
            MIGRATION_STATS[migration_key] = 1
        
        MIGRATION_STATS["current_version"] = new_version
        
        print(f"Migrated workflow {target_id} from {old_version} to {new_version}")
        return new_workflow
    
    except WorkflowNotFoundError:
        # If workflow not found, just proceed with current version
        print(f"Workflow {target_id} not found, starting with current version {new_version}")
        return None
    
    except Exception as e:
        print(f"Error migrating workflow {target_id}: {e}")
        # If migration fails, keep using old version
        return None

def get_workflow_class(version: str):
    """Get the appropriate workflow class based on version"""
    if version == "v1":
        return DynamicScanWorkflowV1
    elif version == "v2":
        return DynamicScanWorkflowV2
    else:
        # Default to v2 if version is unknown
        return DynamicScanWorkflowV2

async def main():
    """The main loop for the autonomous orchestrator."""
    print("Starting autonomous orchestrator...")

    try:
        client = await Client.connect("temporal:7233", namespace="default")
    except Exception as e:
        print(f"Error connecting to Temporal: {e}. Is it running?")
        return

    print(f"Watching for new targets in: {TARGETS_FILE}")
    print(f"Current workflow version: {current_workflow_version}")

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

                # Determine which workflow version to use
                workflow_version = await get_appropriate_workflow(client, target_id, current_workflow_version)
                
                print(f"Using workflow version {workflow_version} for target {target_id}")
                
                # Get the appropriate workflow class
                workflow_class = get_workflow_class(workflow_version)
                
                if target_id not in PROCESSED_TARGETS:
                    print(f"New target found: {target.value} ({target.target_type})")
                    try:
                        # Use the target_id as a unique workflow ID to prevent duplicates
                        await client.start_workflow(
                            workflow_class.run,
                            args=[target],
                            id=target_id,
                            task_queue="xorb-task-queue",
                        )
                        print(f"Successfully started workflow for: {target_id}")
                        PROCESSED_TARGETS[target_id] = workflow_version
                    except WorkflowAlreadyStartedError:
                        print(f"Workflow for {target_id} already running. Adding to processed list.")
                        PROCESSED_TARGETS[target_id] = workflow_version
                    except Exception as e:
                        print(f"Error starting workflow for {target_id}: {e}")

                # Check if we need to migrate to a new version
                elif PROCESSED_TARGETS[target_id] != workflow_version:
                    print(f"Detected version change for {target_id}. Current: {PROCESSED_TARGETS[target_id]}, New: {workflow_version}")
                    migration_result = await migrate_workflow(client, target_id, PROCESSED_TARGETS[target_id], workflow_version)
                    if migration_result:
                        # If migration was successful, update our tracking
                        PROCESSED_TARGETS[target_id] = workflow_version
                        
                # For targets already processed with current version, do nothing
                else:
                    print(f"Target {target_id} already processed with current version {workflow_version}")

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
    finally:
        # Print migration stats on shutdown
        print("\nOrchestrator Shutdown - Migration Statistics:")
        for key, value in MIGRATION_STATS.items():
            if key not in ["current_version"]:
                print(f"{key}: {value}")
        print(f"Current Workflow Version: {MIGRATION_STATS['current_version']}")

# Future Enhancements:
# 1. Add versioned workflow history tracking
# 2. Implement automated version migration policies
# 3. Add workflow version compatibility testing
# 4. Implement canary deployments for new workflow versions
# 5. Add metrics collection for version usage and migration success rates
# 6. Implement version rollback capabilities
# 7. Add workflow state persistence between versions
# 8. Implement versioned API endpoints for workflow management
# 9. Add versioned documentation for each workflow version
# 10. Implement versioned database schema migrations alongside workflow versions
