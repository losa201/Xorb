
from collections import deque
from datetime import timedelta
from typing import List, Set

from temporalio import workflow

# Try multiple import paths for compatibility
try:
    from packages.xorb_core.xorb_core.models import DiscoveryTarget, Finding
except ImportError:
    try:
        from xorb_core.models.agents import DiscoveryTarget, Finding
    except ImportError:
        # Fallback model definitions
        from dataclasses import dataclass
        from typing import Optional, Dict, Any
        from datetime import datetime
        
        @dataclass
        class DiscoveryTarget:
            target_type: str
            value: str
            scope: Optional[str] = None
            metadata: Optional[Dict[str, Any]] = None
        
        @dataclass  
        class Finding:
            id: str
            title: str
            description: str
            severity: str
            target: str
            discovery_method: str
            timestamp: datetime
            finding_type: str = "generic"
            evidence: Dict[str, Any] = None

# Mock agent registry for compatibility
class MockAgentRegistry:
    def get_agents_for_target_type(self, target_type: str):
        return [type('MockAgent', (), {'name': f'mock-agent-{target_type}'})()]

try:
    from xorb_core.agent_registry import agent_registry
except ImportError:
    agent_registry = MockAgentRegistry()

with workflow.unsafe.imports_passed_through():
    from .activities import run_agent


@workflow.defn
class DynamicScanWorkflow:
    """
    A dynamic workflow that orchestrates security scans by intelligently
    selecting and running agents based on discovered findings.
    """

    @workflow.run
    async def run(self, initial_target: DiscoveryTarget) -> List[Finding]:
        all_findings: List[Finding] = []
        
        # Use a deque as a queue for targets to be processed
        target_queue = deque([initial_target])
        
        # Keep track of processed targets to avoid redundant scans
        processed_targets: Set[str] = {f"{initial_target.target_type}::{initial_target.value}"}

        # Get all agents that can handle the initial target type
        initial_agents = agent_registry.get_agents_for_target_type(initial_target.target_type)

        if not initial_agents:
            workflow.logger.warning(f"No agents found for initial target type: {initial_target.target_type}")
            return []

        workflow.logger.info(f"Starting dynamic scan on '{initial_target.value}' with {len(initial_agents)} initial agents.")

        # Start with initial set of agents
        await self._schedule_agent_runs(initial_agents, initial_target, all_findings, target_queue, processed_targets)

        # Process the queue of newly discovered targets
        while target_queue:
            new_target = target_queue.popleft()
            workflow.logger.info(f"Processing new target from queue: {new_target.value} ({new_target.target_type})")

            # Find agents that can process this new target type
            subsequent_agents = agent_registry.get_agents_for_target_type(new_target.target_type)
            await self._schedule_agent_runs(subsequent_agents, new_target, all_findings, target_queue, processed_targets)

        workflow.logger.info(f"Dynamic scan complete. Found a total of {len(all_findings)} findings.")
        return all_findings

    async def _schedule_agent_runs(
        self,
        agents_to_run: List,
        target: DiscoveryTarget,
        all_findings: List[Finding],
        target_queue: deque,
        processed_targets: Set[str],
    ):
        """Helper to schedule and process a batch of agent activities."""
        activity_promises = []
        for agent in agents_to_run:
            promise = workflow.execute_activity(
                run_agent,
                args=[agent.name, target],
                start_to_close_timeout=timedelta(minutes=5),
            )
            activity_promises.append(promise)

        # Wait for all scheduled activities on the current target to complete
        results = await workflow.gather(*activity_promises)

        for findings_list in results:
            if not findings_list:
                continue
            
            all_findings.extend(findings_list)

            # This is the core of the dynamic workflow:
            # We feed the findings of one agent back into the system as new targets.
            for finding in findings_list:
                # Promote a finding to a new target if it's a discoverable asset
                new_target = self._finding_to_target(finding)
                if new_target:
                    target_key = f"{new_target.target_type}::{new_target.value}"
                    if target_key not in processed_targets:
                        workflow.logger.info(f"Promoting finding to new target: {new_target.value}")
                        target_queue.append(new_target)
                        processed_targets.add(target_key)

    def _finding_to_target(self, finding: Finding) -> DiscoveryTarget | None:
        """Converts a finding into a new discovery target if applicable."""
        # This logic determines how the workflow expands. It can be made more sophisticated.
        if finding.finding_type == "subdomain":
            return DiscoveryTarget(value=f"https://{finding.target}", target_type="url")
        # Add more mappings here, e.g., for open ports, discovered APIs, etc.
        # if finding.finding_type == "open_port":
        #     return DiscoveryTarget(value=f"http://{finding.target}:{finding.evidence['port']}", target_type="url")
        return None
