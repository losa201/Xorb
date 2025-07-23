
from temporalio import activity

from xorb_core.agent_registry import agent_registry
from xorb_core.models.agents import DiscoveryTarget, Finding


@activity.defn
async def run_agent(agent_name: str, target: DiscoveryTarget) -> list[Finding]:
    """A generic activity to run any registered agent by name."""
    activity.logger.info(f"Executing agent '{agent_name}' on target '{target.value}'")

    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise ValueError(f"Agent '{agent_name}' not found in registry.")

    if target.target_type not in agent.accepted_target_types:
        activity.logger.warning(
            f"Agent '{agent_name}' does not accept target type '{target.target_type}'. Skipping."
        )
        return []

    try:
        findings = await agent.run(target)
        activity.logger.info(f"Agent '{agent_name}' completed, found {len(findings)} findings.")
        return findings
    except Exception as e:
        activity.logger.error(f"Agent '{agent_name}' failed: {e}", exc_info=True)
        # Depending on desired reliability, you might want to raise the exception
        # to let the workflow handle the failure, or return an empty list.
        raise
