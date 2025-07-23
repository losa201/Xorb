from temporalio import activity
from typing import List
from xorb_core.agents.discovery import enumerate_subdomains, resolve_dns
from xorb_core.models.agents import Finding

@activity.defn
async def enumerate_subdomains_activity(domain: str) -> List[Finding]:
    return await enumerate_subdomains(domain, agent_id="discovery-agent-001")

@activity.defn
async def resolve_dns_activity(hostname: str) -> List[Finding]:
    return await resolve_dns(hostname, agent_id="discovery-agent-001")
