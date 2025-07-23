from datetime import timedelta
from temporalio import workflow
from typing import List

from xorb_core.models.agents import Finding
from .activities import enumerate_subdomains_activity, resolve_dns_activity

@workflow.defn
class DiscoveryWorkflow:
    @workflow.run
    async def run(self, domain: str) -> List[Finding]:
        subdomain_findings = await workflow.execute_activity(
            enumerate_subdomains_activity, domain, start_to_close_timeout=timedelta(minutes=5)
        )

        all_findings = []
        all_findings.extend(subdomain_findings)

        for finding in subdomain_findings:
            dns_findings = await workflow.execute_activity(
                resolve_dns_activity, finding.target, start_to_close_timeout=timedelta(seconds=30)
            )
            all_findings.extend(dns_findings)

        return all_findings
