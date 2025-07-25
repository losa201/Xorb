

import httpx
from typing import List, Set
from ..models.agents import DiscoveryTarget, Finding

class SubdomainEnumerationAgent:
    """An agent for discovering subdomains using certificate transparency."""

    @property
    def name(self) -> str:
        return "subdomain_enumeration_crtsh"

    @property
    def description(self) -> str:
        return "Discovers subdomains by querying the crt.sh certificate transparency log."

    @property
    def accepted_target_types(self) -> Set[str]:
        return {"domain"}

    async def run(self, target: DiscoveryTarget) -> List[Finding]:
        if target.target_type not in self.accepted_target_types:
            return []

        findings = []
        async with httpx.AsyncClient() as client:
            try:
                url = f"https://crt.sh/?q=%.{target.value}&output=json"
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                unique_subdomains = set()

                for entry in data:
                    name_value = entry.get("name_value", "")
                    if name_value:
                        subdomains = name_value.split('\n')
                        for subdomain in subdomains:
                            # Clean up and avoid duplicates
                            clean_subdomain = subdomain.strip().lower()
                            if clean_subdomain and '*' not in clean_subdomain:
                                unique_subdomains.add(clean_subdomain)
                
                for subdomain in unique_subdomains:
                    findings.append(
                        Finding(
                            title=f"Discovered Subdomain: {subdomain}",
                            description=f"Subdomain '{subdomain}' discovered for parent '{target.value}' via crt.sh.",
                            target=subdomain,
                            finding_type="subdomain", # Standardized type
                            severity="info",
                            confidence=0.9
                        )
                    )
            except httpx.HTTPStatusError as e:
                print(f"HTTP error occurred while scanning {target.value}: {e}")
            except Exception as e:
                print(f"An error occurred while scanning {target.value}: {e}")
        
        return findings

