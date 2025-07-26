

import httpx
from typing import List, Set

# Try multiple import paths for compatibility
try:
    from ..models import DiscoveryTarget, Finding
except ImportError:
    try:
        from ..models.agents import DiscoveryTarget, Finding
    except ImportError:
        # Fallback definitions
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
            title: str
            description: str
            target: str
            finding_type: str
            severity: str
            confidence: float

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


# Additional discovery functions for compatibility with existing imports
async def enumerate_subdomains(domain: str) -> List[str]:
    """
    Enumerate subdomains for the given domain.
    
    Args:
        domain: The domain to enumerate subdomains for
        
    Returns:
        List of discovered subdomains
    """
    agent = SubdomainEnumerationAgent()
    target = DiscoveryTarget(target_type="domain", value=domain)
    findings = await agent.run(target)
    
    # Extract subdomain names from findings
    subdomains = []
    for finding in findings:
        if finding.finding_type == "subdomain":
            subdomains.append(finding.target)
    
    return subdomains


async def resolve_dns(hostname: str) -> dict:
    """
    Resolve DNS records for the given hostname.
    
    Args:
        hostname: The hostname to resolve
        
    Returns:
        Dictionary containing DNS resolution results
    """
    import socket
    import asyncio
    
    try:
        # Perform DNS resolution
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, socket.gethostbyname, hostname)
        
        return {
            "hostname": hostname,
            "ip_address": result,
            "resolved": True,
            "error": None
        }
    except socket.gaierror as e:
        return {
            "hostname": hostname,
            "ip_address": None,
            "resolved": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "hostname": hostname,
            "ip_address": None,
            "resolved": False,
            "error": str(e)
        }

