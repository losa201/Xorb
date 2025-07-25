import httpx
import socket
from typing import List, Dict, Any, Set

from ..models.agents import Finding

async def enumerate_subdomains(domain: str, agent_id: str) -> List[Finding]:
    """Enumerate subdomains for a target domain using Certificate Transparency."""
    subdomains = await _cert_transparency_search(domain)
    findings = []
    for subdomain in subdomains:
        finding = Finding(
            finding_type="subdomain",
            title=f"Subdomain discovered: {subdomain}",
            description=f"Active subdomain found for {domain}",
            target=subdomain,
            agent_id=agent_id,
            evidence={"source": "certificate_transparency"},
            confidence=0.9,
        )
        findings.append(finding)
    return findings

async def resolve_dns(hostname: str, agent_id: str) -> List[Finding]:
    """Resolve DNS records for a hostname."""
    findings = []
    try:
        ip_addresses = await _dns_resolve(hostname)
        for ip in ip_addresses:
            finding = Finding(
                finding_type="dns_resolution",
                title=f"DNS resolution for {hostname}: {ip}",
                description=f"Resolved {hostname} to {ip}",
                target=hostname,
                agent_id=agent_id,
                evidence={"ip_address": ip},
                confidence=1.0,
            )
            findings.append(finding)
    except socket.gaierror:
        pass  # Host not found
    return findings

async def _cert_transparency_search(domain: str) -> Set[str]:
    """Search Certificate Transparency logs."""
    subdomains = set()
    async with httpx.AsyncClient() as client:
        try:
            url = f"https://crt.sh/?q=%.{domain}&output=json"
            response = await client.get(url)
            if response.status_code == 200:
                certificates = response.json()
                for cert in certificates:
                    name_value = cert.get("name_value", "")
                    for name in name_value.split("\n"):
                        name = name.strip()
                        if name.endswith(f".{domain}") and "*" not in name:
                            subdomains.add(name)
        except Exception as e:
            # In a real implementation, we would log this error
            print(f"Certificate transparency search failed: {e}")
    return subdomains

async def _dns_resolve(hostname: str) -> List[str]:
    """Simple DNS resolution check."""
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        return [info[4][0] for info in addr_info]
    except socket.gaierror:
        return []
