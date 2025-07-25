
import pytest
from packages.xorb_core.agent_registry import agent_registry
from packages.xorb_core.models.agents import DiscoveryTarget

# Ensure agents are loaded for the tests
agent_registry._discover_agents()

@pytest.mark.asyncio
async def test_subdomain_enumeration_agent_via_registry():
    """Tests the subdomain agent by retrieving it from the registry."""
    agent = agent_registry.get_agent("subdomain_enumeration_crtsh")
    assert agent is not None

    target = DiscoveryTarget(value="example.com", target_type="domain")
    findings = await agent.run(target)

    assert len(findings) > 0
    assert any("www.example.com" in f.target for f in findings)
    for finding in findings:
        assert finding.finding_type == "subdomain"

@pytest.mark.asyncio
async def test_security_headers_agent_via_registry():
    """Tests the security headers agent."""
    agent = agent_registry.get_agent("web_security_headers_check")
    assert agent is not None

    # Using a known site that is likely to have good headers
    target = DiscoveryTarget(value="https://google.com", target_type="url")
    findings = await agent.run(target)

    # This test is less deterministic, but we can check the structure
    # It might find missing headers or none at all.
    assert isinstance(findings, list)

@pytest.mark.asyncio
async def test_agent_handles_wrong_target_type():
    """Ensures an agent returns an empty list for an unsupported target type."""
    agent = agent_registry.get_agent("web_security_headers_check")
    assert agent is not None

    target = DiscoveryTarget(value="example.com", target_type="domain") # Incorrect type
    findings = await agent.run(target)
    assert findings == []
