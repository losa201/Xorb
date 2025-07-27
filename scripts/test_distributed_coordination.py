from typing import Dict, List, Any, Optional

#!/usr/bin/env python3
"""
Distributed Campaign Coordination Test Script
Demonstrates multi-node orchestration with consensus algorithms
"""

import asyncio
import sys
import os
import aiofiles
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xorb_core.orchestration.distributed_campaign_coordinator import (
    DistributedCampaignCoordinator, Campaign, CampaignTarget, 
    NodeRole, TaskAssignment, TaskStatus
)
from xorb_core.agents.base_agent import BaseAgent, AgentCapability
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockReconAgent(BaseAgent):
    """Mock reconnaissance agent for testing"""
    
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = [AgentCapability.RECONNAISSANCE]
        self.name = "MockReconAgent"
    
    async def execute(self, target: str, **kwargs) -> None:
        await asyncio.sleep(2)  # Simulate work
        return {
            "status": "success",
            "findings": [f"Port 80 open on {target}", f"Service fingerprint for {target}"],
            "confidence": 0.85
        }

class MockExploitAgent(BaseAgent):
    """Mock exploitation agent for testing"""
    
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = [AgentCapability.EXPLOITATION]
        self.name = "MockExploitAgent"
    
    async def execute(self, target: str, **kwargs) -> None:
        await asyncio.sleep(3)  # Simulate work
        return {
            "status": "success",
            "access_gained": True,
            "credentials": ["admin:password123"],
            "confidence": 0.92
        }

async def setup_test_nodes() -> None:
    """Setup multiple coordinator nodes for testing"""
    
    # Node 1 - Primary Coordinator
    node1 = DistributedCampaignCoordinator(
        node_id="coord-001",
        host="localhost",
        port=8080
    )
    
    # Node 2 - Secondary Coordinator  
    node2 = DistributedCampaignCoordinator(
        node_id="coord-002", 
        host="localhost",
        port=8081
    )
    
    # Node 3 - Worker Node
    node3 = DistributedCampaignCoordinator(
        node_id="worker-001",
        host="localhost", 
        port=8082
    )
    node3.node_info.role = NodeRole.WORKER
    
    return [node1, node2, node3]

async def test_node_discovery() -> None:
    """Test automatic node discovery and registration"""
    logger.info("=== Testing Node Discovery ===")
    
    nodes = await setup_test_nodes()
    
    # Start all nodes
    for node in nodes:
        await node.start()
        logger.info(f"Started node {node.node_info.node_id}")
    
    # Allow time for discovery
    await asyncio.sleep(2)
    
    # Check peer connections
    for i, node in enumerate(nodes):
        peers = await node.discover_peers()
        logger.info(f"Node {i+1} discovered {len(peers)} peers")
        assert len(peers) >= 2, f"Node {i+1} should discover other nodes"
    
    # Cleanup
    for node in nodes:
        await node.stop()
    
    logger.info("✅ Node discovery test passed")

async def test_consensus_algorithm() -> None:
    """Test consensus algorithm for leader election"""
    logger.info("=== Testing Consensus Algorithm ===")
    
    nodes = await setup_test_nodes()
    
    # Start nodes
    for node in nodes:
        await node.start()
    
    await asyncio.sleep(1)
    
    # Simulate leader election
    proposal = {
        "type": "leader_election",
        "candidate": "coord-001",
        "term": 1
    }
    
    results = []
    for node in nodes:
        if node.node_info.role == NodeRole.COORDINATOR:
            result = await node.consensus_engine.propose(proposal)
            results.append(result)
    
    # Check consensus reached
    consensus_count = sum(1 for r in results if r.get("accepted", False))
    logger.info(f"Consensus reached by {consensus_count}/{len(results)} nodes")
    
    assert consensus_count >= len(results) // 2 + 1, "Majority consensus required"
    
    # Cleanup
    for node in nodes:
        await node.stop()
    
    logger.info("✅ Consensus algorithm test passed")

async def test_distributed_campaign_execution() -> None:
    """Test end-to-end distributed campaign execution"""
    logger.info("=== Testing Distributed Campaign Execution ===")
    
    nodes = await setup_test_nodes()
    
    # Register mock agents
    recon_agent = MockReconAgent()
    exploit_agent = MockExploitAgent()
    
    for node in nodes:
        node.available_agents = [recon_agent, exploit_agent]
        await node.start()
    
    await asyncio.sleep(1)
    
    # Create test campaign
    campaign = Campaign(
        campaign_id="test-campaign-001",
        name="Multi-Node Security Assessment",
        targets=[
            CampaignTarget(target_id="target1", address="192.168.1.100", target_type="host"),
            CampaignTarget(target_id="target2", address="192.168.1.101", target_type="host"),
            CampaignTarget(target_id="target3", address="192.168.1.102", target_type="host")
        ],
        required_capabilities=[AgentCapability.RECONNAISSANCE, AgentCapability.EXPLOITATION]
    )
    
    # Execute campaign on primary coordinator
    coordinator = nodes[0]
    logger.info("Starting distributed campaign execution...")
    
    results = await coordinator.execute_campaign(campaign)
    
    # Verify results
    assert results is not None, "Campaign should return results"
    logger.info(f"Campaign completed with {len(results.get('task_results', []))} task results")
    
    # Check task distribution across nodes
    task_distribution = {}
    for task_result in results.get('task_results', []):
        node_id = task_result.get('executed_by', 'unknown')
        task_distribution[node_id] = task_distribution.get(node_id, 0) + 1
    
    logger.info(f"Task distribution: {task_distribution}")
    assert len(task_distribution) > 1, "Tasks should be distributed across multiple nodes"
    
    # Cleanup
    for node in nodes:
        await node.stop()
    
    logger.info("✅ Distributed campaign execution test passed")

async def test_fault_tolerance() -> None:
    """Test fault tolerance and node failure handling"""
    logger.info("=== Testing Fault Tolerance ===")
    
    nodes = await setup_test_nodes()
    
    # Start all nodes
    for node in nodes:
        await node.start()
    
    await asyncio.sleep(1)
    
    # Simulate node failure
    logger.info("Simulating node failure...")
    failed_node = nodes[1]
    await failed_node.stop()
    
    await asyncio.sleep(2)
    
    # Create campaign with remaining nodes
    campaign = Campaign(
        campaign_id="fault-test-001",
        name="Fault Tolerance Test",
        targets=[CampaignTarget(target_id="ft1", address="192.168.1.200", target_type="host")],
        required_capabilities=[AgentCapability.RECONNAISSANCE]
    )
    
    # Execute campaign on remaining coordinator
    coordinator = nodes[0]
    recon_agent = MockReconAgent()
    coordinator.available_agents = [recon_agent]
    
    try:
        results = await coordinator.execute_campaign(campaign)
        logger.info("Campaign executed successfully despite node failure")
        assert results is not None, "Campaign should still complete"
    except Exception as e:
        logger.error(f"Campaign failed due to node failure: {e}")
        raise
    
    # Cleanup
    for node in nodes:
        if node != failed_node:  # Already stopped
            await node.stop()
    
    logger.info("✅ Fault tolerance test passed")

async def main() -> None:
    """Run all distributed coordination tests"""
    logger.info("Starting Distributed Campaign Coordination Tests")
    logger.info("=" * 60)
    
    try:
        # Run test suite
        await test_node_discovery()
        await test_consensus_algorithm()
        await test_distributed_campaign_execution()
        await test_fault_tolerance()
        
        logger.info("=" * 60)
        logger.info("🎉 All distributed coordination tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())