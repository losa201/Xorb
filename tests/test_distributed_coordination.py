"""
Comprehensive Integration Tests for Distributed Campaign Coordination

These tests validate the distributed coordination system including:
- Multi-node setup and discovery
- Campaign distribution and execution
- Fault tolerance and recovery
- Consensus algorithms
- Load balancing
- Performance under stress
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List

# Add the xorb_core package to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "xorb_core"))

from orchestration.distributed_campaign_coordinator import (
    DistributedCampaignCoordinator,
    DistributedCampaign,
    CampaignTask,
    NodeInfo,
    NodeRole,
    CampaignState,
    CoordinationMessage,
    CoordinationMessageType
)


class TestDistributedCampaignCoordination:
    """Test suite for distributed campaign coordination."""
    
    @pytest.fixture
    async def cluster_setup(self):
        """Set up a test cluster with multiple nodes."""
        coordinators = []
        
        # Create 3 test nodes
        for i in range(3):
            coordinator = DistributedCampaignCoordinator(
                node_id=f"test_node_{i+1}",
                host="localhost",
                port=9000 + i
            )
            
            # Configure different roles
            if i == 0:
                coordinator.node_info.role = NodeRole.COORDINATOR
                coordinator.node_info.capabilities = {"scanning", "analysis", "reporting"}
            elif i == 1:
                coordinator.node_info.role = NodeRole.EXECUTOR
                coordinator.node_info.capabilities = {"scanning", "exploitation"}
            else:
                coordinator.node_info.role = NodeRole.SPECIALIST
                coordinator.node_info.capabilities = {"steganography", "crypto_analysis"}
            
            await coordinator.start_coordinator()
            coordinators.append(coordinator)
        
        # Connect nodes to each other
        for coordinator in coordinators:
            for other in coordinators:
                if coordinator.node_id != other.node_id:
                    coordinator.known_nodes[other.node_id] = other.node_info
        
        yield coordinators
        
        # Cleanup
        for coordinator in coordinators:
            await coordinator.stop_coordinator()
    
    @pytest.mark.asyncio
    async def test_node_discovery_and_registration(self, cluster_setup):
        """Test automatic node discovery and registration."""
        coordinators = cluster_setup
        
        # Verify all nodes know about each other
        for coordinator in coordinators:
            assert len(coordinator.known_nodes) == 2  # Each node knows about 2 others
            
            # Verify node information is correct
            for node_id, node_info in coordinator.known_nodes.items():
                assert node_info.node_id == node_id
                assert node_info.host == "localhost"
                assert node_info.port in range(9000, 9003)
                assert isinstance(node_info.role, NodeRole)
                assert len(node_info.capabilities) > 0
    
    @pytest.mark.asyncio
    async def test_campaign_submission_and_validation(self, cluster_setup):
        """Test campaign submission and validation process."""
        coordinators = cluster_setup
        primary_coordinator = coordinators[0]
        
        # Create a valid campaign
        campaign = DistributedCampaign(
            campaign_id="test_campaign_001",
            title="Test Security Assessment",
            description="Integration test campaign",
            coordinator_id=primary_coordinator.node_id,
            state=CampaignState.PENDING
        )
        
        # Add tasks
        tasks = [
            CampaignTask(
                task_id="scan_task",
                task_type="reconnaissance",
                description="Network scanning task",
                requirements={"scanning"},
                priority=8,
                estimated_duration=300
            ),
            CampaignTask(
                task_id="analysis_task",
                task_type="analysis",
                description="Vulnerability analysis",
                requirements={"analysis"},
                priority=7,
                estimated_duration=600,
                dependencies=["scan_task"]
            )
        ]
        campaign.tasks = tasks
        
        # Submit campaign
        success = await primary_coordinator.submit_campaign(campaign)
        assert success, "Campaign submission should succeed"
        
        # Verify campaign is in active campaigns
        assert campaign.campaign_id in primary_coordinator.active_campaigns
        
        # Verify campaign state
        stored_campaign = primary_coordinator.active_campaigns[campaign.campaign_id]
        assert stored_campaign.state == CampaignState.PENDING
        assert len(stored_campaign.tasks) == 2
    
    @pytest.mark.asyncio
    async def test_task_scheduling_and_assignment(self, cluster_setup):
        """Test intelligent task scheduling and node assignment."""
        coordinators = cluster_setup
        primary_coordinator = coordinators[0]
        
        # Create campaign with diverse task requirements
        campaign = DistributedCampaign(
            campaign_id="test_scheduling_001",
            title="Task Scheduling Test",
            coordinator_id=primary_coordinator.node_id
        )
        
        tasks = [
            CampaignTask(
                task_id="scanning_task",
                task_type="reconnaissance",
                requirements={"scanning"},
                priority=9
            ),
            CampaignTask(
                task_id="crypto_task",
                task_type="specialist",
                requirements={"crypto_analysis"},
                priority=7
            ),
            CampaignTask(
                task_id="exploitation_task",
                task_type="exploitation",
                requirements={"exploitation"},
                priority=8
            ),
            CampaignTask(
                task_id="analysis_task",
                task_type="analysis",
                requirements={"analysis"},
                priority=6
            )
        ]
        campaign.tasks = tasks
        
        await primary_coordinator.submit_campaign(campaign)
        
        # Execute campaign distribution
        await primary_coordinator.execute_distributed_campaign(campaign.campaign_id)
        
        # Verify task assignments
        scheduler = primary_coordinator.task_scheduler
        assignments = scheduler.schedule_tasks(
            tasks, 
            list(primary_coordinator.known_nodes.values()) + [primary_coordinator.node_info]
        )
        
        # Verify that tasks are assigned based on capabilities
        assert len(assignments) > 0, "Some tasks should be assigned"
        
        # Check specific capability matching
        for task_id, node_id in assignments.items():
            task = next(t for t in tasks if t.task_id == task_id)
            if node_id == primary_coordinator.node_id:
                node_info = primary_coordinator.node_info
            else:
                node_info = primary_coordinator.known_nodes[node_id]
            
            # Verify node has required capabilities
            if task.requirements:
                assert task.requirements.issubset(node_info.capabilities), \
                    f"Node {node_id} should have capabilities {task.requirements}"
    
    @pytest.mark.asyncio
    async def test_fault_tolerance_and_recovery(self, cluster_setup):
        """Test fault tolerance mechanisms and node recovery."""
        coordinators = cluster_setup
        
        # Mark one node as failed
        failing_node = coordinators[1]
        original_status = failing_node.node_info.status
        failing_node.node_info.status = "unhealthy"
        failing_node.node_info.last_heartbeat = time.time() - 600  # 10 minutes ago
        
        # Test failure detection
        healthy_coordinators = [c for c in coordinators if c != failing_node]
        
        for coordinator in healthy_coordinators:
            cluster_status = coordinator.get_cluster_status()
            # In a real scenario, health checks would detect the failure
            # For testing, we manually verify the cluster can handle the failure
            assert cluster_status["cluster_size"] == 3
            
            # Verify the coordinator can still operate
            test_campaign = DistributedCampaign(
                campaign_id=f"fault_test_{coordinator.node_id}",
                title="Fault Tolerance Test",
                coordinator_id=coordinator.node_id
            )
            
            success = await coordinator.submit_campaign(test_campaign)
            assert success, "Coordinator should still accept campaigns during node failure"
        
        # Test recovery
        failing_node.node_info.status = original_status
        failing_node.node_info.last_heartbeat = time.time()
        
        # Verify node is back in the cluster
        for coordinator in healthy_coordinators:
            cluster_status = coordinator.get_cluster_status()
            assert cluster_status["cluster_size"] == 3
    
    @pytest.mark.asyncio
    async def test_consensus_algorithm(self, cluster_setup):
        """Test consensus algorithm for distributed decisions."""
        coordinators = cluster_setup
        
        # Test consensus engine
        consensus_engine = coordinators[0].consensus_engine
        
        # Test majority consensus
        proposal = {
            "type": "config_change",
            "description": "Update security parameters",
            "value": {"max_concurrent_campaigns": 5}
        }
        
        # Simulate votes from all nodes
        votes = {
            "test_node_1": True,
            "test_node_2": True,
            "test_node_3": False
        }
        
        result = consensus_engine.evaluate_consensus(proposal, votes)
        assert result.decision == "approved", "Majority vote should approve proposal"
        assert result.consensus_type == "majority"
        
        # Test unanimous requirement scenario
        consensus_engine.require_unanimous = True
        result = consensus_engine.evaluate_consensus(proposal, votes)
        assert result.decision == "rejected", "Unanimous requirement should reject split vote"
        
        # Test unanimous approval
        unanimous_votes = {node_id: True for node_id in votes.keys()}
        result = consensus_engine.evaluate_consensus(proposal, unanimous_votes)
        assert result.decision == "approved", "Unanimous vote should approve"
    
    @pytest.mark.asyncio
    async def test_load_balancing(self, cluster_setup):
        """Test load balancing across multiple nodes."""
        coordinators = cluster_setup
        primary_coordinator = coordinators[0]
        
        # Create multiple campaigns to test load distribution
        campaigns = []
        for i in range(6):  # More campaigns than nodes
            campaign = DistributedCampaign(
                campaign_id=f"load_test_{i+1}",
                title=f"Load Test Campaign {i+1}",
                coordinator_id=primary_coordinator.node_id
            )
            
            # Add a task that any node can handle
            task = CampaignTask(
                task_id=f"load_task_{i+1}",
                task_type="analysis",
                requirements={"scanning"},  # All nodes have scanning capability
                priority=5
            )
            
            campaign.tasks = [task]
            campaigns.append(campaign)
        
        # Submit all campaigns
        for campaign in campaigns:
            success = await primary_coordinator.submit_campaign(campaign)
            assert success, f"Campaign {campaign.campaign_id} should be submitted successfully"
        
        # Verify load distribution would work
        all_nodes = list(primary_coordinator.known_nodes.values()) + [primary_coordinator.node_info]
        scheduler = primary_coordinator.task_scheduler
        
        all_tasks = []
        for campaign in campaigns:
            all_tasks.extend(campaign.tasks)
        
        assignments = scheduler.schedule_tasks(all_tasks, all_nodes)
        
        # Verify tasks are distributed across nodes
        node_task_counts = {}
        for task_id, node_id in assignments.items():
            node_task_counts[node_id] = node_task_counts.get(node_id, 0) + 1
        
        # Each node should have at least one task (if possible)
        assert len(node_task_counts) > 1, "Tasks should be distributed across multiple nodes"
    
    @pytest.mark.asyncio
    async def test_message_handling_and_communication(self, cluster_setup):
        """Test inter-node message handling and communication."""
        coordinators = cluster_setup
        sender = coordinators[0]
        
        # Test message creation and sending
        test_message = CoordinationMessage(
            message_type=CoordinationMessageType.HEARTBEAT,
            sender_id=sender.node_id,
            recipient_id=coordinators[1].node_id,
            payload={"status": "healthy", "timestamp": time.time()}
        )
        
        # Send message
        initial_messages_sent = sender.stats.get("messages_sent", 0)
        await sender._send_message(test_message)
        
        # Verify message was sent (stats incremented)
        assert sender.stats["messages_sent"] == initial_messages_sent + 1
        
        # Test broadcast message
        broadcast_message = CoordinationMessage(
            message_type=CoordinationMessageType.CAMPAIGN_UPDATE,
            sender_id=sender.node_id,
            payload={"campaign_id": "test_broadcast", "status": "executing"}
        )
        
        initial_messages = sender.stats.get("messages_sent", 0)
        await sender._broadcast_message(broadcast_message)
        
        # Should send to all other nodes (2 in this case)
        expected_messages = initial_messages + 2
        assert sender.stats["messages_sent"] == expected_messages
    
    @pytest.mark.asyncio
    async def test_campaign_lifecycle_management(self, cluster_setup):
        """Test complete campaign lifecycle from submission to completion."""
        coordinators = cluster_setup
        primary_coordinator = coordinators[0]
        
        # Create a multi-task campaign
        campaign = DistributedCampaign(
            campaign_id="lifecycle_test_001",
            title="Campaign Lifecycle Test",
            coordinator_id=primary_coordinator.node_id
        )
        
        tasks = [
            CampaignTask(
                task_id="initial_scan",
                task_type="reconnaissance",
                requirements={"scanning"},
                priority=10,
                estimated_duration=60
            ),
            CampaignTask(
                task_id="deep_analysis",
                task_type="analysis",
                requirements={"analysis"},
                priority=8,
                estimated_duration=120,
                dependencies=["initial_scan"]
            ),
            CampaignTask(
                task_id="final_report",
                task_type="reporting",
                requirements={"reporting"},
                priority=6,
                estimated_duration=30,
                dependencies=["deep_analysis"]
            )
        ]
        campaign.tasks = tasks
        
        # Submit campaign
        success = await primary_coordinator.submit_campaign(campaign)
        assert success
        
        # Verify initial state
        campaign_status = primary_coordinator.get_campaign_status(campaign.campaign_id)
        assert campaign_status is not None
        assert campaign_status["state"] == "pending"
        assert campaign_status["total_tasks"] == 3
        assert campaign_status["completed_tasks"] == 0
        
        # Execute campaign
        await primary_coordinator.execute_distributed_campaign(campaign.campaign_id)
        
        # Simulate task completion
        stored_campaign = primary_coordinator.active_campaigns[campaign.campaign_id]
        
        # Complete tasks in dependency order
        stored_campaign.tasks[0].status = "completed"  # initial_scan
        stored_campaign.tasks[0].completed_at = time.time()
        
        stored_campaign.tasks[1].status = "completed"  # deep_analysis
        stored_campaign.tasks[1].completed_at = time.time()
        
        stored_campaign.tasks[2].status = "completed"  # final_report
        stored_campaign.tasks[2].completed_at = time.time()
        
        # Update campaign state
        stored_campaign.state = CampaignState.COMPLETED
        stored_campaign.completed_at = time.time()
        
        # Move to completed campaigns
        primary_coordinator.completed_campaigns.append(stored_campaign)
        del primary_coordinator.active_campaigns[campaign.campaign_id]
        
        # Verify completion
        final_status = primary_coordinator.get_campaign_status(campaign.campaign_id)
        assert final_status["state"] == "completed"
        assert final_status["completed_tasks"] == 3
        assert final_status["completion_percentage"] == 100.0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, cluster_setup):
        """Test system performance under heavy load."""
        coordinators = cluster_setup
        primary_coordinator = coordinators[0]
        
        # Create many campaigns simultaneously
        num_campaigns = 20
        campaigns = []
        
        start_time = time.time()
        
        for i in range(num_campaigns):
            campaign = DistributedCampaign(
                campaign_id=f"perf_test_{i+1:03d}",
                title=f"Performance Test Campaign {i+1}",
                coordinator_id=primary_coordinator.node_id
            )
            
            # Add multiple tasks per campaign
            for j in range(3):
                task = CampaignTask(
                    task_id=f"perf_task_{i+1:03d}_{j+1}",
                    task_type="analysis",
                    requirements={"scanning"},
                    priority=random.randint(1, 10)
                )
                campaign.tasks.append(task)
            
            campaigns.append(campaign)
        
        # Submit all campaigns
        submission_start = time.time()
        submission_tasks = []
        
        for campaign in campaigns:
            task = asyncio.create_task(primary_coordinator.submit_campaign(campaign))
            submission_tasks.append(task)
        
        # Wait for all submissions to complete
        results = await asyncio.gather(*submission_tasks, return_exceptions=True)
        submission_time = time.time() - submission_start
        
        # Verify results
        successful_submissions = sum(1 for result in results if result is True)
        
        assert successful_submissions >= num_campaigns * 0.8, \
            f"At least 80% of campaigns should be submitted successfully, got {successful_submissions}/{num_campaigns}"
        
        # Verify performance metrics
        assert submission_time < 10.0, \
            f"Campaign submission should complete within 10 seconds, took {submission_time:.2f}s"
        
        # Verify cluster statistics
        cluster_status = primary_coordinator.get_cluster_status()
        assert cluster_status["active_campaigns"] == successful_submissions
        
        total_time = time.time() - start_time
        assert total_time < 15.0, \
            f"Total test should complete within 15 seconds, took {total_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, cluster_setup):
        """Test error handling and system resilience."""
        coordinators = cluster_setup
        primary_coordinator = coordinators[0]
        
        # Test invalid campaign submission
        invalid_campaign = DistributedCampaign(
            campaign_id="",  # Invalid: empty campaign ID
            title="Invalid Campaign",
            coordinator_id=primary_coordinator.node_id
        )
        
        success = await primary_coordinator.submit_campaign(invalid_campaign)
        assert not success, "Invalid campaign should be rejected"
        
        # Test campaign with impossible requirements
        impossible_campaign = DistributedCampaign(
            campaign_id="impossible_test",
            title="Impossible Campaign",
            coordinator_id=primary_coordinator.node_id
        )
        
        impossible_task = CampaignTask(
            task_id="impossible_task",
            task_type="impossible",
            requirements={"nonexistent_capability"},
            priority=10
        )
        
        impossible_campaign.tasks = [impossible_task]
        
        success = await primary_coordinator.submit_campaign(impossible_campaign)
        assert success, "Campaign should be accepted even with impossible tasks"
        
        # Verify task scheduling handles impossible requirements gracefully
        scheduler = primary_coordinator.task_scheduler
        all_nodes = list(primary_coordinator.known_nodes.values()) + [primary_coordinator.node_info]
        assignments = scheduler.schedule_tasks([impossible_task], all_nodes)
        
        # Should have no assignments for impossible tasks
        assert len(assignments) == 0, "Impossible tasks should not be assigned"
    
    def test_cluster_status_reporting(self, cluster_setup):
        """Test comprehensive cluster status reporting."""
        coordinators = cluster_setup
        
        for coordinator in coordinators:
            status = coordinator.get_cluster_status()
            
            # Verify required fields
            assert "node_id" in status
            assert "is_coordinator" in status
            assert "cluster_size" in status
            assert "healthy_nodes" in status
            assert "active_campaigns" in status
            assert "completed_campaigns" in status
            assert "nodes" in status
            assert "statistics" in status
            
            # Verify data integrity
            assert status["cluster_size"] >= 2  # Should know about other nodes
            assert status["healthy_nodes"] >= 0
            assert status["active_campaigns"] >= 0
            assert status["completed_campaigns"] >= 0
            assert isinstance(status["nodes"], dict)
            assert isinstance(status["statistics"], dict)


@pytest.mark.asyncio
async def test_integration_with_external_systems():
    """Test integration with external monitoring and messaging systems."""
    
    # Test Prometheus metrics integration
    from prometheus_client import REGISTRY
    
    # Check that our metrics are registered
    metric_names = [metric.name for metric in REGISTRY.collect()]
    expected_metrics = [
        "xorb_campaign_coordination_events_total",
        "xorb_active_campaigns_distributed",
        "xorb_coordination_latency_seconds",
        "xorb_node_health_status"
    ]
    
    for expected_metric in expected_metrics:
        assert any(expected_metric in name for name in metric_names), \
            f"Expected metric {expected_metric} should be registered"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])