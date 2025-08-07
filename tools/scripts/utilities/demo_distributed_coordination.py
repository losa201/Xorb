#!/usr/bin/env python3
"""
Distributed Campaign Coordination Demonstration

This script demonstrates the distributed campaign coordination capabilities
of the XORB ecosystem, including multi-node setup, campaign distribution,
consensus algorithms, and fault tolerance mechanisms.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add the xorb_core package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "xorb_core"))

import structlog
from orchestration.distributed_campaign_coordinator import (
    CampaignState,
    CampaignTask,
    DistributedCampaign,
    DistributedCampaignCoordinator,
    NodeRole,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class DistributedCoordinationDemo:
    """Demonstration of distributed campaign coordination."""

    def __init__(self):
        self.coordinators = []
        self.demo_results = {}

    async def setup_cluster(self, num_nodes: int = 3) -> List[DistributedCampaignCoordinator]:
        """Set up a demonstration cluster with multiple nodes."""
        logger.info("Setting up distributed coordination cluster", num_nodes=num_nodes)

        # Create coordinator nodes
        for i in range(num_nodes):
            coordinator = DistributedCampaignCoordinator(
                node_id=f"demo_node_{i+1}",
                host="localhost",
                port=8080 + i
            )

            # Configure different roles
            if i == 0:
                coordinator.node_info.role = NodeRole.COORDINATOR
                coordinator.node_info.capabilities = {"scanning", "analysis", "reporting", "coordination"}
            elif i == 1:
                coordinator.node_info.role = NodeRole.EXECUTOR
                coordinator.node_info.capabilities = {"scanning", "exploitation", "persistence"}
            else:
                coordinator.node_info.role = NodeRole.SPECIALIST
                coordinator.node_info.capabilities = {"steganography", "crypto_analysis", "forensics"}

            await coordinator.start_coordinator()
            self.coordinators.append(coordinator)

            logger.info("Node started",
                       node_id=coordinator.node_id,
                       role=coordinator.node_info.role.value,
                       capabilities=list(coordinator.node_info.capabilities))

        # Allow nodes to discover each other
        await asyncio.sleep(2)

        # Manually connect nodes for demo (in production, this would be automatic)
        for coordinator in self.coordinators:
            for other in self.coordinators:
                if coordinator.node_id != other.node_id:
                    coordinator.known_nodes[other.node_id] = other.node_info

        logger.info("Cluster setup complete",
                   total_nodes=len(self.coordinators),
                   coordinator_nodes=[c.node_id for c in self.coordinators if c.node_info.role == NodeRole.COORDINATOR])

        return self.coordinators

    async def demonstrate_campaign_distribution(self):
        """Demonstrate distributed campaign execution."""
        logger.info("Starting campaign distribution demonstration")

        # Get the primary coordinator
        primary_coordinator = self.coordinators[0]

        # Create a comprehensive security assessment campaign
        campaign = DistributedCampaign(
            campaign_id="demo_security_assessment",
            title="Comprehensive Security Assessment",
            description="Multi-phase security assessment across multiple nodes",
            coordinator_id=primary_coordinator.node_id,
            state=CampaignState.PENDING
        )

        # Add diverse tasks requiring different capabilities
        tasks = [
            CampaignTask(
                task_id="network_scan",
                task_type="reconnaissance",
                description="Comprehensive network discovery scan",
                requirements={"scanning"},
                priority=9,
                estimated_duration=300,
                payload={"target": "192.168.1.0/24", "scan_type": "comprehensive"}
            ),
            CampaignTask(
                task_id="vulnerability_analysis",
                task_type="analysis",
                description="Analyze discovered services for vulnerabilities",
                requirements={"analysis"},
                priority=8,
                estimated_duration=600,
                payload={"scan_results": "network_scan_output"},
                dependencies=["network_scan"]
            ),
            CampaignTask(
                task_id="crypto_assessment",
                task_type="specialist",
                description="Cryptographic implementation analysis",
                requirements={"crypto_analysis"},
                priority=7,
                estimated_duration=450,
                payload={"target_protocols": ["TLS", "SSH", "IPSec"]}
            ),
            CampaignTask(
                task_id="steganography_detection",
                task_type="specialist",
                description="Hidden data detection in media files",
                requirements={"steganography"},
                priority=6,
                estimated_duration=300,
                payload={"media_sources": ["web_images", "document_attachments"]}
            ),
            CampaignTask(
                task_id="exploitation_attempt",
                task_type="exploitation",
                description="Controlled exploitation of identified vulnerabilities",
                requirements={"exploitation"},
                priority=8,
                estimated_duration=900,
                payload={"vuln_list": "vulnerability_analysis_output"},
                dependencies=["vulnerability_analysis"]
            ),
            CampaignTask(
                task_id="forensic_analysis",
                task_type="forensics",
                description="Post-exploitation forensic analysis",
                requirements={"forensics"},
                priority=5,
                estimated_duration=600,
                payload={"evidence_sources": ["system_logs", "memory_dumps"]},
                dependencies=["exploitation_attempt"]
            ),
            CampaignTask(
                task_id="comprehensive_report",
                task_type="reporting",
                description="Generate comprehensive security assessment report",
                requirements={"reporting"},
                priority=10,
                estimated_duration=300,
                payload={"include_all_phases": True},
                dependencies=["network_scan", "vulnerability_analysis", "crypto_assessment", "steganography_detection", "exploitation_attempt", "forensic_analysis"]
            )
        ]

        campaign.tasks = tasks

        # Submit campaign for distributed execution
        success = await primary_coordinator.submit_campaign(campaign)

        if success:
            logger.info("Campaign submitted successfully", campaign_id=campaign.campaign_id)

            # Monitor campaign progress
            await self._monitor_campaign_progress(campaign.campaign_id)

            # Execute campaign distribution
            await primary_coordinator.execute_distributed_campaign(campaign.campaign_id)

            self.demo_results["campaign_distribution"] = {
                "status": "success",
                "campaign_id": campaign.campaign_id,
                "tasks_distributed": len(tasks),
                "participating_nodes": len(self.coordinators)
            }
        else:
            logger.error("Failed to submit campaign")
            self.demo_results["campaign_distribution"] = {"status": "failed"}

    async def demonstrate_fault_tolerance(self):
        """Demonstrate fault tolerance mechanisms."""
        logger.info("Starting fault tolerance demonstration")

        if len(self.coordinators) < 3:
            logger.warning("Need at least 3 nodes for fault tolerance demo")
            return

        # Simulate node failure
        failing_node = self.coordinators[1]
        logger.info("Simulating node failure", node_id=failing_node.node_id)

        # Mark node as unhealthy
        failing_node.node_info.status = "unhealthy"
        failing_node.node_info.last_heartbeat = time.time() - 300  # 5 minutes ago

        # Allow other nodes to detect the failure
        await asyncio.sleep(3)

        # Check if remaining nodes detected the failure
        healthy_coordinators = [c for c in self.coordinators if c.node_id != failing_node.node_id]

        fault_tolerance_results = {}
        for coordinator in healthy_coordinators:
            cluster_status = coordinator.get_cluster_status()
            fault_tolerance_results[coordinator.node_id] = {
                "detected_failure": cluster_status["healthy_nodes"] == len(self.coordinators) - 1,
                "healthy_nodes": cluster_status["healthy_nodes"],
                "total_nodes": cluster_status["cluster_size"]
            }

        # Simulate node recovery
        logger.info("Simulating node recovery", node_id=failing_node.node_id)
        failing_node.node_info.status = "healthy"
        failing_node.node_info.last_heartbeat = time.time()

        await asyncio.sleep(2)

        # Check recovery detection
        for coordinator in healthy_coordinators:
            cluster_status = coordinator.get_cluster_status()
            fault_tolerance_results[coordinator.node_id]["recovery_detected"] = cluster_status["healthy_nodes"] == len(self.coordinators)

        self.demo_results["fault_tolerance"] = fault_tolerance_results
        logger.info("Fault tolerance demonstration complete", results=fault_tolerance_results)

    async def demonstrate_consensus_algorithm(self):
        """Demonstrate consensus algorithm for critical decisions."""
        logger.info("Starting consensus algorithm demonstration")

        # Propose a critical configuration change
        proposal = {
            "type": "security_policy_update",
            "description": "Update maximum concurrent campaign limit",
            "proposed_value": 5,
            "current_value": 3,
            "proposer": self.coordinators[0].node_id
        }

        consensus_results = {}

        # Each node votes on the proposal
        for i, coordinator in enumerate(self.coordinators):
            # Simulate different voting patterns
            if i == 0 or i == 1:  # Proposer always votes yes
                vote = True
            else:  # Other nodes vote based on some logic
                vote = proposal["proposed_value"] <= 5  # Accept if reasonable

            consensus_results[coordinator.node_id] = {
                "vote": vote,
                "reasoning": "Security policy within acceptable limits" if vote else "Proposed limit too high"
            }

        # Calculate consensus result
        yes_votes = sum(1 for result in consensus_results.values() if result["vote"])
        total_votes = len(consensus_results)
        consensus_reached = yes_votes > total_votes / 2

        consensus_summary = {
            "proposal": proposal,
            "votes": consensus_results,
            "yes_votes": yes_votes,
            "total_votes": total_votes,
            "consensus_reached": consensus_reached,
            "decision": "approved" if consensus_reached else "rejected"
        }

        self.demo_results["consensus"] = consensus_summary
        logger.info("Consensus algorithm demonstration complete",
                   consensus_reached=consensus_reached,
                   decision=consensus_summary["decision"])

    async def demonstrate_load_balancing(self):
        """Demonstrate intelligent load balancing across nodes."""
        logger.info("Starting load balancing demonstration")

        # Create multiple small campaigns to test load distribution
        load_test_campaigns = []

        for i in range(6):  # More campaigns than nodes
            campaign = DistributedCampaign(
                campaign_id=f"load_test_campaign_{i+1}",
                title=f"Load Test Campaign {i+1}",
                description=f"Campaign for load balancing test {i+1}",
                coordinator_id=self.coordinators[0].node_id,
                state=CampaignState.PENDING
            )

            # Add a simple task
            task = CampaignTask(
                task_id=f"load_test_task_{i+1}",
                task_type="analysis",
                description=f"Load test task {i+1}",
                requirements={"analysis"},
                priority=5,
                estimated_duration=60
            )

            campaign.tasks = [task]
            load_test_campaigns.append(campaign)

        # Submit all campaigns
        coordinator = self.coordinators[0]
        for campaign in load_test_campaigns:
            await coordinator.submit_campaign(campaign)

        # Check load distribution
        load_distribution = {}
        for node_coordinator in self.coordinators:
            status = node_coordinator.get_cluster_status()
            load_distribution[node_coordinator.node_id] = {
                "active_campaigns": status["active_campaigns"],
                "role": node_coordinator.node_info.role.value,
                "capabilities": list(node_coordinator.node_info.capabilities)
            }

        self.demo_results["load_balancing"] = {
            "campaigns_submitted": len(load_test_campaigns),
            "node_distribution": load_distribution,
            "total_nodes": len(self.coordinators)
        }

        logger.info("Load balancing demonstration complete",
                   campaigns=len(load_test_campaigns),
                   distribution=load_distribution)

    async def _monitor_campaign_progress(self, campaign_id: str, duration: int = 30):
        """Monitor campaign progress for a specified duration."""
        logger.info("Monitoring campaign progress", campaign_id=campaign_id, duration=duration)

        start_time = time.time()
        while time.time() - start_time < duration:
            # Get status from primary coordinator
            coordinator = self.coordinators[0]
            status = coordinator.get_campaign_status(campaign_id)

            if status:
                logger.info("Campaign progress update",
                           campaign_id=campaign_id,
                           state=status["state"],
                           completion=f"{status['completion_percentage']:.1f}%",
                           completed_tasks=status["completed_tasks"],
                           total_tasks=status["total_tasks"])

                if status["state"] in ["completed", "failed"]:
                    break

            await asyncio.sleep(5)

    async def generate_demonstration_report(self):
        """Generate a comprehensive demonstration report."""
        logger.info("Generating demonstration report")

        # Collect cluster statistics
        cluster_stats = {}
        for coordinator in self.coordinators:
            status = coordinator.get_cluster_status()
            cluster_stats[coordinator.node_id] = status

        report = {
            "demonstration_summary": {
                "timestamp": time.time(),
                "cluster_size": len(self.coordinators),
                "total_demonstrations": len(self.demo_results),
                "successful_demonstrations": len([r for r in self.demo_results.values() if r.get("status") != "failed"])
            },
            "cluster_configuration": {
                "nodes": [
                    {
                        "node_id": c.node_id,
                        "role": c.node_info.role.value,
                        "capabilities": list(c.node_info.capabilities),
                        "host": c.host,
                        "port": c.port
                    }
                    for c in self.coordinators
                ]
            },
            "demonstration_results": self.demo_results,
            "cluster_statistics": cluster_stats,
            "performance_metrics": {
                "total_messages_sent": sum(stats.get("messages_sent", 0) for stats in cluster_stats.values()),
                "total_campaigns_processed": sum(stats.get("active_campaigns", 0) + stats.get("completed_campaigns", 0) for stats in cluster_stats.values()),
                "average_node_health": sum(1 for stats in cluster_stats.values() if stats.get("healthy_nodes", 0) > 0) / len(cluster_stats)
            }
        }

        # Save report to file
        report_path = Path(__file__).parent.parent / "distributed_coordination_demo_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Demonstration report saved", report_path=str(report_path))

        return report

    async def cleanup(self):
        """Clean up demonstration resources."""
        logger.info("Cleaning up demonstration resources")

        for coordinator in self.coordinators:
            await coordinator.stop_coordinator()

        logger.info("Cleanup complete")


async def main():
    """Main demonstration function."""
    print("🔗 XORB Distributed Campaign Coordination Demonstration")
    print("=" * 60)

    demo = DistributedCoordinationDemo()

    try:
        # Setup cluster
        print("\n📡 Setting up demonstration cluster...")
        await demo.setup_cluster(num_nodes=3)

        # Run demonstrations
        print("\n🎯 Demonstrating campaign distribution...")
        await demo.demonstrate_campaign_distribution()

        print("\n🛡️ Demonstrating fault tolerance...")
        await demo.demonstrate_fault_tolerance()

        print("\n🗳️ Demonstrating consensus algorithm...")
        await demo.demonstrate_consensus_algorithm()

        print("\n⚖️ Demonstrating load balancing...")
        await demo.demonstrate_load_balancing()

        # Generate report
        print("\n📊 Generating demonstration report...")
        report = await demo.generate_demonstration_report()

        print("\n✅ Demonstration Complete!")
        print("📄 Report saved to: distributed_coordination_demo_report.json")
        print(f"🎯 Total demonstrations: {len(demo.demo_results)}")
        print(f"📊 Cluster size: {len(demo.coordinators)} nodes")

        # Print summary
        print("\n📋 Demonstration Summary:")
        for demo_name, results in demo.demo_results.items():
            status = results.get("status", "completed")
            print(f"  • {demo_name.replace('_', ' ').title()}: {status.upper()}")

    except Exception as e:
        logger.error("Demonstration failed", error=str(e), exc_info=True)
        print(f"\n❌ Demonstration failed: {e}")

    finally:
        # Cleanup
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
