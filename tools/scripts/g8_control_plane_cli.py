#!/usr/bin/env python3
"""
XORB Phase G8 Control Plane CLI Tool
Command-line interface for WFQ scheduler and quota management operations.

Provides administrative tools for monitoring and managing the control plane,
including tenant management, quota adjustment, and fairness analysis.
"""

import asyncio
import json
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
import time

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "api"))

from app.services.g8_control_plane_service import (
    G8ControlPlaneService,
    ResourceType,
    RequestPriority
)


class G8ControlPlaneCLI:
    """CLI interface for G8 Control Plane management."""

    def __init__(self):
        self.control_plane: G8ControlPlaneService = None

    async def initialize(self, storage_path: str = "control_plane_storage"):
        """Initialize the control plane service."""
        print("🎛️ Initializing G8 Control Plane...")
        self.control_plane = G8ControlPlaneService(storage_path)
        await self.control_plane.start()
        print("✅ Control Plane initialized successfully")

    async def cleanup(self):
        """Cleanup control plane service."""
        if self.control_plane:
            await self.control_plane.stop()

    async def create_tenant(self, tenant_id: str, tier: str = "starter"):
        """Create a new tenant profile."""
        print(f"🏢 Creating tenant profile for {tenant_id} (tier: {tier})")

        try:
            profile = self.control_plane.quota_manager.create_tenant_profile(
                tenant_id=tenant_id,
                tier=tier
            )

            print(f"✅ Tenant {tenant_id} created successfully")
            print(f"   Tier: {profile.tier}")
            print(f"   WFQ Weight: {profile.weight}")
            print(f"   Quotas:")
            for resource_type, limit in profile.quotas.items():
                print(f"     • {resource_type.value}: {limit}")

            return True

        except Exception as e:
            print(f"❌ Failed to create tenant: {e}")
            return False

    async def submit_request(
        self,
        tenant_id: str,
        resource_type: str,
        priority: str = "medium",
        amount: int = 1,
        duration: float = 1.0
    ):
        """Submit a resource request."""
        print(f"📥 Submitting {resource_type} request for tenant {tenant_id}")

        try:
            resource_enum = ResourceType(resource_type)
            priority_enum = RequestPriority[priority.upper()]

            accepted, message, request_id = await self.control_plane.submit_request(
                tenant_id=tenant_id,
                resource_type=resource_enum,
                priority=priority_enum,
                resource_amount=amount,
                estimated_duration=duration
            )

            if accepted:
                print(f"✅ Request accepted: {message}")
                print(f"   Request ID: {request_id}")
            else:
                print(f"❌ Request rejected: {message}")

            return accepted, request_id

        except ValueError as e:
            print(f"❌ Invalid parameter: {e}")
            return False, None
        except Exception as e:
            print(f"❌ Failed to submit request: {e}")
            return False, None

    async def get_tenant_status(self, tenant_id: str):
        """Get comprehensive tenant status."""
        print(f"📊 Getting status for tenant {tenant_id}")

        try:
            status = await self.control_plane.get_tenant_status(tenant_id)

            print(f"\\n{'='*60}")
            print(f"📊 Tenant Status: {tenant_id}")
            print(f"{'='*60}")
            print(f"Queue Length: {status['queue_length']}")
            print(f"Fairness Score: {status['fairness_metrics']['fairness_score']:.3f}")
            print(f"Resource Starvation Count: {status['fairness_metrics']['resource_starvation_count']}")

            # Usage statistics
            if 'usage_statistics' in status and 'usage_statistics' in status['usage_statistics']:
                print(f"\\n📈 Usage Statistics:")
                for resource_type, stats in status['usage_statistics']['usage_statistics'].items():
                    print(f"  {resource_type}:")
                    print(f"    Current: {stats['current_usage']}/{stats['quota_limit']} ({stats['utilization_percent']:.1f}%)")
                    print(f"    Total Requests: {stats['total_requests']}")
                    print(f"    Rejected: {stats['rejected_requests']}")

            return status

        except Exception as e:
            print(f"❌ Failed to get tenant status: {e}")
            return None

    async def get_system_status(self):
        """Get system-wide control plane status."""
        print("📊 Getting system status...")

        try:
            status = await self.control_plane.get_system_status()

            print(f"\\n{'='*60}")
            print(f"🎛️ Control Plane System Status")
            print(f"{'='*60}")

            # System health
            health = status['system_health']
            print(f"Control Plane Running: {'✅' if health['control_plane_running'] else '❌'}")
            print(f"Total Tenants: {health['total_tenants']}")
            print(f"Total Queued Requests: {health['total_queued_requests']}")
            print(f"Total Processed: {health['processing_statistics']['total_processed']}")
            print(f"Total Rejected: {health['processing_statistics']['total_rejected']}")
            print(f"Fairness Index: {health['fairness_index']:.3f}")
            print(f"Fairness Violations: {health['fairness_violations']}")

            # Queue statistics
            queue_stats = status['queue_statistics']
            print(f"\\n🏃‍♂️ Queue Statistics:")
            print(f"Virtual Time: {queue_stats['virtual_time']:.3f}")
            print(f"Average Wait Time: {queue_stats['average_wait_time_ms']:.1f}ms")
            print(f"Tenant Queues:")
            for tenant_id, queue_length in queue_stats['tenant_queues'].items():
                print(f"  • {tenant_id}: {queue_length} requests")

            # Fairness report
            fairness = status['fairness_report']
            if fairness['violations_count'] > 0:
                print(f"\\n⚠️ Fairness Violations:")
                for violation in fairness['fairness_violations']:
                    print(f"  • {violation['tenant_id']}: {violation['violation_type']} ({violation['severity']})")
            else:
                print(f"\\n✅ No fairness violations detected")

            return status

        except Exception as e:
            print(f"❌ Failed to get system status: {e}")
            return None

    async def update_tenant_quota(
        self,
        tenant_id: str,
        resource_type: str,
        new_limit: int,
        burst_allowance: int = None
    ):
        """Update tenant quota for a specific resource type."""
        print(f"📝 Updating quota for {tenant_id}: {resource_type} → {new_limit}")

        try:
            resource_enum = ResourceType(resource_type)

            # Get current profile
            if tenant_id not in self.control_plane.quota_manager.tenant_profiles:
                print(f"❌ Tenant {tenant_id} not found")
                return False

            profile = self.control_plane.quota_manager.tenant_profiles[tenant_id]
            old_limit = profile.quotas.get(resource_enum, 0)

            # Update quota
            profile.quotas[resource_enum] = new_limit
            if burst_allowance is not None:
                profile.burst_allowance[resource_enum] = burst_allowance
            profile.updated_at = datetime.now(timezone.utc)

            # Save profile
            self.control_plane.quota_manager._save_tenant_profile(profile)

            print(f"✅ Quota updated: {old_limit} → {new_limit}")
            if burst_allowance is not None:
                print(f"   Burst allowance: {burst_allowance}")

            return True

        except ValueError as e:
            print(f"❌ Invalid resource type: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to update quota: {e}")
            return False

    async def monitor_fairness(self, duration: int = 60, interval: int = 10):
        """Monitor fairness metrics in real-time."""
        print(f"👁️ Monitoring fairness for {duration} seconds (interval: {interval}s)")
        print("Press Ctrl+C to stop monitoring...")

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                # Generate fairness report
                report = self.control_plane.fairness_engine.generate_fairness_report()

                # Clear screen and display report
                print("\\033[2J\\033[H")  # Clear screen
                print(f"{'='*70}")
                print(f"🎯 Real-time Fairness Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")

                print(f"System Fairness Index: {report['system_fairness_index']:.3f}")
                print(f"Active Tenants: {report['tenant_count']}")
                print(f"Violations: {report['violations_count']}")

                print(f"\\nTenant Fairness Scores:")
                for tenant_id, metrics in report['tenant_metrics'].items():
                    score = metrics['fairness_score']
                    status = "🟢" if score > 0.8 else "🟡" if score > 0.5 else "🔴"
                    print(f"  {status} {tenant_id}: {score:.3f}")

                if report['violations_count'] > 0:
                    print(f"\\n⚠️ Current Violations:")
                    for violation in report['fairness_violations']:
                        print(f"  • {violation['tenant_id']}: {violation['violation_type']} ({violation['severity']})")

                print(f"\\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")

                # Wait for next interval
                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\\n\\n🛑 Monitoring stopped by user")

    async def load_test(
        self,
        num_tenants: int = 5,
        requests_per_tenant: int = 10,
        request_rate: float = 1.0
    ):
        """Run a load test to demonstrate WFQ scheduling."""
        print(f"🧪 Running load test: {num_tenants} tenants, {requests_per_tenant} requests each")

        # Create test tenants with different tiers
        tiers = ["enterprise", "professional", "starter"]
        test_tenants = []

        for i in range(num_tenants):
            tenant_id = f"test-tenant-{i+1}"
            tier = tiers[i % len(tiers)]

            await self.create_tenant(tenant_id, tier)
            test_tenants.append((tenant_id, tier))

        print(f"\\n🚀 Submitting {num_tenants * requests_per_tenant} requests...")

        # Submit requests from all tenants
        submitted_requests = []
        for tenant_id, tier in test_tenants:
            for j in range(requests_per_tenant):
                accepted, request_id = await self.submit_request(
                    tenant_id=tenant_id,
                    resource_type="api_requests",
                    priority="medium",
                    amount=1,
                    duration=1.0
                )

                if accepted:
                    submitted_requests.append((tenant_id, request_id))

                # Rate limiting
                await asyncio.sleep(1.0 / request_rate)

        print(f"✅ Submitted {len(submitted_requests)} requests")

        # Monitor processing for a while
        print(f"\\n📊 Monitoring processing...")
        for _ in range(30):  # Monitor for 30 seconds
            status = await self.get_system_status()
            if status:
                queued = status['system_health']['total_queued_requests']
                processed = status['system_health']['processing_statistics']['total_processed']
                print(f"  Queued: {queued}, Processed: {processed}")

                if queued == 0:
                    break

            await asyncio.sleep(1)

        print(f"\\n🏁 Load test completed")

    def print_help(self):
        """Print CLI help information."""
        print(f"{'='*70}")
        print(f"🎛️ XORB Phase G8 Control Plane CLI")
        print(f"{'='*70}")
        print(f"Available commands:")
        print(f"  create-tenant <tenant_id> <tier>     - Create tenant profile")
        print(f"  submit-request <tenant> <resource>   - Submit resource request")
        print(f"  tenant-status <tenant_id>            - Get tenant status")
        print(f"  system-status                        - Get system status")
        print(f"  update-quota <tenant> <resource> <limit> - Update tenant quota")
        print(f"  monitor-fairness [duration]          - Monitor fairness metrics")
        print(f"  load-test [tenants] [requests]       - Run load test")
        print(f"")
        print(f"Resource types: {', '.join([rt.value for rt in ResourceType])}")
        print(f"Tenant tiers: enterprise, professional, starter")
        print(f"Priorities: critical, high, medium, low, background")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="XORB G8 Control Plane CLI")
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--storage-path", default="control_plane_storage", help="Storage path for control plane data")

    args = parser.parse_args()

    cli = G8ControlPlaneCLI()

    try:
        await cli.initialize(args.storage_path)

        if args.command == "create-tenant":
            if len(args.args) < 1:
                print("Usage: create-tenant <tenant_id> [tier]")
                return 1
            tenant_id = args.args[0]
            tier = args.args[1] if len(args.args) > 1 else "starter"
            await cli.create_tenant(tenant_id, tier)

        elif args.command == "submit-request":
            if len(args.args) < 2:
                print("Usage: submit-request <tenant_id> <resource_type> [priority] [amount] [duration]")
                return 1
            tenant_id = args.args[0]
            resource_type = args.args[1]
            priority = args.args[2] if len(args.args) > 2 else "medium"
            amount = int(args.args[3]) if len(args.args) > 3 else 1
            duration = float(args.args[4]) if len(args.args) > 4 else 1.0
            await cli.submit_request(tenant_id, resource_type, priority, amount, duration)

        elif args.command == "tenant-status":
            if len(args.args) < 1:
                print("Usage: tenant-status <tenant_id>")
                return 1
            tenant_id = args.args[0]
            await cli.get_tenant_status(tenant_id)

        elif args.command == "system-status":
            await cli.get_system_status()

        elif args.command == "update-quota":
            if len(args.args) < 3:
                print("Usage: update-quota <tenant_id> <resource_type> <new_limit> [burst_allowance]")
                return 1
            tenant_id = args.args[0]
            resource_type = args.args[1]
            new_limit = int(args.args[2])
            burst_allowance = int(args.args[3]) if len(args.args) > 3 else None
            await cli.update_tenant_quota(tenant_id, resource_type, new_limit, burst_allowance)

        elif args.command == "monitor-fairness":
            duration = int(args.args[0]) if args.args else 60
            await cli.monitor_fairness(duration)

        elif args.command == "load-test":
            num_tenants = int(args.args[0]) if args.args else 5
            requests_per_tenant = int(args.args[1]) if len(args.args) > 1 else 10
            await cli.load_test(num_tenants, requests_per_tenant)

        elif args.command == "help":
            cli.print_help()

        else:
            print(f"Unknown command: {args.command}")
            cli.print_help()
            return 1

        return 0

    except Exception as e:
        print(f"❌ CLI Error: {e}")
        return 1

    finally:
        await cli.cleanup()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\\n🛑 Interrupted by user")
        sys.exit(1)
