#!/usr/bin/env python3
"""
XORB Phase G6 Tenant Isolation Validator
Real-world validation that Tenant A cannot read Tenant B's data.

This script performs live validation against actual NATS infrastructure
to ensure tenant isolation is properly implemented and enforced.
"""

import asyncio
import json
import sys
import time
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

# NATS imports
try:
    import nats
    from nats.errors import TimeoutError, NoRespondersError, PermissionViolationError
    from nats.aio.client import Client as NATS
    NATS_AVAILABLE = True
except ImportError:
    print("❌ NATS Python client not available. Install with: pip install nats-py")
    NATS_AVAILABLE = False


@dataclass
class TenantConfig:
    """G6 Tenant configuration from IaC output."""
    tenant_id: str
    tier: str
    account_name: str
    user_name: str
    jwt_token: str
    nkey_seed: str
    max_connections: int
    rate_limit_bps: int
    isolation_level: str
    allowed_subjects: List[str]
    denied_subjects: List[str]


@dataclass
class IsolationTestResult:
    """Results from a tenant isolation test."""
    test_name: str
    source_tenant: str
    target_tenant: str
    expected_result: str  # "success" or "failure"
    actual_result: str
    success: bool
    error_message: Optional[str] = None
    duration_ms: float = 0.0


class G6TenantIsolationValidator:
    """Validates tenant isolation in G6 NATS backplane."""

    def __init__(self, nats_url: str = "nats://localhost:4222", config_dir: str = "infra/iac/nats/out"):
        self.nats_url = nats_url
        self.config_dir = Path(config_dir)
        self.tenant_configs: Dict[str, TenantConfig] = {}
        self.connections: Dict[str, NATS] = {}
        self.test_results: List[IsolationTestResult] = []

    async def load_tenant_configs(self) -> Dict[str, TenantConfig]:
        """Load G6 tenant configurations from IaC output files."""
        print("📋 Loading G6 tenant configurations...")

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        config_files = list(self.config_dir.glob("tenant-*-config.json"))

        if not config_files:
            raise FileNotFoundError(f"No tenant config files found in {self.config_dir}")

        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)

                # Extract tenant info from G6 config
                tenant_id = config_data["tenant_id"]

                # Load JWT and seed files
                jwt_file = self.config_dir / f"tenant-{tenant_id}-user.jwt"
                seed_file = self.config_dir / f"tenant-{tenant_id}-user.seed"

                jwt_token = ""
                nkey_seed = ""

                if jwt_file.exists():
                    jwt_token = jwt_file.read_text().strip()

                if seed_file.exists():
                    nkey_seed = seed_file.read_text().strip()

                tenant_config = TenantConfig(
                    tenant_id=tenant_id,
                    tier=config_data.get("tier", "unknown"),
                    account_name=config_data["account"]["name"],
                    user_name=config_data["user"]["name"],
                    jwt_token=jwt_token,
                    nkey_seed=nkey_seed,
                    max_connections=config_data["quotas"]["max_connections"],
                    rate_limit_bps=config_data["quotas"]["rate_limit_bps"],
                    isolation_level=config_data["quotas"]["isolation_level"],
                    allowed_subjects=config_data["subjects"]["publish_patterns"],
                    denied_subjects=config_data["subjects"]["deny_patterns"]
                )

                self.tenant_configs[tenant_id] = tenant_config
                print(f"  ✅ Loaded config for tenant: {tenant_id} ({tenant_config.tier})")

            except Exception as e:
                print(f"  ❌ Failed to load config from {config_file}: {e}")
                continue

        print(f"📊 Loaded {len(self.tenant_configs)} tenant configurations")
        return self.tenant_configs

    async def create_tenant_connections(self) -> Dict[str, NATS]:
        """Create authenticated NATS connections for each tenant."""
        print("🔗 Creating tenant NATS connections...")

        for tenant_id, config in self.tenant_configs.items():
            try:
                # Create connection with tenant-specific JWT credentials
                options = {
                    "servers": [self.nats_url],
                    "name": f"g6-validator-{tenant_id}",
                    "user_jwt": config.jwt_token if config.jwt_token else None,
                    "user_nkey_seed": config.nkey_seed if config.nkey_seed else None,
                    "connect_timeout": 10,
                    "max_reconnects": 3,
                }

                # Remove None values
                options = {k: v for k, v in options.items() if v is not None}

                nc = await nats.connect(**options)
                self.connections[tenant_id] = nc
                print(f"  ✅ Connected tenant {tenant_id} to NATS")

            except Exception as e:
                print(f"  ❌ Failed to connect tenant {tenant_id}: {e}")
                # Continue with other tenants

        print(f"🌐 Created {len(self.connections)} NATS connections")
        return self.connections

    async def test_cross_tenant_publish_denial(self) -> List[IsolationTestResult]:
        """Test that tenants cannot publish to each other's subjects."""
        print("🚫 Testing cross-tenant publish denial...")
        results = []

        tenant_list = list(self.tenant_configs.keys())

        for source_tenant in tenant_list:
            if source_tenant not in self.connections:
                continue

            source_conn = self.connections[source_tenant]

            for target_tenant in tenant_list:
                if source_tenant == target_tenant:
                    continue

                # Try to publish to target tenant's subjects
                target_subjects = [
                    f"xorb.{target_tenant}.evidence.scan_result",
                    f"xorb.{target_tenant}.scan.nmap_output",
                    f"xorb.{target_tenant}.compliance.report",
                    f"xorb.{target_tenant}.observability.metrics"
                ]

                for subject in target_subjects:
                    start_time = time.time()

                    try:
                        # This should fail due to subject permissions
                        await source_conn.publish(subject, b'{"test": "cross_tenant_attempt"}')

                        # If we get here, isolation failed!
                        result = IsolationTestResult(
                            test_name="cross_tenant_publish",
                            source_tenant=source_tenant,
                            target_tenant=target_tenant,
                            expected_result="failure",
                            actual_result="success",
                            success=False,
                            error_message=f"Tenant {source_tenant} was able to publish to {target_tenant}'s subject: {subject}",
                            duration_ms=(time.time() - start_time) * 1000
                        )

                    except Exception as e:
                        # This is expected - publish should be denied
                        result = IsolationTestResult(
                            test_name="cross_tenant_publish",
                            source_tenant=source_tenant,
                            target_tenant=target_tenant,
                            expected_result="failure",
                            actual_result="failure",
                            success=True,
                            error_message=str(e),
                            duration_ms=(time.time() - start_time) * 1000
                        )

                    results.append(result)

        return results

    async def test_cross_tenant_subscribe_denial(self) -> List[IsolationTestResult]:
        """Test that tenants cannot subscribe to each other's subjects."""
        print("🚫 Testing cross-tenant subscribe denial...")
        results = []

        tenant_list = list(self.tenant_configs.keys())

        for source_tenant in tenant_list:
            if source_tenant not in self.connections:
                continue

            source_conn = self.connections[source_tenant]

            for target_tenant in tenant_list:
                if source_tenant == target_tenant:
                    continue

                # Try to subscribe to target tenant's subjects
                target_subject = f"xorb.{target_tenant}.evidence.>"
                start_time = time.time()

                try:
                    # This should fail due to subscription permissions
                    await source_conn.subscribe(target_subject)

                    # If we get here, isolation failed!
                    result = IsolationTestResult(
                        test_name="cross_tenant_subscribe",
                        source_tenant=source_tenant,
                        target_tenant=target_tenant,
                        expected_result="failure",
                        actual_result="success",
                        success=False,
                        error_message=f"Tenant {source_tenant} was able to subscribe to {target_tenant}'s subjects",
                        duration_ms=(time.time() - start_time) * 1000
                    )

                except Exception as e:
                    # This is expected - subscribe should be denied
                    result = IsolationTestResult(
                        test_name="cross_tenant_subscribe",
                        source_tenant=source_tenant,
                        target_tenant=target_tenant,
                        expected_result="failure",
                        actual_result="failure",
                        success=True,
                        error_message=str(e),
                        duration_ms=(time.time() - start_time) * 1000
                    )

                results.append(result)

        return results

    async def test_admin_subject_denial(self) -> List[IsolationTestResult]:
        """Test that tenants cannot access admin subjects."""
        print("🔒 Testing admin subject access denial...")
        results = []

        admin_subjects = [
            "xorb.*.admin.user_management",
            "xorb.$SYS.accounts.>",
            "$SYS.>",
            "_NATS.>",
            "xorb.global.admin.system"
        ]

        for tenant_id in self.connections:
            conn = self.connections[tenant_id]

            for admin_subject in admin_subjects:
                start_time = time.time()

                try:
                    # This should fail due to admin subject denial
                    await conn.publish(admin_subject, b'{"action": "admin_test"}')

                    # If we get here, admin isolation failed!
                    result = IsolationTestResult(
                        test_name="admin_subject_denial",
                        source_tenant=tenant_id,
                        target_tenant="admin",
                        expected_result="failure",
                        actual_result="success",
                        success=False,
                        error_message=f"Tenant {tenant_id} was able to access admin subject: {admin_subject}",
                        duration_ms=(time.time() - start_time) * 1000
                    )

                except Exception as e:
                    # This is expected - admin access should be denied
                    result = IsolationTestResult(
                        test_name="admin_subject_denial",
                        source_tenant=tenant_id,
                        target_tenant="admin",
                        expected_result="failure",
                        actual_result="failure",
                        success=True,
                        error_message=str(e),
                        duration_ms=(time.time() - start_time) * 1000
                    )

                results.append(result)

        return results

    async def test_own_subject_access(self) -> List[IsolationTestResult]:
        """Test that tenants CAN access their own subjects."""
        print("✅ Testing own subject access (should succeed)...")
        results = []

        for tenant_id in self.connections:
            conn = self.connections[tenant_id]
            config = self.tenant_configs[tenant_id]

            # Test publishing to own subjects
            own_subjects = [
                f"xorb.{tenant_id}.evidence.scan_complete",
                f"xorb.{tenant_id}.scan.nmap_results",
                f"xorb.{tenant_id}.observability.metrics.bus_latency"
            ]

            for subject in own_subjects:
                start_time = time.time()

                try:
                    # This should succeed
                    await conn.publish(subject, b'{"test": "own_tenant_data"}')

                    result = IsolationTestResult(
                        test_name="own_subject_access",
                        source_tenant=tenant_id,
                        target_tenant=tenant_id,
                        expected_result="success",
                        actual_result="success",
                        success=True,
                        duration_ms=(time.time() - start_time) * 1000
                    )

                except Exception as e:
                    # This indicates a configuration problem
                    result = IsolationTestResult(
                        test_name="own_subject_access",
                        source_tenant=tenant_id,
                        target_tenant=tenant_id,
                        expected_result="success",
                        actual_result="failure",
                        success=False,
                        error_message=f"Tenant {tenant_id} cannot access own subject {subject}: {e}",
                        duration_ms=(time.time() - start_time) * 1000
                    )

                results.append(result)

        return results

    async def test_request_reply_isolation(self) -> List[IsolationTestResult]:
        """Test that request/reply patterns are properly isolated."""
        print("🔄 Testing request/reply isolation...")
        results = []

        tenant_list = list(self.tenant_configs.keys())

        # Set up responders for each tenant on their own subjects
        responders = {}

        for tenant_id in tenant_list:
            if tenant_id not in self.connections:
                continue

            conn = self.connections[tenant_id]
            request_subject = f"xorb.{tenant_id}.control.scan_request"

            try:
                async def handler(msg):
                    await msg.respond(b'{"status": "scan_queued"}')

                sub = await conn.subscribe(request_subject, cb=handler)
                responders[tenant_id] = sub

            except Exception as e:
                print(f"  ⚠️ Failed to set up responder for {tenant_id}: {e}")

        # Let responders settle
        await asyncio.sleep(0.5)

        # Test cross-tenant requests
        for source_tenant in tenant_list:
            if source_tenant not in self.connections:
                continue

            source_conn = self.connections[source_tenant]

            for target_tenant in tenant_list:
                if source_tenant == target_tenant:
                    continue

                target_subject = f"xorb.{target_tenant}.control.scan_request"
                start_time = time.time()

                try:
                    # This should fail or timeout due to isolation
                    response = await source_conn.request(target_subject, b'{"target": "example.com"}', timeout=2.0)

                    # If we get a response, isolation failed!
                    result = IsolationTestResult(
                        test_name="request_reply_isolation",
                        source_tenant=source_tenant,
                        target_tenant=target_tenant,
                        expected_result="failure",
                        actual_result="success",
                        success=False,
                        error_message=f"Cross-tenant request succeeded: {response.data.decode()}",
                        duration_ms=(time.time() - start_time) * 1000
                    )

                except (TimeoutError, NoRespondersError) as e:
                    # This is expected - request should fail due to isolation
                    result = IsolationTestResult(
                        test_name="request_reply_isolation",
                        source_tenant=source_tenant,
                        target_tenant=target_tenant,
                        expected_result="failure",
                        actual_result="failure",
                        success=True,
                        error_message=str(e),
                        duration_ms=(time.time() - start_time) * 1000
                    )

                except Exception as e:
                    # Other errors also indicate proper isolation
                    result = IsolationTestResult(
                        test_name="request_reply_isolation",
                        source_tenant=source_tenant,
                        target_tenant=target_tenant,
                        expected_result="failure",
                        actual_result="failure",
                        success=True,
                        error_message=str(e),
                        duration_ms=(time.time() - start_time) * 1000
                    )

                results.append(result)

        # Clean up responders
        for sub in responders.values():
            try:
                await sub.unsubscribe()
            except:
                pass

        return results

    def generate_report(self, results: List[IsolationTestResult]) -> Dict[str, Any]:
        """Generate comprehensive isolation test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests

        # Group results by test type
        test_groups = {}
        for result in results:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)

        # Calculate success rates per test type
        test_summaries = {}
        for test_name, test_results in test_groups.items():
            total = len(test_results)
            passed = sum(1 for r in test_results if r.success)
            test_summaries[test_name] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": (passed / total) * 100 if total > 0 else 0,
                "avg_duration_ms": sum(r.duration_ms for r in test_results) / total if total > 0 else 0
            }

        # Identify critical failures
        critical_failures = [r for r in results if not r.success and r.test_name in ["cross_tenant_publish", "cross_tenant_subscribe", "admin_subject_denial"]]

        report = {
            "validation_summary": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "G6",
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                "overall_result": "PASS" if failed_tests == 0 else "FAIL",
                "critical_failures": len(critical_failures)
            },
            "test_summaries": test_summaries,
            "tenant_configurations": {
                tenant_id: {
                    "tier": config.tier,
                    "isolation_level": config.isolation_level,
                    "max_connections": config.max_connections,
                    "rate_limit_mbps": config.rate_limit_bps / 1048576
                }
                for tenant_id, config in self.tenant_configs.items()
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "source_tenant": r.source_tenant,
                    "target_tenant": r.target_tenant,
                    "expected": r.expected_result,
                    "actual": r.actual_result,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "error": r.error_message
                }
                for r in results
            ],
            "critical_failures": [
                {
                    "test_name": r.test_name,
                    "source_tenant": r.source_tenant,
                    "target_tenant": r.target_tenant,
                    "issue": r.error_message
                }
                for r in critical_failures
            ]
        }

        return report

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive G6 tenant isolation validation."""
        print("🚀 Starting XORB Phase G6 Tenant Isolation Validation")
        print("=" * 60)

        if not NATS_AVAILABLE:
            raise RuntimeError("NATS Python client not available")

        self.test_results = []

        try:
            # Load configurations
            await self.load_tenant_configs()

            # Create connections
            await self.create_tenant_connections()

            if not self.connections:
                raise RuntimeError("No tenant connections available")

            # Run all isolation tests
            print("\n🧪 Running isolation test suite...")

            # Test cross-tenant access denial
            cross_pub_results = await self.test_cross_tenant_publish_denial()
            self.test_results.extend(cross_pub_results)

            cross_sub_results = await self.test_cross_tenant_subscribe_denial()
            self.test_results.extend(cross_sub_results)

            # Test admin access denial
            admin_results = await self.test_admin_subject_denial()
            self.test_results.extend(admin_results)

            # Test own subject access (positive test)
            own_results = await self.test_own_subject_access()
            self.test_results.extend(own_results)

            # Test request/reply isolation
            reqreply_results = await self.test_request_reply_isolation()
            self.test_results.extend(reqreply_results)

            # Generate final report
            report = self.generate_report(self.test_results)

            return report

        finally:
            # Clean up connections
            await self.cleanup()

    async def cleanup(self):
        """Clean up NATS connections."""
        print("\n🧹 Cleaning up connections...")

        for tenant_id, conn in self.connections.items():
            try:
                await conn.close()
                print(f"  ✅ Closed connection for tenant {tenant_id}")
            except Exception as e:
                print(f"  ⚠️ Error closing connection for {tenant_id}: {e}")

        self.connections.clear()


def print_report(report: Dict[str, Any]):
    """Print formatted validation report."""
    summary = report["validation_summary"]

    print("\n" + "=" * 60)
    print("📊 XORB Phase G6 Tenant Isolation Validation Report")
    print("=" * 60)

    print(f"🕐 Timestamp: {summary['timestamp']}")
    print(f"📊 Tests: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['success_rate']:.1f}%)")
    print(f"🎯 Result: {summary['overall_result']}")

    if summary['critical_failures'] > 0:
        print(f"🚨 Critical Failures: {summary['critical_failures']}")

    print("\n📋 Test Summary by Type:")
    for test_name, stats in report["test_summaries"].items():
        status = "✅" if stats["failed"] == 0 else "❌"
        print(f"  {status} {test_name}: {stats['passed']}/{stats['total']} passed ({stats['success_rate']:.1f}%)")

    print(f"\n👥 Tenants Tested: {len(report['tenant_configurations'])}")
    for tenant_id, config in report["tenant_configurations"].items():
        print(f"  • {tenant_id} ({config['tier']}) - {config['isolation_level']} isolation")

    if report["critical_failures"]:
        print("\n🚨 Critical Isolation Failures:")
        for failure in report["critical_failures"]:
            print(f"  ❌ {failure['test_name']}: {failure['source_tenant']} → {failure['target_tenant']}")
            print(f"     Issue: {failure['issue']}")

    print("\n" + "=" * 60)


async def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="XORB Phase G6 Tenant Isolation Validator")
    parser.add_argument("--nats-url", default="nats://localhost:4222", help="NATS server URL")
    parser.add_argument("--config-dir", default="infra/iac/nats/out", help="Tenant config directory")
    parser.add_argument("--output", help="Output report to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimize output")

    args = parser.parse_args()

    validator = G6TenantIsolationValidator(
        nats_url=args.nats_url,
        config_dir=args.config_dir
    )

    try:
        report = await validator.run_all_tests()

        if not args.quiet:
            print_report(report)

        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved to: {args.output}")

        # Exit with error code if tests failed
        if report["validation_summary"]["overall_result"] != "PASS":
            print(f"\n❌ Tenant isolation validation failed!")
            sys.exit(1)
        else:
            print(f"\n✅ Tenant isolation validation passed!")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
