#!/usr/bin/env python3
"""
XORB Developer CLI - Make target wrapper
Simple CLI entrypoint to wrap common Make targets for developer experience.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("Installing click...")
    subprocess.run([sys.executable, "-m", "pip", "install", "click"], check=True)
    import click


def get_repo_root() -> Path:
    """Find the repository root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "Makefile").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root with Makefile")


def run_make_target(target: str, cwd: Optional[Path] = None) -> int:
    """Run a make target and return the exit code."""
    if cwd is None:
        cwd = get_repo_root()

    cmd = ["make", target]
    click.echo(f"🔧 Running: {' '.join(cmd)}")
    click.echo(f"📁 Working directory: {cwd}")
    click.echo()

    try:
        result = subprocess.run(cmd, cwd=cwd)
        return result.returncode
    except FileNotFoundError:
        click.echo("❌ Error: 'make' command not found", err=True)
        return 1
    except KeyboardInterrupt:
        click.echo("\n⚠️ Interrupted by user", err=True)
        return 130


@click.group()
@click.version_option(version="1.0.0", prog_name="xorbctl")
def cli():
    """XORB Developer CLI - Common development operations."""
    pass


@cli.command()
def init():
    """Initialize the development environment."""
    click.echo("🚀 Initializing XORB development environment...")
    return run_make_target("doctor")


@cli.command()
@click.option("--coverage", is_flag=True, help="Run tests with coverage")
def test_fast(coverage: bool):
    """Run fast unit tests without coverage."""
    if coverage:
        click.echo("🧪 Running tests with coverage...")
        return run_make_target("test")
    else:
        click.echo("🧪 Running fast unit tests...")
        return run_make_target("test-fast")


@cli.command()
def ci_fast():
    """Run fast CI checks (lint + test-fast)."""
    click.echo("🔍 Running fast CI checks...")

    # Run lint first
    click.echo("Step 1: Linting...")
    lint_result = run_make_target("lint")
    if lint_result != 0:
        click.echo("❌ Linting failed", err=True)
        return lint_result

    # Run fast tests
    click.echo("Step 2: Fast tests...")
    test_result = run_make_target("test-fast")
    if test_result != 0:
        click.echo("❌ Tests failed", err=True)
        return test_result

    click.echo("✅ Fast CI checks completed successfully")
    return 0


@cli.command()
def control_plane_status():
    """Check control plane status and health."""
    click.echo("🎯 Checking XORB control plane status...")

    repo_root = get_repo_root()

    # Check if services are running
    services = [
        ("API Server", "http://localhost:8000/api/v1/health"),
        ("Orchestrator", "http://localhost:8000/api/v1/orchestrator/health"),
        ("NATS Server", "nats://localhost:4222")
    ]

    click.echo("🔍 Service Health Check:")
    click.echo("=" * 50)

    all_healthy = True

    for service_name, endpoint in services:
        try:
            if endpoint.startswith("http"):
                import requests
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    click.echo(f"✅ {service_name}: Healthy")
                else:
                    click.echo(f"❌ {service_name}: Unhealthy (HTTP {response.status_code})")
                    all_healthy = False
            else:
                # For NATS, we'll just check if the process is running
                result = subprocess.run(
                    ["pgrep", "-f", "nats-server"],
                    capture_output=True
                )
                if result.returncode == 0:
                    click.echo(f"✅ {service_name}: Running")
                else:
                    click.echo(f"❌ {service_name}: Not running")
                    all_healthy = False

        except Exception as e:
            click.echo(f"❌ {service_name}: Error - {e}")
            all_healthy = False

    click.echo("=" * 50)

    if all_healthy:
        click.echo("✅ All services are healthy")
        return 0
    else:
        click.echo("❌ Some services are unhealthy")
        click.echo("\n💡 To start services:")
        click.echo("   make up    # Start with Docker Compose")
        click.echo("   make api   # Start API server locally")
        return 1


@cli.command()
def security_scan():
    """Run comprehensive security scan."""
    click.echo("🔒 Running comprehensive security scan...")
    return run_make_target("security-scan")


@cli.command()
def doctor():
    """Run repository health checks."""
    click.echo("🏥 Running repository doctor...")
    return run_make_target("doctor")


@cli.command()
def up():
    """Start development services with Docker Compose."""
    click.echo("🐳 Starting development services...")
    return run_make_target("up")


@cli.command()
def down():
    """Stop development services."""
    click.echo("🛑 Stopping development services...")
    return run_make_target("down")


@cli.command()
@click.argument("target")
def make(target: str):
    """Run arbitrary make target."""
    click.echo(f"🔧 Running make target: {target}")
    return run_make_target(target)


@cli.command()
def status():
    """Show overall XORB platform status."""
    click.echo("📊 XORB Platform Status")
    click.echo("=" * 60)

    # Run control plane status
    cp_result = control_plane_status.callback()

    click.echo("\n🔒 Security Status:")
    click.echo("-" * 30)

    # Quick security scan
    sec_result = subprocess.run(
        ["python3", "tools/security/security_scan.py", "--format", "text"],
        cwd=get_repo_root(),
        capture_output=True,
        text=True
    )

    if sec_result.returncode == 0:
        # Extract just the summary
        lines = sec_result.stdout.split('\n')
        summary_started = False
        for line in lines:
            if line.startswith("Overall Status:"):
                summary_started = True
            if summary_started and line.strip():
                click.echo(line)
            if summary_started and line.startswith("="):
                break
    else:
        click.echo("❌ Security scan failed")

    click.echo("\n💡 Quick Commands:")
    click.echo("   xorbctl init          # Initialize environment")
    click.echo("   xorbctl up            # Start services")
    click.echo("   xorbctl test-fast     # Run tests")
    click.echo("   xorbctl ci-fast       # Run CI checks")
    click.echo("   xorbctl security-scan # Full security scan")

    return cp_result


@cli.group()
def ops():
    """Operations Pack (v2025.08-rc1) commands."""
    pass


@ops.command()
def status():
    """Show operations pack status and runbook locations."""
    click.echo("📚 XORB Operations Pack v2025.08-rc1 Status")
    click.echo("=" * 60)

    # Check runbook files exist
    repo_root = get_repo_root()
    runbooks = [
        ("Incident Response Runbook", "RUNBOOK_INCIDENT_RESPONSE.md"),
        ("Rollback Runbook", "RUNBOOK_ROLLBACK.md"),
        ("Chaos Engineering Drills", "docs/CHAOS_DRILLS.md"),
        ("Release Confidence Report", "docs/RELEASE_CONFIDENCE_REPORT.md")
    ]

    click.echo("📋 Runbook Availability:")
    all_present = True

    for name, path in runbooks:
        file_path = repo_root / path
        if file_path.exists():
            click.echo(f"✅ {name}: {path}")
        else:
            click.echo(f"❌ {name}: {path} (missing)")
            all_present = False

    click.echo("\n🔍 Monitoring Integration:")

    # Check Prometheus rules
    prometheus_rules = repo_root / "infra/monitoring/prometheus/prometheus-rules.yml"
    if prometheus_rules.exists():
        click.echo("✅ Prometheus Rules: infra/monitoring/prometheus/prometheus-rules.yml")
    else:
        click.echo("❌ Prometheus Rules: Missing")
        all_present = False

    # Check Grafana dashboard
    grafana_dashboard = repo_root / "infra/monitoring/grafana/dashboards/xorb-release-slo-dashboard.json"
    if grafana_dashboard.exists():
        click.echo("✅ Grafana Dashboard: infra/monitoring/grafana/dashboards/xorb-release-slo-dashboard.json")
    else:
        click.echo("❌ Grafana Dashboard: Missing")
        all_present = False

    click.echo(f"\n📊 Overall Status: {'✅ Complete' if all_present else '❌ Incomplete'}")

    click.echo("\n💡 Operations Commands:")
    click.echo("   xorbctl ops status        # Show operations pack status")
    click.echo("   make ops-runbooks         # Show runbook locations")
    click.echo("   make ops-alerts-validate  # Validate alert rules")
    click.echo("   make chaos-dry-run        # Preview chaos experiments")

    return 0 if all_present else 1


@cli.group()
def chaos():
    """Chaos engineering commands."""
    pass


@chaos.command()
def list():
    """List available chaos experiments."""
    click.echo("🧪 XORB Chaos Engineering Experiments")
    click.echo("=" * 50)

    experiments = [
        ("nats-node-kill", "NATS Node Kill", "10 min", "Validate cluster resilience and message delivery SLO compliance"),
        ("replay-storm", "Replay Storm Injection", "15 min", "Test traffic isolation under 10x replay load with auto-remediation"),
        ("evidence-corruption", "Evidence Corruption", "12 min", "Validate evidence integrity under malicious injection")
    ]

    for exp_id, name, duration, description in experiments:
        click.echo(f"\n🔬 {name} ({exp_id})")
        click.echo(f"   Duration: {duration}")
        click.echo(f"   Objective: {description}")

    click.echo("\n💡 Chaos Commands:")
    click.echo("   xorbctl chaos list        # List available experiments")
    click.echo("   make chaos-dry-run        # Preview experiment steps")
    click.echo("   See docs/CHAOS_DRILLS.md  # Full execution guide")

    return 0


if __name__ == "__main__":
    # Handle the case where this is run directly
    sys.exit(cli())
