#!/usr/bin/env python3
"""
XORB Bootstrap Dashboard

Real-time monitoring dashboard for XORB auto-configuration and deployment process.
Provides visual feedback during system detection, configuration, and deployment.

Features:
- Real-time configuration progress
- System capability visualization
- Deployment status monitoring
- Interactive configuration adjustment
- Post-deployment health checks

Author: XORB DevOps AI
Version: 2.0.0
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import rich
    from rich.align import Align
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class ConfigurationStep:
    """Configuration step tracking"""
    name: str
    description: str
    completed: bool = False
    error: str | None = None
    duration: float = 0.0


class XorbBootstrapDashboard:
    """
    Interactive dashboard for XORB bootstrap process monitoring.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.console = Console() if RICH_AVAILABLE else None
        self.configuration_steps = [
            ConfigurationStep("detect_system", "Detecting system capabilities"),
            ConfigurationStep("classify_profile", "Classifying system profile"),
            ConfigurationStep("generate_config", "Generating XORB configuration"),
            ConfigurationStep("write_env", "Writing environment file"),
            ConfigurationStep("select_compose", "Selecting Docker Compose configuration"),
            ConfigurationStep("validate_config", "Validating configuration"),
            ConfigurationStep("generate_report", "Generating bootstrap report")
        ]
        self.system_info = {}
        self.configuration = {}
        self.deployment_status = {}

    def load_bootstrap_report(self) -> dict[str, Any] | None:
        """Load bootstrap report if available"""
        report_file = self.project_root / "logs" / "bootstrap_report.json"

        if report_file.exists():
            try:
                with open(report_file) as f:
                    return json.load(f)
            except Exception as e:
                if self.console:
                    self.console.print(f"[red]Error loading bootstrap report: {e}[/red]")
                return None
        return None

    def create_system_info_table(self, report: dict[str, Any]) -> Table:
        """Create system information table"""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="🖥️  System Information", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value", style="yellow", width=30)
        table.add_column("Status", style="green", width=10)

        capabilities = report.get("system_capabilities", {})

        # System details
        table.add_row("OS Type", capabilities.get("os_type", "Unknown"), "✅")
        table.add_row("Architecture", capabilities.get("architecture", "Unknown"), "✅")
        table.add_row("CPU Cores", str(capabilities.get("cpu_cores", 0)),
                     "✅" if capabilities.get("cpu_cores", 0) >= 2 else "⚠️")
        table.add_row("RAM (GB)", f"{capabilities.get('ram_gb', 0):.1f}",
                     "✅" if capabilities.get("ram_gb", 0) >= 4 else "⚠️")
        table.add_row("System Profile", capabilities.get("profile", "Unknown"), "✅")
        table.add_row("Docker Version", capabilities.get("docker_version", "Unknown"),
                     "✅" if capabilities.get("docker_version") != "unknown" else "❌")
        table.add_row("BuildKit", "Enabled" if capabilities.get("docker_buildkit") else "Disabled",
                     "✅" if capabilities.get("docker_buildkit") else "⚠️")

        return table

    def create_configuration_table(self, report: dict[str, Any]) -> Table:
        """Create configuration table"""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="⚙️  XORB Configuration", show_header=True, header_style="bold green")
        table.add_column("Setting", style="cyan", width=25)
        table.add_column("Value", style="yellow", width=20)
        table.add_column("Optimization", style="green", width=25)

        config = report.get("generated_configuration", {})

        table.add_row("Mode", config.get("mode", "Unknown"), "Auto-selected based on hardware")
        table.add_row("System Profile", config.get("system_profile", "Unknown"), "Hardware classification")
        table.add_row("Agent Concurrency", str(config.get("agent_concurrency", 0)), "CPU-optimized")
        table.add_row("Max Missions", str(config.get("max_concurrent_missions", 0)), "Resource-based limit")
        table.add_row("Worker Threads", str(config.get("worker_threads", 0)), "Threading optimization")
        table.add_row("Monitoring", "Enabled" if config.get("monitoring_enabled") else "Disabled",
                     "Resource-based decision")
        table.add_row("Memory Limit (MB)", str(config.get("memory_limit_mb", 0)), "80% of available RAM")
        table.add_row("CPU Limit", str(config.get("cpu_limit", 0)), "Conservative allocation")

        return table

    def create_services_table(self, report: dict[str, Any]) -> Table:
        """Create enabled services table"""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="🔧 Enabled Services", show_header=True, header_style="bold blue")
        table.add_column("Service", style="cyan", width=15)
        table.add_column("Type", style="yellow", width=15)
        table.add_column("Resource Allocation", style="green", width=30)

        services = report.get("generated_configuration", {}).get("services_enabled", [])
        resource_limits = report.get("generated_configuration", {}).get("resource_limits", {})

        service_types = {
            "postgres": "Database",
            "redis": "Cache",
            "temporal": "Workflow",
            "nats": "Messaging",
            "neo4j": "Graph DB",
            "qdrant": "Vector DB",
            "api": "Core Service",
            "worker": "Core Service",
            "orchestrator": "Core Service",
            "scanner-go": "Scanner",
            "prometheus": "Monitoring",
            "grafana": "Monitoring",
            "tempo": "Tracing",
            "alertmanager": "Alerting"
        }

        for service in services:
            service_type = service_types.get(service, "Unknown")

            if service in resource_limits:
                limits = resource_limits[service].get("deploy", {}).get("resources", {}).get("limits", {})
                memory = limits.get("memory", "Auto")
                cpu = limits.get("cpus", "Auto")
                allocation = f"Memory: {memory}, CPU: {cpu}"
            else:
                allocation = "Auto-allocated"

            table.add_row(service, service_type, allocation)

        return table

    def create_readiness_panel(self, report: dict[str, Any]) -> Panel:
        """Create deployment readiness panel"""
        if not RICH_AVAILABLE:
            return None

        readiness = report.get("deployment_readiness", {})

        status_text = []
        all_ready = True

        checks = [
            ("Docker Available", readiness.get("docker_available", False)),
            ("BuildKit Enabled", readiness.get("buildkit_enabled", False)),
            ("Compose Available", readiness.get("compose_available", False)),
            ("Sufficient Resources", readiness.get("sufficient_resources", False)),
            ("Disk Space Available", readiness.get("disk_space_available", False))
        ]

        for check_name, status in checks:
            icon = "✅" if status else "❌"
            color = "green" if status else "red"
            status_text.append(f"[{color}]{icon} {check_name}[/{color}]")
            if not status:
                all_ready = False

        overall_status = "🚀 Ready for Deployment" if all_ready else "⚠️  Issues Detected"
        status_color = "green" if all_ready else "red"

        content = f"[bold {status_color}]{overall_status}[/bold {status_color}]\n\n" + "\n".join(status_text)

        return Panel(content, title="🎯 Deployment Readiness", border_style="green" if all_ready else "red")

    def create_recommendations_panel(self, report: dict[str, Any]) -> Panel:
        """Create recommendations panel"""
        if not RICH_AVAILABLE:
            return None

        recommendations = report.get("recommendations", [])

        if not recommendations:
            content = "[green]✅ No recommendations - system optimally configured![/green]"
        else:
            content = "\n".join([f"• {rec}" for rec in recommendations])

        return Panel(content, title="💡 Recommendations", border_style="yellow")

    def run_interactive_dashboard(self):
        """Run interactive dashboard"""
        if not RICH_AVAILABLE:
            self.run_text_dashboard()
            return

        # Try to load existing report
        report = self.load_bootstrap_report()

        if not report:
            self.console.print("[red]No bootstrap report found. Run autoconfigure.py first.[/red]")
            return

        # Create layout
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=10)
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(name="system"),
            Layout(name="config")
        )

        layout["right"].split_column(
            Layout(name="services"),
            Layout(name="status")
        )

        # Create content
        header = Panel(
            Align.center(
                Text("🚀 XORB Bootstrap Dashboard", style="bold blue"),
                vertical="middle"
            ),
            style="blue"
        )

        system_table = self.create_system_info_table(report)
        config_table = self.create_configuration_table(report)
        services_table = self.create_services_table(report)
        readiness_panel = self.create_readiness_panel(report)
        recommendations_panel = self.create_recommendations_panel(report)

        # Update layout
        layout["header"].update(header)
        layout["system"].update(system_table)
        layout["config"].update(config_table)
        layout["services"].update(services_table)
        layout["status"].update(readiness_panel)
        layout["footer"].update(recommendations_panel)

        # Display dashboard
        with Live(layout, refresh_per_second=1, screen=True):
            self.console.print("\n[bold green]Dashboard running... Press Ctrl+C to exit[/bold green]")

            try:
                while True:
                    # Check for updated report
                    updated_report = self.load_bootstrap_report()
                    if updated_report and updated_report != report:
                        report = updated_report
                        # Update tables (would need to rebuild)

                    time.sleep(1)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Dashboard closed.[/yellow]")

    def run_text_dashboard(self):
        """Run text-based dashboard for systems without rich"""
        print("\n" + "="*80)
        print("🚀 XORB Bootstrap Dashboard (Text Mode)")
        print("="*80)

        report = self.load_bootstrap_report()

        if not report:
            print("❌ No bootstrap report found. Run autoconfigure.py first.")
            return

        # System Information
        print("\n🖥️  SYSTEM INFORMATION:")
        capabilities = report.get("system_capabilities", {})
        print(f"   OS: {capabilities.get('os_type', 'Unknown')} {capabilities.get('architecture', '')}")
        print(f"   CPU: {capabilities.get('cpu_cores', 0)} cores")
        print(f"   RAM: {capabilities.get('ram_gb', 0):.1f}GB")
        print(f"   Profile: {capabilities.get('profile', 'Unknown')}")
        print(f"   Docker: {capabilities.get('docker_version', 'Unknown')}")

        # Configuration
        print("\n⚙️  XORB CONFIGURATION:")
        config = report.get("generated_configuration", {})
        print(f"   Mode: {config.get('mode', 'Unknown')}")
        print(f"   Agent Concurrency: {config.get('agent_concurrency', 0)}")
        print(f"   Max Missions: {config.get('max_concurrent_missions', 0)}")
        print(f"   Monitoring: {'Enabled' if config.get('monitoring_enabled') else 'Disabled'}")

        # Services
        print("\n🔧 ENABLED SERVICES:")
        services = config.get("services_enabled", [])
        for i, service in enumerate(services, 1):
            print(f"   {i:2d}. {service}")

        # Readiness
        print("\n🎯 DEPLOYMENT READINESS:")
        readiness = report.get("deployment_readiness", {})
        for check, status in readiness.items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {check.replace('_', ' ').title()}")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        print("\n" + "="*80)

    def monitor_deployment(self, compose_file: str = "docker-compose.auto.yml"):
        """Monitor deployment progress"""
        if not RICH_AVAILABLE:
            self.monitor_deployment_text(compose_file)
            return

        self.console.print(f"[bold blue]Monitoring deployment: {compose_file}[/bold blue]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:

            task = progress.add_task("Deploying XORB services...", total=100)

            try:
                # Start deployment
                cmd = f"docker compose -f {compose_file} --env-file .xorb.env up -d"
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Monitor progress
                for i in range(100):
                    time.sleep(0.5)
                    progress.update(task, advance=1)

                    # Check if process completed
                    if process.poll() is not None:
                        break

                # Get result
                stdout, stderr = process.communicate()

                if process.returncode == 0:
                    self.console.print("[green]✅ Deployment completed successfully![/green]")
                    self.show_deployment_urls()
                else:
                    self.console.print(f"[red]❌ Deployment failed: {stderr}[/red]")

            except Exception as e:
                self.console.print(f"[red]Error monitoring deployment: {e}[/red]")

    def monitor_deployment_text(self, compose_file: str):
        """Monitor deployment in text mode"""
        print(f"\n🚀 Monitoring deployment: {compose_file}")
        print("Please wait while services start...")

        try:
            cmd = f"docker compose -f {compose_file} --env-file .xorb.env up -d"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Deployment completed successfully!")
                self.show_deployment_urls_text()
            else:
                print(f"❌ Deployment failed: {result.stderr}")

        except Exception as e:
            print(f"Error: {e}")

    def show_deployment_urls(self):
        """Show deployment URLs with rich formatting"""
        if not RICH_AVAILABLE:
            self.show_deployment_urls_text()
            return

        table = Table(title="🌐 Service URLs", show_header=True, header_style="bold green")
        table.add_column("Service", style="cyan")
        table.add_column("URL", style="yellow")
        table.add_column("Description", style="white")

        urls = [
            ("API", "http://localhost:8000", "XORB REST API"),
            ("Worker Metrics", "http://localhost:9000/metrics", "Worker performance metrics"),
            ("Orchestrator", "http://localhost:8080", "Mission orchestration"),
            ("Prometheus", "http://localhost:9090", "Metrics collection"),
            ("Grafana", "http://localhost:3000", "Monitoring dashboard"),
            ("Temporal UI", "http://localhost:8233", "Workflow management"),
        ]

        for service, url, description in urls:
            table.add_row(service, url, description)

        self.console.print(table)

    def show_deployment_urls_text(self):
        """Show deployment URLs in text mode"""
        print("\n🌐 SERVICE URLS:")
        print("   API:              http://localhost:8000")
        print("   Worker Metrics:   http://localhost:9001/metrics")
        print("   Orchestrator:     http://localhost:8080")
        print("   Prometheus:       http://localhost:9090")
        print("   Grafana:          http://localhost:3000 (admin/xorb_admin_2024)")
        print("   Temporal UI:      http://localhost:8233")


def main():
    """Main entry point"""
    dashboard = XorbBootstrapDashboard()

    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            compose_file = sys.argv[2] if len(sys.argv) > 2 else "docker-compose.auto.yml"
            dashboard.monitor_deployment(compose_file)
        elif sys.argv[1] == "report":
            dashboard.run_interactive_dashboard()
        else:
            print("Usage: python bootstrap_dashboard.py [report|monitor] [compose_file]")
    else:
        dashboard.run_interactive_dashboard()


if __name__ == "__main__":
    main()
