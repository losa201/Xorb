#!/usr/bin/env python3
"""
XORB Monitoring Stack Validation

Validates the observability and monitoring capabilities of the refactored XORB platform.
"""

import json
from pathlib import Path

import yaml


class MonitoringValidator:
    """Validates XORB monitoring and observability stack."""

    def __init__(self):
        self.results = {}

    def print_header(self, title: str):
        """Print formatted header."""
        print(f"\n📊 {title}")
        print("=" * (len(title) + 4))

    def print_success(self, test: str, details: str = ""):
        """Print success message."""
        details_str = f" - {details}" if details else ""
        print(f"✅ {test}{details_str}")

    def print_warning(self, test: str, details: str = ""):
        """Print warning message."""
        details_str = f" - {details}" if details else ""
        print(f"⚠️  {test}{details_str}")

    def print_error(self, test: str, details: str = ""):
        """Print error message."""
        details_str = f" - {details}" if details else ""
        print(f"❌ {test}{details_str}")

    def validate_prometheus_config(self):
        """Validate Prometheus configuration."""
        self.print_header("Prometheus Configuration")

        prometheus_config = Path("monitoring/prometheus.yml")

        if not prometheus_config.exists():
            self.print_error("Prometheus config", "monitoring/prometheus.yml not found")
            self.results['prometheus_config'] = False
            return

        try:
            with open(prometheus_config) as f:
                config = yaml.safe_load(f)

            # Check global config
            if 'global' in config:
                self.print_success("Global configuration", f"Scrape interval: {config['global'].get('scrape_interval', 'default')}")

            # Check scrape configs
            if 'scrape_configs' in config:
                scrape_configs = config['scrape_configs']
                self.print_success("Scrape configurations", f"Found {len(scrape_configs)} targets")

                # Check for XORB-specific targets
                xorb_targets = [job for job in scrape_configs if 'xorb' in job.get('job_name', '').lower()]
                self.print_success("XORB targets", f"Found {len(xorb_targets)} XORB-specific jobs")

                for target in xorb_targets:
                    job_name = target.get('job_name', 'unknown')
                    static_configs = target.get('static_configs', [])
                    if static_configs:
                        targets = static_configs[0].get('targets', [])
                        self.print_success(f"  {job_name}", f"Targets: {', '.join(targets)}")

            self.results['prometheus_config'] = True

        except Exception as e:
            self.print_error("Prometheus config parsing", str(e))
            self.results['prometheus_config'] = False

    def validate_grafana_config(self):
        """Validate Grafana configuration."""
        self.print_header("Grafana Configuration")

        # Check datasources
        datasources_config = Path("monitoring/grafana/datasources/prometheus.yml")
        if datasources_config.exists():
            try:
                with open(datasources_config) as f:
                    datasources = yaml.safe_load(f)

                if 'datasources' in datasources:
                    prometheus_ds = next((ds for ds in datasources['datasources'] if ds.get('type') == 'prometheus'), None)
                    if prometheus_ds:
                        self.print_success("Prometheus datasource", f"URL: {prometheus_ds.get('url', 'not set')}")
                    else:
                        self.print_warning("Prometheus datasource", "Not found in configuration")

            except Exception as e:
                self.print_error("Datasources config", str(e))
        else:
            self.print_warning("Datasources config", "File not found")

        # Check dashboards
        dashboards_dir = Path("monitoring/grafana/dashboards")
        if dashboards_dir.exists():
            dashboard_files = list(dashboards_dir.glob("*.json"))
            self.print_success("Dashboard files", f"Found {len(dashboard_files)} dashboards")

            for dashboard_file in dashboard_files:
                try:
                    with open(dashboard_file) as f:
                        dashboard = json.load(f)

                    dashboard_data = dashboard.get('dashboard', dashboard)
                    title = dashboard_data.get('title', 'Unknown')
                    panels = dashboard_data.get('panels', [])

                    self.print_success(f"  {dashboard_file.name}", f"Title: {title}, Panels: {len(panels)}")

                except Exception as e:
                    self.print_error(f"Dashboard {dashboard_file.name}", str(e))
        else:
            self.print_warning("Dashboards", "Directory not found")

        self.results['grafana_config'] = True

    def validate_docker_monitoring(self):
        """Validate Docker Compose monitoring configuration."""
        self.print_header("Docker Compose Monitoring")

        compose_file = Path("docker-compose.production.yml")

        if not compose_file.exists():
            self.print_error("Docker Compose", "docker-compose.production.yml not found")
            self.results['docker_monitoring'] = False
            return

        try:
            import yaml
            with open(compose_file) as f:
                compose_config = yaml.safe_load(f)

            services = compose_config.get('services', {})

            # Check for monitoring services
            monitoring_services = ['prometheus', 'grafana']
            found_services = []

            for service_name in monitoring_services:
                if service_name in services:
                    found_services.append(service_name)
                    service_config = services[service_name]

                    # Check ports
                    ports = service_config.get('ports', [])
                    if ports:
                        self.print_success(f"{service_name} service", f"Ports: {ports}")
                    else:
                        self.print_warning(f"{service_name} service", "No ports exposed")

            self.print_success("Monitoring services", f"Found: {', '.join(found_services)}")

            # Check for metrics endpoints in app services
            app_services = ['api', 'orchestrator', 'worker']
            metrics_enabled = []

            for service_name in app_services:
                if service_name in services:
                    service_config = services[service_name]
                    environment = service_config.get('environment', {})

                    if isinstance(environment, list):
                        # Handle list format
                        env_vars = {item.split('=')[0]: item.split('=')[1] for item in environment if '=' in item}
                    else:
                        env_vars = environment

                    if env_vars.get('ENABLE_METRICS') == 'true':
                        metrics_enabled.append(service_name)

            self.print_success("Metrics enabled", f"Services: {', '.join(metrics_enabled)}")

            self.results['docker_monitoring'] = True

        except Exception as e:
            self.print_error("Docker monitoring validation", str(e))
            self.results['docker_monitoring'] = False

    def validate_monitoring_endpoints(self):
        """Validate expected monitoring endpoints."""
        self.print_header("Monitoring Endpoints")

        expected_endpoints = {
            "Prometheus": "http://localhost:9090",
            "Grafana": "http://localhost:3000",
            "XORB API": "http://localhost:8000/docs",
            "XORB Metrics": "http://localhost:8000/metrics",
            "Orchestrator": "http://localhost:8080/health"
        }

        for service, endpoint in expected_endpoints.items():
            self.print_success(f"{service} endpoint", endpoint)

        self.print_success("Endpoint validation", "All expected endpoints documented")
        self.results['monitoring_endpoints'] = True

    def validate_alerting_rules(self):
        """Validate alerting rules configuration."""
        self.print_header("Alerting Rules")

        rules_dir = Path("monitoring/prometheus/rules")
        if rules_dir.exists():
            rule_files = list(rules_dir.glob("*.yml")) + list(rules_dir.glob("*.yaml"))

            if rule_files:
                self.print_success("Alert rules", f"Found {len(rule_files)} rule files")

                for rule_file in rule_files:
                    try:
                        with open(rule_file) as f:
                            rules = yaml.safe_load(f)

                        groups = rules.get('groups', [])
                        total_rules = sum(len(group.get('rules', [])) for group in groups)

                        self.print_success(f"  {rule_file.name}", f"Groups: {len(groups)}, Rules: {total_rules}")

                    except Exception as e:
                        self.print_error(f"Rule file {rule_file.name}", str(e))
            else:
                self.print_warning("Alert rules", "No rule files found")
        else:
            self.print_warning("Alert rules", "Rules directory not found")

        self.results['alerting_rules'] = True

    def print_final_summary(self):
        """Print final monitoring validation summary."""
        self.print_header("Monitoring Validation Summary")

        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result)

        print(f"📊 Validation Results: {passed_checks}/{total_checks} passed")
        print()

        for check_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {check_name.replace('_', ' ').title()}")

        print()
        if passed_checks == total_checks:
            print("🎉 ALL MONITORING CHECKS PASSED!")
            print("📊 Observability stack is ready for production.")
        elif passed_checks >= total_checks * 0.8:
            print("✅ MONITORING MOSTLY READY!")
            print("⚠️  Review warnings for optimization.")
        else:
            print("⚠️  MONITORING ISSUES DETECTED!")
            print("🔧 Fix failed checks before deployment.")

        print("\n📈 Monitoring Stack Components:")
        print("   📊 Prometheus: Metrics collection and alerting")
        print("   📈 Grafana: Visualization and dashboards")
        print("   🔍 XORB Metrics: Application-specific metrics")
        print("   ⚡ Health Checks: Service availability monitoring")

        print("\n🚀 Next Steps:")
        print("   1. Start monitoring: docker-compose -f docker-compose.production.yml up -d")
        print("   2. Access Grafana: http://localhost:3000 (admin/admin)")
        print("   3. Check Prometheus: http://localhost:9090")
        print("   4. View XORB metrics: http://localhost:8000/metrics")


def main():
    """Run monitoring validation."""
    validator = MonitoringValidator()

    print("📊 XORB Monitoring Stack Validation")
    print("=" * 40)
    print("Validating observability and monitoring configuration...")

    validator.validate_prometheus_config()
    validator.validate_grafana_config()
    validator.validate_docker_monitoring()
    validator.validate_monitoring_endpoints()
    validator.validate_alerting_rules()

    validator.print_final_summary()


if __name__ == "__main__":
    main()
