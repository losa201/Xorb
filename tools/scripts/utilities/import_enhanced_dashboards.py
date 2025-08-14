#!/usr/bin/env python3

import requests
import json
import time
import os
from datetime import datetime

class XORBDashboardImporter:
    def __init__(self, grafana_url="http://localhost:3002", username="admin", password="xorb_rl_admin_2025"):
        self.grafana_url = grafana_url
        self.auth = (username, password)
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    def import_dashboard(self, dashboard_file, dashboard_name):
        """Import a single dashboard from JSON file"""
        print(f"📊 Importing {dashboard_name}...")

        try:
            # Read dashboard JSON
            with open(dashboard_file, 'r') as f:
                dashboard_json = json.load(f)

            # Prepare import payload
            import_payload = {
                "dashboard": dashboard_json,
                "overwrite": True,
                "inputs": [],
                "folderId": 0
            }

            # Import dashboard
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=import_payload,
                headers=self.headers,
                auth=self.auth
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Successfully imported {dashboard_name}")
                print(f"   Dashboard ID: {result.get('id', 'N/A')}")
                print(f"   Dashboard URL: {result.get('url', 'N/A')}")
                return True
            else:
                print(f"❌ Failed to import {dashboard_name}: {response.status_code}")
                print(f"   Error: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Error importing {dashboard_name}: {e}")
            return False

    def setup_datasource(self):
        """Ensure Prometheus datasource is configured"""
        print("🔧 Setting up Prometheus datasource...")

        datasource_config = {
            "name": "Prometheus",
            "type": "prometheus",
            "url": "http://localhost:9090",
            "access": "proxy",
            "isDefault": True
        }

        try:
            response = requests.post(
                f"{self.grafana_url}/api/datasources",
                json=datasource_config,
                headers=self.headers,
                auth=self.auth
            )

            if response.status_code in [200, 409]:  # 409 = already exists
                print("✅ Prometheus datasource configured")
                return True
            else:
                print(f"⚠️  Datasource setup warning: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Error setting up datasource: {e}")
            return False

    def import_all_dashboards(self):
        """Import all XORB enhanced dashboards"""
        print("🎯 XORB ENHANCED DASHBOARD IMPORT")
        print("=" * 50)
        print(f"📅 Import Time: {datetime.now().isoformat()}")
        print(f"🌐 Grafana URL: {self.grafana_url}")
        print()

        # Setup datasource
        self.setup_datasource()
        print()

        # Dashboard configurations
        dashboards = [
            {
                "file": "/root/Xorb/config/xorb_comprehensive_telemetry_dashboard.json",
                "name": "🎯 Comprehensive Telemetry Dashboard",
                "description": "Complete system telemetry and metrics overview"
            },
            {
                "file": "/root/Xorb/config/xorb_rl_agent_performance_dashboard.json",
                "name": "🤖 RL Agent Performance Dashboard",
                "description": "Reinforcement learning agent metrics and performance tracking"
            },
            {
                "file": "/root/Xorb/config/xorb_strategic_coordination_dashboard.json",
                "name": "🎯 Strategic Coordination Dashboard",
                "description": "Strategic coordination patterns and effectiveness visualization"
            },
            {
                "file": "/root/Xorb/config/xorb_service_health_monitoring_dashboard.json",
                "name": "🚀 Service Health Monitoring Dashboard",
                "description": "Service uptime, health status, and resource monitoring"
            },
            {
                "file": "/root/Xorb/config/xorb_historical_analysis_dashboard.json",
                "name": "📈 Historical Analysis Dashboard",
                "description": "Historical data patterns and success trend analysis"
            }
        ]

        # Import each dashboard
        successful_imports = 0
        for dashboard in dashboards:
            print(f"📋 {dashboard['name']}")
            print(f"   Description: {dashboard['description']}")

            if os.path.exists(dashboard['file']):
                if self.import_dashboard(dashboard['file'], dashboard['name']):
                    successful_imports += 1
                    time.sleep(1)  # Brief pause between imports
            else:
                print(f"❌ Dashboard file not found: {dashboard['file']}")

            print()

        print("🎊 DASHBOARD IMPORT COMPLETE")
        print("=" * 50)
        print(f"✅ Successfully imported: {successful_imports}/{len(dashboards)} dashboards")
        print(f"🌐 Access your dashboards at: {self.grafana_url}")
        print(f"🔑 Login credentials:")
        print(f"   Username: admin")
        print(f"   Password: xorb_rl_admin_2025")
        print()

        print("📊 AVAILABLE DASHBOARDS:")
        for i, dashboard in enumerate(dashboards, 1):
            status = "✅" if i <= successful_imports else "❌"
            print(f"  {status} {dashboard['name']}")

        print()
        print("🎯 DASHBOARD FEATURES:")
        print("  • 🧠 Real-time RL agent performance tracking")
        print("  • ⚡ Strategic coordination pattern visualization")
        print("  • 📊 Comprehensive system telemetry")
        print("  • 🚀 Service health and uptime monitoring")
        print("  • 📈 Historical data analysis and trends")
        print("  • 🎨 Dark theme with interactive visualizations")
        print("  • 🔄 Auto-refresh every 5-10 seconds")

        return successful_imports == len(dashboards)

def main():
    """Main execution function"""
    importer = XORBDashboardImporter()

    # Wait for Grafana to be ready
    print("⏳ Waiting for Grafana to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{importer.grafana_url}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Grafana is ready!")
                break
        except:
            pass

        if i < max_retries - 1:
            print(f"   Retry {i+1}/{max_retries}...")
            time.sleep(2)
        else:
            print("❌ Grafana not ready, proceeding anyway...")

    print()

    # Import all dashboards
    success = importer.import_all_dashboards()

    if success:
        print("🎉 ALL DASHBOARDS IMPORTED SUCCESSFULLY!")
        print()
        print("🚀 NEXT STEPS:")
        print("1. Open Grafana: http://localhost:3002")
        print("2. Login with admin / xorb_rl_admin_2025")
        print("3. Navigate to Dashboards to view your enhanced telemetry")
        print("4. Each dashboard auto-refreshes with live data")
        print()
        print("📊 DASHBOARD HIGHLIGHTS:")
        print("• Comprehensive system telemetry with memory, GC, and runtime metrics")
        print("• RL agent performance tracking with 60 agents across 4 clusters")
        print("• Strategic coordination patterns with 95%+ efficiency visualization")
        print("• Service health monitoring with uptime and resource usage")
        print("• Historical analysis with 23 learned patterns and trends")
    else:
        print("⚠️  Some dashboards failed to import. Check the logs above.")

    return success

if __name__ == "__main__":
    main()
