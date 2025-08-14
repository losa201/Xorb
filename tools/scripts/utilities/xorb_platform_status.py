#!/usr/bin/env python3
"""
XORB Platform Status Check Script
Validates and reports the status of all XORB services
"""

import requests
import json
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print status banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🔍 XORB PLATFORM STATUS CHECK                  ║
║                     Version 2.0.0                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_service_health(service_name, url, port):
    """Check health of a service"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return "✅ HEALTHY", response.json() if response.headers.get('content-type', '').startswith('application/json') else "OK"
        else:
            return f"⚠️ UNHEALTHY ({response.status_code})", None
    except requests.exceptions.ConnectionError:
        return "❌ UNREACHABLE", None
    except requests.exceptions.Timeout:
        return "⏰ TIMEOUT", None
    except Exception as e:
        return f"❌ ERROR", str(e)

def check_docker_service(container_name):
    """Check Docker container status"""
    try:
        result = subprocess.run(['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Status}}'],
                               capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            status = result.stdout.strip()
            if "Up" in status:
                return "✅ RUNNING", status
            else:
                return "⚠️ NOT RUNNING", status
        else:
            return "❌ NOT FOUND", None
    except Exception as e:
        return "❌ ERROR", str(e)

def main():
    """Main status check function"""
    print_banner()
    print(f"🕐 Status Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Core XORB Services
    print("🧠 CORE XORB SERVICES")
    print("=" * 60)

    services = [
        ("Neural Orchestrator", "http://localhost:8003/health", 8003),
        ("Learning Service", "http://localhost:8004/health", 8004),
        ("Threat Detection", "http://localhost:8005/health", 8005),
    ]

    for service_name, health_url, port in services:
        status, data = check_service_health(service_name, health_url, port)
        print(f"   {status:<15} {service_name:<20} (Port {port})")
        if data and isinstance(data, dict):
            # Show key health metrics
            if 'status' in data:
                print(f"     Status: {data.get('status', 'unknown')}")
            if 'active_agents' in data:
                print(f"     Active Agents: {data.get('active_agents', 0)}")
            if 'learning_active' in data:
                print(f"     Learning Active: {data.get('learning_active', False)}")

    print()

    # Infrastructure Services
    print("🏗️ INFRASTRUCTURE SERVICES")
    print("=" * 60)

    infra_services = [
        ("PostgreSQL", "xorb_production_postgres_1", 5432),
        ("Redis", "xorb-redis", 6379),
        ("Neo4j", "xorb-neo4j", 7474),
    ]

    for service_name, container_name, port in infra_services:
        status, data = check_docker_service(container_name)
        print(f"   {status:<15} {service_name:<20} (Port {port})")
        if data:
            print(f"     Container Status: {data}")

    print()

    # Monitoring Services
    print("📊 MONITORING & OBSERVABILITY")
    print("=" * 60)

    monitoring_services = [
        ("Prometheus", "http://localhost:9090/-/healthy", 9090),
        ("Grafana", "http://localhost:3000/api/health", 3000),
    ]

    for service_name, health_url, port in monitoring_services:
        status, data = check_service_health(service_name, health_url, port)
        print(f"   {status:<15} {service_name:<20} (Port {port})")

    print()

    # Network Status
    print("🌐 NETWORK STATUS")
    print("=" * 60)

    try:
        result = subprocess.run(['docker', 'network', 'ls', '--filter', 'name=xorb'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            networks = [line for line in result.stdout.split('\n') if 'xorb' in line.lower()]
            print(f"   ✅ CONFIGURED   Docker Networks ({len(networks)} found)")
            for network in networks:
                if network.strip():
                    parts = network.split()
                    if len(parts) >= 2:
                        print(f"     - {parts[1]} ({parts[2]})")
        else:
            print("   ❌ ERROR       Failed to check Docker networks")
    except Exception as e:
        print(f"   ❌ ERROR       Network check failed: {e}")

    print()

    # Access Points
    print("🔗 ACCESS POINTS")
    print("=" * 60)
    print("   📊 Grafana Dashboard:     http://localhost:3000")
    print("   📈 Prometheus Metrics:    http://localhost:9090")
    print("   🧠 Neural Orchestrator:   http://localhost:8003")
    print("   🎓 Learning Service:      http://localhost:8004")
    print("   🛡️  Threat Detection:     http://localhost:8005")
    print("   🗄️  Neo4j Browser:        http://localhost:7474")

    print()

    # Overall Status
    print("🏆 OVERALL PLATFORM STATUS")
    print("=" * 60)

    # Count healthy services
    core_healthy = 0
    for service_name, health_url, port in services:
        status, _ = check_service_health(service_name, health_url, port)
        if "HEALTHY" in status:
            core_healthy += 1

    if core_healthy == len(services):
        print("   🎉 EXCELLENT    All core services are healthy and operational!")
        print("   🚀 READY        XORB Platform is ready for production use")
    elif core_healthy >= len(services) * 0.7:
        print("   ⚠️  GOOD        Most core services are healthy")
        print("   🔧 ACTION       Some services may need attention")
    else:
        print("   ❌ POOR         Multiple services are unhealthy")
        print("   🚨 CRITICAL     Platform requires immediate attention")

    print(f"   📊 HEALTH       {core_healthy}/{len(services)} core services healthy")

    print()
    print("✅ Status check completed!")
    print(f"🕐 Check Duration: {time.time():.1f} seconds")

if __name__ == "__main__":
    main()
