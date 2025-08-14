#!/usr/bin/env python3
"""
XORB Platform Deployment Validation Script
Comprehensive validation of all deployed services and infrastructure
"""

import json
import time
import asyncio
import requests
import psycopg2
import redis
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple


class XORBDeploymentValidator:
    """Comprehensive XORB platform deployment validator"""

    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "pending",
            "services": {},
            "infrastructure": {},
            "connectivity": {},
            "performance": {},
            "errors": []
        }

    async def validate_full_deployment(self):
        """Run complete deployment validation"""
        print("🚀 Starting XORB Platform Deployment Validation")
        print("=" * 60)

        try:
            # Core Infrastructure
            await self.validate_databases()
            await self.validate_message_queues()
            await self.validate_monitoring()

            # AI Services
            await self.validate_ai_services()

            # API Services
            await self.validate_api_services()

            # Connectivity Tests
            await self.validate_service_connectivity()

            # Performance Tests
            await self.validate_performance()

            # Generate final report
            self.generate_final_report()

        except Exception as e:
            self.validation_results["errors"].append(f"Critical validation error: {str(e)}")
            self.validation_results["overall_status"] = "failed"

    async def validate_databases(self):
        """Validate database services"""
        print("\n📊 Validating Database Services")

        # PostgreSQL validation (using docker exec since external psycopg2 may have connection issues)
        try:
            result = subprocess.run([
                'docker', 'exec', '-e', 'PGPASSWORD=xorb_secure_2024',
                'xorb_production_postgres_1', 'psql', '-h', 'localhost',
                '-U', 'xorb', '-d', 'xorb_ptaas', '-t', '-c', 'SELECT version();'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                version = result.stdout.strip()
                self.validation_results["infrastructure"]["postgresql"] = {
                    "status": "healthy",
                    "version": version,
                    "port": 5432,
                    "database": "xorb_ptaas"
                }
                print("✅ PostgreSQL: Healthy")
            else:
                raise Exception(f"Database query failed: {result.stderr}")


        except Exception as e:
            self.validation_results["infrastructure"]["postgresql"] = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ PostgreSQL: {str(e)}")

        # Redis validation
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            info = r.info()

            self.validation_results["infrastructure"]["redis"] = {
                "status": "healthy",
                "version": info.get("redis_version"),
                "port": 6379,
                "memory_used": info.get("used_memory_human")
            }
            print("✅ Redis: Healthy")

        except Exception as e:
            self.validation_results["infrastructure"]["redis"] = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ Redis: {str(e)}")

        # Neo4j validation
        try:
            response = requests.get("http://localhost:7474", timeout=5)
            if response.status_code == 200:
                self.validation_results["infrastructure"]["neo4j"] = {
                    "status": "healthy",
                    "port": 7474,
                    "bolt_port": 7687
                }
                print("✅ Neo4j: Healthy")
            else:
                raise Exception(f"Neo4j returned status {response.status_code}")

        except Exception as e:
            self.validation_results["infrastructure"]["neo4j"] = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ Neo4j: {str(e)}")

    async def validate_message_queues(self):
        """Validate message queue services"""
        print("\n📨 Validating Message Queue Services")

        # Check if we have NATS or similar running
        try:
            # Look for running message queue containers
            result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}'],
                                  capture_output=True, text=True)

            running_containers = result.stdout

            if 'nats' in running_containers.lower():
                self.validation_results["infrastructure"]["message_queue"] = {
                    "status": "healthy",
                    "type": "NATS"
                }
                print("✅ Message Queue (NATS): Healthy")
            elif 'temporal' in running_containers.lower():
                self.validation_results["infrastructure"]["message_queue"] = {
                    "status": "degraded",
                    "type": "Temporal",
                    "note": "Temporal container found but may be restarting"
                }
                print("⚠️  Message Queue (Temporal): Degraded")
            else:
                self.validation_results["infrastructure"]["message_queue"] = {
                    "status": "missing",
                    "error": "No message queue service detected"
                }
                print("❌ Message Queue: Missing")

        except Exception as e:
            print(f"❌ Message Queue validation error: {str(e)}")

    async def validate_monitoring(self):
        """Validate monitoring stack"""
        print("\n📈 Validating Monitoring Stack")

        # Prometheus (try both instances)
        prometheus_healthy = False
        for port in [9090, 9092]:
            try:
                response = requests.get(f"http://localhost:{port}/api/v1/status/config", timeout=3)
                if response.status_code == 200:
                    self.validation_results["infrastructure"]["prometheus"] = {
                        "status": "healthy",
                        "port": port,
                        "primary": port == 9090
                    }
                    print(f"✅ Prometheus: Healthy on port {port}")
                    prometheus_healthy = True
                    break
            except Exception:
                continue

        if not prometheus_healthy:
            self.validation_results["infrastructure"]["prometheus"] = {
                "status": "error",
                "error": "No Prometheus instance responding"
            }
            print("❌ Prometheus: No instances responding")

        # Grafana (try both instances)
        grafana_healthy = False
        for port in [3000, 3002]:
            try:
                response = requests.get(f"http://localhost:{port}/api/health", timeout=3)
                if response.status_code == 200:
                    health_data = response.json()
                    self.validation_results["infrastructure"]["grafana"] = {
                        "status": "healthy",
                        "port": port,
                        "version": health_data.get("version", "unknown"),
                        "primary": port == 3000
                    }
                    print(f"✅ Grafana: Healthy on port {port}")
                    grafana_healthy = True
                    break
            except Exception:
                continue

        if not grafana_healthy:
            self.validation_results["infrastructure"]["grafana"] = {
                "status": "error",
                "error": "No Grafana instance responding"
            }
            print("❌ Grafana: No instances responding")

    async def validate_ai_services(self):
        """Validate AI/ML services"""
        print("\n🤖 Validating AI Services")

        ai_services = [
            ("Neural Orchestrator", "http://localhost:8003/health"),
            ("Learning Service", "http://localhost:8004/health"),
            ("Threat Detection", "http://localhost:8005/health")
        ]

        for service_name, health_url in ai_services:
            try:
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    self.validation_results["services"][service_name.lower().replace(" ", "_")] = {
                        "status": "healthy",
                        "health_data": health_data,
                        "url": health_url
                    }
                    print(f"✅ {service_name}: Healthy")
                else:
                    raise Exception(f"Service returned status {response.status_code}")

            except Exception as e:
                self.validation_results["services"][service_name.lower().replace(" ", "_")] = {
                    "status": "error",
                    "error": str(e),
                    "url": health_url
                }
                print(f"❌ {service_name}: {str(e)}")

    async def validate_api_services(self):
        """Validate API services (if running)"""
        print("\n🌐 Validating API Services")

        # Check for common API ports
        api_ports = [8000, 8001, 8002, 8006, 8007, 8008]

        for port in api_ports:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=3)
                if response.status_code == 200:
                    health_data = response.json()
                    service_name = health_data.get("service", f"api_port_{port}")
                    self.validation_results["services"][service_name] = {
                        "status": "healthy",
                        "port": port,
                        "health_data": health_data
                    }
                    print(f"✅ API Service ({service_name}): Healthy on port {port}")

            except requests.exceptions.ConnectionError:
                # Service not running on this port, which is expected
                continue
            except Exception as e:
                print(f"⚠️  Port {port}: {str(e)}")

    async def validate_service_connectivity(self):
        """Validate inter-service connectivity"""
        print("\n🔗 Validating Service Connectivity")

        connectivity_tests = []

        # Test database connectivity from services
        for service in ["neural_orchestrator", "learning_service", "threat_detection"]:
            if service in self.validation_results["services"]:
                if self.validation_results["services"][service]["status"] == "healthy":
                    connectivity_tests.append(f"{service} -> PostgreSQL")

        self.validation_results["connectivity"]["tests"] = connectivity_tests
        print(f"✅ Connectivity tests identified: {len(connectivity_tests)}")

    async def validate_performance(self):
        """Basic performance validation"""
        print("\n⚡ Validating Performance")

        performance_metrics = {}

        # Container resource usage
        try:
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format',
                'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                performance_metrics["container_stats"] = result.stdout
                print("✅ Container performance metrics collected")

        except Exception as e:
            print(f"⚠️  Performance metrics: {str(e)}")

        # Response time tests
        response_times = {}
        test_endpoints = [
            ("Neural Orchestrator", "http://localhost:8003/health"),
            ("Learning Service", "http://localhost:8004/health"),
            ("Threat Detection", "http://localhost:8005/health")
        ]

        for service_name, url in test_endpoints:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                end_time = time.time()

                if response.status_code == 200:
                    response_times[service_name] = round((end_time - start_time) * 1000, 2)  # milliseconds

            except Exception as e:
                response_times[service_name] = f"Error: {str(e)}"

        performance_metrics["response_times_ms"] = response_times
        self.validation_results["performance"] = performance_metrics

        print("✅ Performance validation completed")

    def generate_final_report(self):
        """Generate final validation report"""
        print("\n" + "=" * 60)
        print("📋 XORB PLATFORM DEPLOYMENT VALIDATION REPORT")
        print("=" * 60)

        # Count healthy vs error services
        healthy_services = 0
        error_services = 0

        for category in ["infrastructure", "services"]:
            for service, data in self.validation_results.get(category, {}).items():
                if data.get("status") == "healthy":
                    healthy_services += 1
                elif data.get("status") in ["error", "missing"]:
                    error_services += 1

        total_services = healthy_services + error_services

        if error_services == 0:
            self.validation_results["overall_status"] = "healthy"
            status_emoji = "✅"
        elif error_services <= total_services * 0.2:  # Less than 20% errors
            self.validation_results["overall_status"] = "degraded"
            status_emoji = "⚠️"
        else:
            self.validation_results["overall_status"] = "critical"
            status_emoji = "❌"

        print(f"{status_emoji} Overall Status: {self.validation_results['overall_status'].upper()}")
        print(f"📊 Services Summary: {healthy_services} healthy, {error_services} errors")

        # Infrastructure summary
        print(f"\n🏗️  Infrastructure Status:")
        for service, data in self.validation_results.get("infrastructure", {}).items():
            status = data.get("status", "unknown")
            if status == "healthy":
                print(f"   ✅ {service.title()}: {status}")
            else:
                print(f"   ❌ {service.title()}: {status}")

        # Services summary
        print(f"\n🚀 Application Services Status:")
        for service, data in self.validation_results.get("services", {}).items():
            status = data.get("status", "unknown")
            if status == "healthy":
                print(f"   ✅ {service.replace('_', ' ').title()}: {status}")
            else:
                print(f"   ❌ {service.replace('_', ' ').title()}: {status}")

        # Performance summary
        if "response_times_ms" in self.validation_results.get("performance", {}):
            print(f"\n⚡ Response Times:")
            for service, time_ms in self.validation_results["performance"]["response_times_ms"].items():
                if isinstance(time_ms, (int, float)):
                    print(f"   📊 {service}: {time_ms}ms")

        # Save report to file
        report_file = f"/tmp/xorb_deployment_validation_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"\n💾 Full report saved to: {report_file}")
        print("\n🎉 XORB Platform Deployment Validation Complete!")

        return self.validation_results


async def main():
    """Main validation function"""
    validator = XORBDeploymentValidator()
    results = await validator.validate_full_deployment()
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
