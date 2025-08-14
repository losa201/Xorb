#!/usr/bin/env python3
"""
XORB PTaaS Platform Validation Script
Comprehensive end-to-end testing suite
"""

import asyncio
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional

class PTaaSValidator:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results = []
        self.tenant_id = None
        self.scan_id = None

    def log_result(self, test_name: str, success: bool, message: str, duration: float = 0):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "duration": f"{duration:.2f}s",
            "timestamp": datetime.utcnow().isoformat()
        }
        self.results.append(result)

        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message} ({duration:.2f}s)")

    def test_infrastructure_services(self):
        """Test all infrastructure services are healthy"""
        services = {
            "PostgreSQL": "http://localhost:5432",
            "Redis": "http://localhost:6381",
            "RabbitMQ": "http://localhost:15672",
            "Qdrant": "http://localhost:6335",
            "Prometheus": "http://localhost:9091",
            "Grafana": "http://localhost:3002"
        }

        for service, url in services.items():
            start_time = time.time()
            try:
                # Simple connectivity test
                import socket
                host, port = url.replace("http://", "").split(":")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, int(port)))
                sock.close()

                duration = time.time() - start_time
                if result == 0:
                    self.log_result(f"Infrastructure: {service}", True, "Service is accessible", duration)
                else:
                    self.log_result(f"Infrastructure: {service}", False, "Service not accessible", duration)
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(f"Infrastructure: {service}", False, f"Connection error: {str(e)}", duration)

    def test_api_gateway_health(self):
        """Test API Gateway health endpoint"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data", {}).get("status") == "healthy":
                    self.log_result("API Gateway Health", True, "Gateway is healthy", duration)
                else:
                    self.log_result("API Gateway Health", False, "Gateway reports unhealthy status", duration)
            else:
                self.log_result("API Gateway Health", False, f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("API Gateway Health", False, str(e), duration)

    def test_orchestrator_connectivity(self):
        """Test Service Orchestrator connectivity"""
        start_time = time.time()
        try:
            response = requests.get("http://localhost:8200/health", timeout=10)
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_result("Orchestrator Health", True, "Orchestrator is healthy", duration)
                else:
                    self.log_result("Orchestrator Health", False, "Orchestrator reports unhealthy", duration)
            else:
                self.log_result("Orchestrator Health", False, f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Orchestrator Health", False, str(e), duration)

    def test_tenant_management(self):
        """Test tenant creation and management"""
        start_time = time.time()
        try:
            # Create a test tenant
            tenant_data = {
                "name": "Validation Test Corp",
                "plan": "enterprise",
                "email": "validation@test.com"
            }

            response = requests.post(
                f"{self.base_url}/api/v1/tenants",
                json=tenant_data,
                timeout=10
            )

            duration = time.time() - start_time

            if response.status_code == 200:
                tenant = response.json()
                if tenant.get("id") and tenant.get("name") == tenant_data["name"]:
                    self.tenant_id = tenant["id"]
                    self.log_result("Tenant Creation", True, f"Created tenant: {self.tenant_id}", duration)

                    # Test tenant retrieval
                    self.test_tenant_retrieval()
                else:
                    self.log_result("Tenant Creation", False, "Invalid tenant response", duration)
            else:
                self.log_result("Tenant Creation", False, f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Tenant Creation", False, str(e), duration)

    def test_tenant_retrieval(self):
        """Test tenant retrieval"""
        if not self.tenant_id:
            self.log_result("Tenant Retrieval", False, "No tenant ID available", 0)
            return

        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/api/v1/tenants/{self.tenant_id}", timeout=10)
            duration = time.time() - start_time

            if response.status_code == 200:
                tenant = response.json()
                if tenant.get("id") == self.tenant_id:
                    self.log_result("Tenant Retrieval", True, "Tenant retrieved successfully", duration)
                else:
                    self.log_result("Tenant Retrieval", False, "Tenant ID mismatch", duration)
            else:
                self.log_result("Tenant Retrieval", False, f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Tenant Retrieval", False, str(e), duration)

    def test_scan_creation(self):
        """Test scan creation and execution"""
        if not self.tenant_id:
            self.log_result("Scan Creation", False, "No tenant ID available", 0)
            return

        start_time = time.time()
        try:
            scan_data = {
                "tenant_id": self.tenant_id,
                "target": "validation.test.local",
                "scan_type": "validation",
                "config": {"test_mode": True}
            }

            response = requests.post(
                f"{self.base_url}/api/v1/scans",
                json=scan_data,
                timeout=10
            )

            duration = time.time() - start_time

            if response.status_code == 200:
                scan = response.json()
                if scan.get("id") and scan.get("status") in ["queued", "running"]:
                    self.scan_id = scan["id"]
                    self.log_result("Scan Creation", True, f"Created scan: {self.scan_id}", duration)

                    # Wait and test scan progress
                    time.sleep(5)
                    self.test_scan_progress()
                else:
                    self.log_result("Scan Creation", False, "Invalid scan response", duration)
            else:
                self.log_result("Scan Creation", False, f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Scan Creation", False, str(e), duration)

    def test_scan_progress(self):
        """Test scan progress tracking"""
        if not self.scan_id:
            self.log_result("Scan Progress", False, "No scan ID available", 0)
            return

        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/api/v1/scans/{self.scan_id}", timeout=10)
            duration = time.time() - start_time

            if response.status_code == 200:
                scan = response.json()
                status = scan.get("status")
                progress = scan.get("progress", 0)

                if status in ["queued", "running", "completed"]:
                    self.log_result("Scan Progress", True, f"Status: {status}, Progress: {progress}%", duration)
                else:
                    self.log_result("Scan Progress", False, f"Invalid status: {status}", duration)
            else:
                self.log_result("Scan Progress", False, f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Scan Progress", False, str(e), duration)

    def test_dashboard_data(self):
        """Test dashboard data endpoint"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/api/v1/dashboard", timeout=10)
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if (data.get("success") and
                    "stats" in data.get("data", {}) and
                    "service_health" in data.get("data", {})):
                    stats = data["data"]["stats"]
                    self.log_result("Dashboard Data", True,
                                  f"Tenants: {stats.get('total_tenants')}, Scans: {stats.get('total_scans')}",
                                  duration)
                else:
                    self.log_result("Dashboard Data", False, "Invalid dashboard response structure", duration)
            else:
                self.log_result("Dashboard Data", False, f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Dashboard Data", False, str(e), duration)

    def test_metrics_endpoints(self):
        """Test Prometheus metrics endpoints"""
        endpoints = [
            ("API Gateway Metrics", f"{self.base_url}/metrics"),
            ("Orchestrator Metrics", "http://localhost:8200/metrics")
        ]

        for name, url in endpoints:
            start_time = time.time()
            try:
                response = requests.get(url, timeout=10)
                duration = time.time() - start_time

                if response.status_code == 200:
                    content = response.text
                    if "# HELP" in content and "# TYPE" in content:
                        self.log_result(name, True, "Metrics format valid", duration)
                    else:
                        self.log_result(name, False, "Invalid metrics format", duration)
                else:
                    self.log_result(name, False, f"HTTP {response.status_code}", duration)
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(name, False, str(e), duration)

    def test_web_dashboard(self):
        """Test web dashboard accessibility"""
        start_time = time.time()
        try:
            response = requests.get("http://localhost:3000", timeout=10)
            duration = time.time() - start_time

            if response.status_code == 200:
                content = response.text
                if "XORB PTaaS Platform" in content:
                    self.log_result("Web Dashboard", True, "Dashboard accessible", duration)
                else:
                    self.log_result("Web Dashboard", False, "Dashboard content invalid", duration)
            else:
                self.log_result("Web Dashboard", False, f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Web Dashboard", False, str(e), duration)

    def run_validation(self):
        """Run complete platform validation"""
        print("🚀 Starting XORB PTaaS Platform Validation")
        print("=" * 60)

        validation_start = time.time()

        # Run all tests
        self.test_infrastructure_services()
        self.test_api_gateway_health()
        self.test_orchestrator_connectivity()
        self.test_tenant_management()
        self.test_scan_creation()
        self.test_dashboard_data()
        self.test_metrics_endpoints()
        self.test_web_dashboard()

        total_duration = time.time() - validation_start

        # Generate summary
        self.generate_summary(total_duration)

        return self.results

    def generate_summary(self, total_duration: float):
        """Generate validation summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️ Total Duration: {total_duration:.2f}s")

        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result["success"]:
                    print(f"  • {result['test']}: {result['message']}")

        overall_status = "🎉 PLATFORM HEALTHY" if success_rate >= 80 else "⚠️ PLATFORM ISSUES DETECTED"
        print(f"\n{overall_status}")
        print("=" * 60)

        # Save detailed results
        report_file = f"/tmp/ptaas_validation_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "success_rate": success_rate,
                    "duration": total_duration,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "results": self.results
            }, f, indent=2)

        print(f"📄 Detailed report saved to: {report_file}")

def main():
    validator = PTaaSValidator()
    results = validator.run_validation()
    return results

if __name__ == "__main__":
    main()
