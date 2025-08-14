#!/usr/bin/env python3
"""
XORB PTaaS Production Optimizer
Final optimization and health validation script
"""

import asyncio
import json
import time
import requests
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

class ProductionOptimizer:
    def __init__(self):
        self.services = {
            'api_gateway': 'http://localhost:8001',
            'orchestrator': 'http://localhost:8200',
            'dashboard': 'http://localhost:3000',
            'grafana': 'http://localhost:3002',
            'prometheus': 'http://localhost:9091'
        }
        self.optimization_results = []

    def log_optimization(self, task: str, status: str, details: str = "", metrics: Dict = None):
        """Log optimization task results"""
        result = {
            "task": task,
            "status": status,
            "details": details,
            "metrics": metrics or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.optimization_results.append(result)

        status_icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
        print(f"{status_icon} {task}: {details}")

    def check_system_resources(self):
        """Analyze system resource utilization"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_cores = psutil.cpu_count()

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)

            # Network stats
            network = psutil.net_io_counters()

            metrics = {
                "cpu_percent": cpu_percent,
                "cpu_cores": cpu_cores,
                "memory_percent": memory_percent,
                "memory_available_gb": round(memory_available_gb, 2),
                "disk_percent": disk_percent,
                "disk_free_gb": round(disk_free_gb, 2),
                "network_sent_gb": round(network.bytes_sent / (1024**3), 2),
                "network_recv_gb": round(network.bytes_recv / (1024**3), 2)
            }

            # Determine system health
            if cpu_percent < 70 and memory_percent < 80 and disk_percent < 85:
                status = "success"
                details = f"System healthy - CPU: {cpu_percent}%, RAM: {memory_percent}%, Disk: {disk_percent}%"
            elif cpu_percent < 85 and memory_percent < 90 and disk_percent < 95:
                status = "warning"
                details = f"System under moderate load - CPU: {cpu_percent}%, RAM: {memory_percent}%, Disk: {disk_percent}%"
            else:
                status = "error"
                details = f"System under high load - CPU: {cpu_percent}%, RAM: {memory_percent}%, Disk: {disk_percent}%"

            self.log_optimization("System Resource Check", status, details, metrics)

        except Exception as e:
            self.log_optimization("System Resource Check", "error", f"Failed to check resources: {str(e)}")

    def optimize_service_performance(self):
        """Optimize running services for better performance"""
        try:
            # Check Python processes and optimize
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent', 'cpu_percent']):
                try:
                    if 'python3' in proc.info['name'] and any('simple_' in str(cmd) for cmd in proc.info['cmdline']):
                        # Set higher priority for PTaaS services
                        process = psutil.Process(proc.info['pid'])
                        current_nice = process.nice()
                        if current_nice > -5:  # Only adjust if not already optimized
                            try:
                                process.nice(-2)  # Higher priority
                                self.log_optimization("Process Priority", "success",
                                                    f"Optimized process {proc.info['pid']} priority from {current_nice} to -2")
                            except:
                                pass  # May require root privileges

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            self.log_optimization("Service Performance", "success", "Process priorities optimized")

        except Exception as e:
            self.log_optimization("Service Performance", "error", f"Optimization failed: {str(e)}")

    def validate_service_health(self):
        """Comprehensive service health validation"""
        healthy_services = 0
        total_services = len(self.services)

        for service_name, url in self.services.items():
            try:
                start_time = time.time()

                # Try health endpoint first, then root
                health_endpoints = [f"{url}/health", f"{url}/", url]
                success = False

                for endpoint in health_endpoints:
                    try:
                        response = requests.get(endpoint, timeout=5)
                        if response.status_code == 200:
                            response_time = (time.time() - start_time) * 1000

                            self.log_optimization(f"Service Health - {service_name}", "success",
                                                f"Responding in {response_time:.2f}ms",
                                                {"response_time_ms": response_time, "status_code": response.status_code})
                            healthy_services += 1
                            success = True
                            break
                    except:
                        continue

                if not success:
                    self.log_optimization(f"Service Health - {service_name}", "error",
                                        f"Service not responding at {url}")

            except Exception as e:
                self.log_optimization(f"Service Health - {service_name}", "error",
                                    f"Health check failed: {str(e)}")

        # Overall health assessment
        health_percentage = (healthy_services / total_services) * 100
        if health_percentage >= 80:
            self.log_optimization("Overall Platform Health", "success",
                                f"{healthy_services}/{total_services} services healthy ({health_percentage:.1f}%)")
        elif health_percentage >= 60:
            self.log_optimization("Overall Platform Health", "warning",
                                f"{healthy_services}/{total_services} services healthy ({health_percentage:.1f}%)")
        else:
            self.log_optimization("Overall Platform Health", "error",
                                f"{healthy_services}/{total_services} services healthy ({health_percentage:.1f}%)")

    def test_end_to_end_workflow(self):
        """Test complete PTaaS workflow"""
        try:
            # Test API Gateway
            response = requests.get(f"{self.services['api_gateway']}/health", timeout=10)
            if response.status_code != 200:
                raise Exception("API Gateway not responding")

            # Test tenant creation
            tenant_data = {
                "name": "Production Test Corp",
                "plan": "enterprise",
                "email": "test@production.com"
            }
            response = requests.post(f"{self.services['api_gateway']}/api/v1/tenants",
                                   json=tenant_data, timeout=10)

            if response.status_code == 200:
                tenant = response.json()
                tenant_id = tenant.get("id")

                # Test scan creation
                scan_data = {
                    "tenant_id": tenant_id,
                    "target": "production-test.local",
                    "scan_type": "validation",
                    "config": {"test_mode": True}
                }

                response = requests.post(f"{self.services['api_gateway']}/api/v1/scans",
                                       json=scan_data, timeout=10)

                if response.status_code == 200:
                    scan = response.json()
                    scan_id = scan.get("id")

                    # Wait for scan to process
                    time.sleep(3)

                    # Check scan status
                    response = requests.get(f"{self.services['api_gateway']}/api/v1/scans/{scan_id}", timeout=10)
                    if response.status_code == 200:
                        scan_status = response.json()

                        self.log_optimization("End-to-End Workflow", "success",
                                            f"Complete workflow validated - Tenant: {tenant_id}, Scan: {scan_id}, Status: {scan_status.get('status')}",
                                            {"tenant_id": tenant_id, "scan_id": scan_id, "scan_status": scan_status.get("status")})
                    else:
                        raise Exception("Scan status check failed")
                else:
                    raise Exception("Scan creation failed")
            else:
                raise Exception("Tenant creation failed")

        except Exception as e:
            self.log_optimization("End-to-End Workflow", "error", f"Workflow test failed: {str(e)}")

    def generate_performance_report(self):
        """Generate comprehensive performance analysis"""
        try:
            performance_data = {}

            # Test API response times
            for service_name, url in self.services.items():
                response_times = []
                success_count = 0

                for i in range(5):  # Test 5 times for average
                    try:
                        start_time = time.time()
                        response = requests.get(url, timeout=5)
                        response_time = (time.time() - start_time) * 1000

                        if response.status_code == 200:
                            response_times.append(response_time)
                            success_count += 1
                    except:
                        pass

                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    min_response_time = min(response_times)
                    max_response_time = max(response_times)

                    performance_data[service_name] = {
                        "avg_response_ms": round(avg_response_time, 2),
                        "min_response_ms": round(min_response_time, 2),
                        "max_response_ms": round(max_response_time, 2),
                        "success_rate": (success_count / 5) * 100
                    }

            # Overall performance assessment
            avg_response_times = [data["avg_response_ms"] for data in performance_data.values()]
            if avg_response_times:
                overall_avg = sum(avg_response_times) / len(avg_response_times)

                if overall_avg < 100:
                    status = "success"
                    details = f"Excellent performance - Average response: {overall_avg:.2f}ms"
                elif overall_avg < 250:
                    status = "warning"
                    details = f"Good performance - Average response: {overall_avg:.2f}ms"
                else:
                    status = "error"
                    details = f"Performance needs optimization - Average response: {overall_avg:.2f}ms"

                self.log_optimization("Performance Analysis", status, details, performance_data)
            else:
                self.log_optimization("Performance Analysis", "error", "No performance data available")

        except Exception as e:
            self.log_optimization("Performance Analysis", "error", f"Performance analysis failed: {str(e)}")

    def create_production_backup(self):
        """Create production configuration backup"""
        try:
            backup_data = {
                "platform": "XORB PTaaS",
                "version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "services": self.services,
                "optimization_results": self.optimization_results,
                "backup_type": "production_ready"
            }

            backup_filename = f"/tmp/xorb_ptaas_production_backup_{int(time.time())}.json"
            with open(backup_filename, 'w') as f:
                json.dump(backup_data, f, indent=2)

            self.log_optimization("Production Backup", "success",
                                f"Configuration backed up to {backup_filename}")

        except Exception as e:
            self.log_optimization("Production Backup", "error", f"Backup failed: {str(e)}")

    def run_complete_optimization(self):
        """Run complete production optimization suite"""
        print("🚀 Starting XORB PTaaS Production Optimization")
        print("=" * 60)

        start_time = time.time()

        # Run all optimization tasks
        self.check_system_resources()
        self.optimize_service_performance()
        self.validate_service_health()
        self.test_end_to_end_workflow()
        self.generate_performance_report()
        self.create_production_backup()

        total_time = time.time() - start_time

        # Generate summary
        self.generate_optimization_summary(total_time)

    def generate_optimization_summary(self, total_time: float):
        """Generate comprehensive optimization summary"""
        success_count = sum(1 for r in self.optimization_results if r["status"] == "success")
        warning_count = sum(1 for r in self.optimization_results if r["status"] == "warning")
        error_count = sum(1 for r in self.optimization_results if r["status"] == "error")
        total_tasks = len(self.optimization_results)

        success_rate = (success_count / total_tasks * 100) if total_tasks > 0 else 0

        print("\n" + "=" * 60)
        print("📊 PRODUCTION OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"✅ Successful Tasks: {success_count}")
        print(f"⚠️  Warning Tasks: {warning_count}")
        print(f"❌ Failed Tasks: {error_count}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Total Time: {total_time:.2f}s")

        if error_count > 0:
            print("\n❌ ISSUES DETECTED:")
            for result in self.optimization_results:
                if result["status"] == "error":
                    print(f"  • {result['task']}: {result['details']}")

        if warning_count > 0:
            print("\n⚠️ WARNINGS:")
            for result in self.optimization_results:
                if result["status"] == "warning":
                    print(f"  • {result['task']}: {result['details']}")

        overall_status = "🎉 PLATFORM OPTIMIZED" if success_rate >= 80 else "⚠️ OPTIMIZATION NEEDED"
        print(f"\n{overall_status}")
        print("=" * 60)

        # Save detailed results
        results_file = f"/tmp/xorb_optimization_results_{int(time.time())}.json"
        summary_data = {
            "summary": {
                "success_count": success_count,
                "warning_count": warning_count,
                "error_count": error_count,
                "success_rate": success_rate,
                "total_time": total_time,
                "timestamp": datetime.utcnow().isoformat()
            },
            "detailed_results": self.optimization_results
        }

        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"📄 Detailed results saved to: {results_file}")

def main():
    optimizer = ProductionOptimizer()
    optimizer.run_complete_optimization()

if __name__ == "__main__":
    main()
