#!/usr/bin/env python3
"""
Xorb Game Day Automation Orchestrator
Automates disaster recovery drills and chaos engineering exercises
"""

import asyncio
import json
import logging
import random
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameDayOrchestrator:
    """Orchestrates Game Day exercises and chaos engineering drills"""
    
    def __init__(self, environment: str = "staging"):
        self.environment = environment
        self.exercise_id = f"gameday_{int(time.time())}"
        self.results = {
            "exercise_id": self.exercise_id,
            "start_time": datetime.now().isoformat(),
            "environment": environment,
            "participants": [],
            "scenarios": [],
            "metrics": {},
            "issues": [],
            "lessons_learned": []
        }
        self.prometheus_url = "http://localhost:9090"
        self.active_failures = []
    
    async def setup_monitoring(self, exercise_type: str):
        """Setup enhanced monitoring for Game Day exercise"""
        logger.info(f"Setting up monitoring for {exercise_type} exercise...")
        
        try:
            # Create temporary Grafana dashboard for the exercise
            dashboard_config = {
                "dashboard": {
                    "title": f"Game Day - {exercise_type} - {self.exercise_id}",
                    "tags": ["gameday", exercise_type, self.exercise_id],
                    "time": {"from": "now-1h", "to": "now"},
                    "refresh": "5s",
                    "panels": [
                        {
                            "title": "Service Availability",
                            "type": "stat",
                            "targets": [{"expr": "up{job=~'xorb-.*'}"}]
                        },
                        {
                            "title": "Response Time P95",
                            "type": "timeseries",
                            "targets": [{"expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"}]
                        },
                        {
                            "title": "Error Rate",
                            "type": "timeseries",
                            "targets": [{"expr": "rate(http_requests_total{status=~'5..'}[5m])"}]
                        }
                    ]
                }
            }
            
            # Save dashboard config
            dashboard_file = f"/tmp/gameday_dashboard_{self.exercise_id}.json"
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_config, f, indent=2)
            
            logger.info(f"Game Day dashboard saved to {dashboard_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            self.results["issues"].append(f"Monitoring setup failed: {e}")
            return False
    
    async def inject_database_failure(self, failure_type: str = "process_kill") -> Dict:
        """Inject database failure for testing"""
        logger.info(f"Injecting database failure: {failure_type}")
        
        failure_start = datetime.now()
        failure_id = f"db_failure_{int(time.time())}"
        
        try:
            if failure_type == "process_kill":
                # Kill PostgreSQL process
                result = subprocess.run(
                    ["docker", "exec", "xorb_postgres", "pkill", "-9", "postgres"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                failure_info = {
                    "id": failure_id,
                    "type": "database_process_kill",
                    "start_time": failure_start.isoformat(),
                    "method": "pkill postgres",
                    "success": result.returncode == 0,
                    "output": result.stdout if result.returncode == 0 else result.stderr
                }
            
            elif failure_type == "network_partition":
                # Simulate network partition using iptables
                result = subprocess.run([
                    "docker", "exec", "xorb_postgres", "iptables", 
                    "-A", "INPUT", "-p", "tcp", "--dport", "5432", "-j", "DROP"
                ], capture_output=True, text=True, timeout=10)
                
                failure_info = {
                    "id": failure_id,
                    "type": "database_network_partition",
                    "start_time": failure_start.isoformat(),
                    "method": "iptables DROP",
                    "success": result.returncode == 0,
                    "output": result.stdout if result.returncode == 0 else result.stderr
                }
            
            elif failure_type == "disk_full":
                # Simulate disk full by creating large file
                result = subprocess.run([
                    "docker", "exec", "xorb_postgres", "dd", 
                    "if=/dev/zero", "of=/var/lib/postgresql/data/bigfile", 
                    "bs=1M", "count=1000"
                ], capture_output=True, text=True, timeout=30)
                
                failure_info = {
                    "id": failure_id,
                    "type": "database_disk_full",
                    "start_time": failure_start.isoformat(),
                    "method": "dd large file",
                    "success": result.returncode == 0,
                    "output": result.stdout if result.returncode == 0 else result.stderr
                }
            
            else:
                raise ValueError(f"Unknown failure type: {failure_type}")
            
            self.active_failures.append(failure_info)
            logger.info(f"Database failure injected: {failure_id}")
            return failure_info
            
        except Exception as e:
            logger.error(f"Failed to inject database failure: {e}")
            return {
                "id": failure_id,
                "type": f"database_{failure_type}",
                "start_time": failure_start.isoformat(),
                "success": False,
                "error": str(e)
            }
    
    async def inject_traffic_surge(self, multiplier: int = 10, duration_minutes: int = 5) -> Dict:
        """Inject artificial traffic surge"""
        logger.info(f"Injecting {multiplier}x traffic surge for {duration_minutes} minutes")
        
        surge_start = datetime.now()
        surge_id = f"traffic_surge_{int(time.time())}"
        
        try:
            # Use k6 or curl to generate load
            load_script = f"""
import http from 'k6/http';
import {{ sleep }} from 'k6';

export let options = {{
  vus: {multiplier * 10},
  duration: '{duration_minutes}m',
}};

export default function() {{
  http.get('http://localhost:8000/health');
  http.get('http://localhost:8000/api/v1/scans');
  http.get('http://localhost:8000/api/v1/findings');
  sleep(1);
}}
"""
            
            # Save k6 script
            script_file = f"/tmp/load_test_{surge_id}.js"
            with open(script_file, 'w') as f:
                f.write(load_script)
            
            # Start load test in background
            process = subprocess.Popen([
                "k6", "run", script_file
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            surge_info = {
                "id": surge_id,
                "type": "traffic_surge",
                "start_time": surge_start.isoformat(),
                "multiplier": multiplier,
                "duration_minutes": duration_minutes,
                "process_id": process.pid,
                "script_file": script_file
            }
            
            self.active_failures.append(surge_info)
            logger.info(f"Traffic surge started: {surge_id}")
            return surge_info
            
        except Exception as e:
            logger.error(f"Failed to inject traffic surge: {e}")
            return {
                "id": surge_id,
                "type": "traffic_surge",
                "start_time": surge_start.isoformat(),
                "success": False,
                "error": str(e)
            }
    
    async def inject_container_failure(self, service: str = "random") -> Dict:
        """Inject container failure (kill random container)"""
        logger.info(f"Injecting container failure for service: {service}")
        
        failure_start = datetime.now()
        failure_id = f"container_failure_{int(time.time())}"
        
        try:
            if service == "random":
                # Get list of running Xorb containers
                result = subprocess.run([
                    "docker", "ps", "--filter", "name=xorb_", "--format", "{{.Names}}"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    containers = result.stdout.strip().split('\n')
                    if containers and containers[0]:  # Check if list is not empty
                        service = random.choice(containers)
                    else:
                        service = "xorb_api"  # fallback
                else:
                    service = "xorb_api"  # fallback
            
            # Kill the container
            result = subprocess.run([
                "docker", "kill", service
            ], capture_output=True, text=True, timeout=10)
            
            failure_info = {
                "id": failure_id,
                "type": "container_failure",
                "start_time": failure_start.isoformat(),
                "target_service": service,
                "method": "docker kill",
                "success": result.returncode == 0,
                "output": result.stdout if result.returncode == 0 else result.stderr
            }
            
            self.active_failures.append(failure_info)
            logger.info(f"Container failure injected: {failure_id} (service: {service})")
            return failure_info
            
        except Exception as e:
            logger.error(f"Failed to inject container failure: {e}")
            return {
                "id": failure_id,
                "type": "container_failure",
                "start_time": failure_start.isoformat(),
                "success": False,
                "error": str(e)
            }
    
    async def monitor_recovery_metrics(self, duration_minutes: int = 30) -> Dict:
        """Monitor system recovery metrics during exercise"""
        logger.info(f"Monitoring recovery metrics for {duration_minutes} minutes...")
        
        metrics = {
            "start_time": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "samples": []
        }
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            try:
                sample_time = datetime.now()
                
                # Query Prometheus for key metrics
                async with aiohttp.ClientSession() as session:
                    # Service availability
                    availability_query = "avg(up{job=~'xorb-.*'})"
                    async with session.get(f"{self.prometheus_url}/api/v1/query",
                                         params={"query": availability_query}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            result = data.get("data", {}).get("result", [])
                            availability = float(result[0]["value"][1]) if result else 0.0
                        else:
                            availability = 0.0
                    
                    # Response time P95
                    latency_query = "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
                    async with session.get(f"{self.prometheus_url}/api/v1/query",
                                         params={"query": latency_query}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            result = data.get("data", {}).get("result", [])
                            p95_latency = float(result[0]["value"][1]) if result else 0.0
                        else:
                            p95_latency = 0.0
                    
                    # Error rate
                    error_query = "rate(http_requests_total{status=~'5..'}[5m])"
                    async with session.get(f"{self.prometheus_url}/api/v1/query",
                                         params={"query": error_query}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            result = data.get("data", {}).get("result", [])
                            error_rate = float(result[0]["value"][1]) if result else 0.0
                        else:
                            error_rate = 0.0
                
                sample = {
                    "timestamp": sample_time.isoformat(),
                    "availability": availability,
                    "p95_latency_seconds": p95_latency,
                    "error_rate_per_second": error_rate
                }
                
                metrics["samples"].append(sample)
                logger.info(f"Sample: availability={availability:.3f}, "
                           f"p95={p95_latency:.3f}s, errors={error_rate:.3f}/s")
                
                await asyncio.sleep(30)  # Sample every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)
        
        # Calculate summary statistics
        if metrics["samples"]:
            availabilities = [s["availability"] for s in metrics["samples"]]
            latencies = [s["p95_latency_seconds"] for s in metrics["samples"]]
            error_rates = [s["error_rate_per_second"] for s in metrics["samples"]]
            
            metrics["summary"] = {
                "avg_availability": sum(availabilities) / len(availabilities),
                "min_availability": min(availabilities),
                "max_p95_latency": max(latencies) if latencies else 0,
                "avg_p95_latency": sum(latencies) / len(latencies) if latencies else 0,
                "max_error_rate": max(error_rates) if error_rates else 0,
                "total_samples": len(metrics["samples"])
            }
        
        return metrics
    
    async def cleanup_failures(self):
        """Cleanup all injected failures"""
        logger.info("Cleaning up injected failures...")
        
        for failure in self.active_failures:
            try:
                if failure["type"] == "database_process_kill":
                    # Restart PostgreSQL container
                    subprocess.run(["docker", "restart", "xorb_postgres"], 
                                 capture_output=True, timeout=30)
                
                elif failure["type"] == "database_network_partition":
                    # Remove iptables rule
                    subprocess.run([
                        "docker", "exec", "xorb_postgres", "iptables", 
                        "-D", "INPUT", "-p", "tcp", "--dport", "5432", "-j", "DROP"
                    ], capture_output=True, timeout=10)
                
                elif failure["type"] == "database_disk_full":
                    # Remove large file
                    subprocess.run([
                        "docker", "exec", "xorb_postgres", "rm", "-f", 
                        "/var/lib/postgresql/data/bigfile"
                    ], capture_output=True, timeout=10)
                
                elif failure["type"] == "traffic_surge":
                    # Kill load test process
                    if "process_id" in failure:
                        subprocess.run(["kill", str(failure["process_id"])], 
                                     capture_output=True)
                    
                    # Remove script file
                    if "script_file" in failure:
                        Path(failure["script_file"]).unlink(missing_ok=True)
                
                elif failure["type"] == "container_failure":
                    # Restart the container
                    service = failure.get("target_service", "")
                    if service:
                        subprocess.run(["docker", "start", service], 
                                     capture_output=True, timeout=30)
                
                logger.info(f"Cleaned up failure: {failure['id']}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup {failure['id']}: {e}")
                self.results["issues"].append(f"Cleanup failed for {failure['id']}: {e}")
        
        self.active_failures.clear()
        logger.info("Failure cleanup completed")
    
    async def run_database_failover_drill(self) -> Dict:
        """Run comprehensive database failover drill"""
        logger.info("Starting database failover drill...")
        
        scenario = {
            "name": "Database Failover Drill",
            "type": "database_failure",
            "start_time": datetime.now().isoformat(),
            "phases": []
        }
        
        try:
            # Phase 1: Setup monitoring
            await self.setup_monitoring("database_failover")
            scenario["phases"].append({"phase": "monitoring_setup", "status": "completed"})
            
            # Phase 2: Baseline metrics
            logger.info("Collecting baseline metrics...")
            await asyncio.sleep(60)  # 1 minute baseline
            scenario["phases"].append({"phase": "baseline_collection", "status": "completed"})
            
            # Phase 3: Inject failure
            failure_info = await self.inject_database_failure("process_kill")
            scenario["failure_info"] = failure_info
            scenario["phases"].append({"phase": "failure_injection", "status": "completed"})
            
            # Phase 4: Monitor recovery
            logger.info("Monitoring system recovery...")
            recovery_metrics = await self.monitor_recovery_metrics(15)  # 15 minutes
            scenario["recovery_metrics"] = recovery_metrics
            scenario["phases"].append({"phase": "recovery_monitoring", "status": "completed"})
            
            # Phase 5: Validation
            logger.info("Validating system recovery...")
            await asyncio.sleep(60)  # Allow additional time for full recovery
            
            # Check if services are back up
            result = subprocess.run([
                "curl", "-f", "http://localhost:8000/health"
            ], capture_output=True, timeout=10)
            
            scenario["recovery_validated"] = result.returncode == 0
            scenario["phases"].append({
                "phase": "recovery_validation", 
                "status": "completed" if result.returncode == 0 else "failed"
            })
            
            scenario["end_time"] = datetime.now().isoformat()
            scenario["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Database failover drill failed: {e}")
            scenario["error"] = str(e)
            scenario["status"] = "failed"
            self.results["issues"].append(f"Database drill failed: {e}")
        
        finally:
            # Always cleanup
            await self.cleanup_failures()
        
        return scenario
    
    async def run_traffic_surge_drill(self) -> Dict:
        """Run traffic surge handling drill"""
        logger.info("Starting traffic surge drill...")
        
        scenario = {
            "name": "Traffic Surge Drill",
            "type": "traffic_surge",
            "start_time": datetime.now().isoformat(),
            "phases": []
        }
        
        try:
            # Setup monitoring
            await self.setup_monitoring("traffic_surge")
            scenario["phases"].append({"phase": "monitoring_setup", "status": "completed"})
            
            # Baseline
            await asyncio.sleep(30)
            scenario["phases"].append({"phase": "baseline_collection", "status": "completed"})
            
            # Inject traffic surge
            surge_info = await self.inject_traffic_surge(multiplier=5, duration_minutes=10)
            scenario["surge_info"] = surge_info
            scenario["phases"].append({"phase": "surge_injection", "status": "completed"})
            
            # Monitor performance
            performance_metrics = await self.monitor_recovery_metrics(12)  # 12 minutes
            scenario["performance_metrics"] = performance_metrics
            scenario["phases"].append({"phase": "performance_monitoring", "status": "completed"})
            
            scenario["end_time"] = datetime.now().isoformat()
            scenario["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Traffic surge drill failed: {e}")
            scenario["error"] = str(e)
            scenario["status"] = "failed"
        
        finally:
            await self.cleanup_failures()
        
        return scenario
    
    def generate_report(self) -> Dict:
        """Generate comprehensive Game Day report"""
        self.results["end_time"] = datetime.now().isoformat()
        
        # Calculate total duration
        start_time = datetime.fromisoformat(self.results["start_time"])
        end_time = datetime.fromisoformat(self.results["end_time"])
        duration = end_time - start_time
        self.results["total_duration_minutes"] = duration.total_seconds() / 60
        
        # Add summary
        self.results["summary"] = {
            "total_scenarios": len(self.results["scenarios"]),
            "successful_scenarios": len([s for s in self.results["scenarios"] if s.get("status") == "completed"]),
            "failed_scenarios": len([s for s in self.results["scenarios"] if s.get("status") == "failed"]),
            "total_issues": len(self.results["issues"]),
            "environment": self.environment
        }
        
        # Add lessons learned based on scenarios
        for scenario in self.results["scenarios"]:
            if scenario.get("status") == "completed":
                recovery_metrics = scenario.get("recovery_metrics", {}).get("summary", {})
                if recovery_metrics:
                    min_availability = recovery_metrics.get("min_availability", 1.0)
                    max_latency = recovery_metrics.get("max_p95_latency", 0)
                    
                    if min_availability < 0.9:
                        self.results["lessons_learned"].append(
                            f"Service availability dropped to {min_availability:.1%} during {scenario['name']}"
                        )
                    
                    if max_latency > 1.0:
                        self.results["lessons_learned"].append(
                            f"P95 latency spiked to {max_latency:.2f}s during {scenario['name']}"
                        )
        
        return self.results

async def main():
    """Main function for Game Day orchestration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Xorb Game Day Automation Orchestrator")
    parser.add_argument("--environment", default="staging", choices=["staging", "development"],
                       help="Target environment for Game Day")
    parser.add_argument("--scenario", choices=["database", "traffic", "container", "all"],
                       default="all", help="Scenario to run")
    parser.add_argument("--duration", type=int, default=30,
                       help="Total exercise duration in minutes")
    parser.add_argument("--participants", nargs="*", default=[],
                       help="List of participant names/emails")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = GameDayOrchestrator(environment=args.environment)
    orchestrator.results["participants"] = args.participants
    
    logger.info(f"🎮 Starting Game Day exercise: {orchestrator.exercise_id}")
    logger.info(f"🎯 Environment: {args.environment}")
    logger.info(f"🕐 Duration: {args.duration} minutes")
    
    try:
        # Run scenarios based on selection
        if args.scenario in ["database", "all"]:
            db_scenario = await orchestrator.run_database_failover_drill()
            orchestrator.results["scenarios"].append(db_scenario)
        
        if args.scenario in ["traffic", "all"]:
            traffic_scenario = await orchestrator.run_traffic_surge_drill()
            orchestrator.results["scenarios"].append(traffic_scenario)
        
        if args.scenario in ["container", "all"]:
            # Simple container failure test
            container_failure = await orchestrator.inject_container_failure()
            await asyncio.sleep(300)  # 5 minutes
            await orchestrator.cleanup_failures()
            
            container_scenario = {
                "name": "Container Failure Drill",
                "type": "container_failure",
                "start_time": datetime.now().isoformat(),
                "failure_info": container_failure,
                "status": "completed"
            }
            orchestrator.results["scenarios"].append(container_scenario)
        
        # Generate final report
        report = orchestrator.generate_report()
        
        # Save report
        report_file = f"/tmp/gameday_report_{orchestrator.exercise_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Game Day report saved to: {report_file}")
        
        # Print summary
        print(f"\n🎮 Game Day Exercise Complete!")
        print(f"📋 Exercise ID: {orchestrator.exercise_id}")
        print(f"⏱️  Duration: {report['total_duration_minutes']:.1f} minutes")
        print(f"✅ Successful scenarios: {report['summary']['successful_scenarios']}")
        print(f"❌ Failed scenarios: {report['summary']['failed_scenarios']}")
        print(f"⚠️  Issues encountered: {report['summary']['total_issues']}")
        
        if report["lessons_learned"]:
            print(f"\n💡 Key Lessons Learned:")
            for lesson in report["lessons_learned"]:
                print(f"  • {lesson}")
        
        # Exit with appropriate code
        if report["summary"]["failed_scenarios"] > 0:
            exit(1)
        else:
            exit(0)
            
    except KeyboardInterrupt:
        logger.info("Game Day exercise interrupted by user")
        await orchestrator.cleanup_failures()
        exit(130)
    except Exception as e:
        logger.error(f"Game Day exercise failed: {e}")
        await orchestrator.cleanup_failures()
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())