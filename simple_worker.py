#!/usr/bin/env python3
"""
XORB Worker Service - Advanced Task Execution
Handles distributed task processing and agent workflow execution
"""

import json
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


class TaskProcessor:
    """Advanced task processing engine"""

    def __init__(self):
        self.task_queue = queue.Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.worker_threads = []
        self.running = True

    def start_workers(self, num_workers=4):
        """Start worker threads"""
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(f"worker-{i}",))
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)

    def _worker_loop(self, worker_id):
        """Main worker processing loop"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task:
                    self._process_task(task, worker_id)
                    self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

    def _process_task(self, task, worker_id):
        """Process individual task"""
        task_id = task["id"]
        task_type = task["type"]

        self.active_tasks[task_id] = {
            **task,
            "worker_id": worker_id,
            "started_at": time.time(),
            "status": "processing"
        }

        try:
            # Simulate different task types
            if task_type == "reconnaissance":
                result = self._execute_recon_task(task)
            elif task_type == "vulnerability_scan":
                result = self._execute_vuln_task(task)
            elif task_type == "threat_hunting":
                result = self._execute_threat_task(task)
            elif task_type == "ai_analysis":
                result = self._execute_ai_task(task)
            else:
                result = self._execute_generic_task(task)

            # Mark as completed
            completed_task = self.active_tasks.pop(task_id)
            completed_task.update({
                "status": "completed",
                "completed_at": time.time(),
                "result": result,
                "execution_time": time.time() - completed_task["started_at"]
            })
            self.completed_tasks[task_id] = completed_task

        except Exception as e:
            # Mark as failed
            failed_task = self.active_tasks.pop(task_id)
            failed_task.update({
                "status": "failed",
                "completed_at": time.time(),
                "error": str(e),
                "execution_time": time.time() - failed_task["started_at"]
            })
            self.completed_tasks[task_id] = failed_task

    def _execute_recon_task(self, task):
        """Execute reconnaissance task"""
        time.sleep(2)  # Simulate port scanning
        return {
            "type": "reconnaissance",
            "target": task.get("target", "unknown"),
            "ports_found": [80, 443, 22, 3389],
            "services": ["HTTP", "HTTPS", "SSH", "RDP"],
            "os_fingerprint": "Linux Ubuntu 20.04",
            "vulnerability_indicators": 3
        }

    def _execute_vuln_task(self, task):
        """Execute vulnerability scanning task"""
        time.sleep(3)  # Simulate vulnerability scanning
        return {
            "type": "vulnerability_scan",
            "target": task.get("target", "unknown"),
            "vulnerabilities": [
                {"id": "CVE-2023-1234", "severity": "HIGH", "service": "Apache"},
                {"id": "CVE-2023-5678", "severity": "MEDIUM", "service": "OpenSSH"},
                {"id": "CVE-2023-9999", "severity": "LOW", "service": "Nginx"}
            ],
            "risk_score": 7.8,
            "recommendations": ["Update Apache", "Patch OpenSSH", "Configure firewall"]
        }

    def _execute_threat_task(self, task):
        """Execute threat hunting task"""
        time.sleep(4)  # Simulate threat analysis
        return {
            "type": "threat_hunting",
            "target": task.get("target", "unknown"),
            "iocs_found": [
                {"type": "ip", "value": "192.168.1.100", "confidence": 85},
                {"type": "domain", "value": "malicious.example.com", "confidence": 92},
                {"type": "hash", "value": "abc123...", "confidence": 78}
            ],
            "threat_score": 8.5,
            "threats_detected": ["APT29", "Cobalt Strike"],
            "mitigation_actions": ["Block IP", "Update signatures", "Monitor traffic"]
        }

    def _execute_ai_task(self, task):
        """Execute AI-powered analysis task"""
        time.sleep(5)  # Simulate AI processing
        return {
            "type": "ai_analysis",
            "target": task.get("target", "unknown"),
            "ai_insights": [
                "Unusual network traffic patterns detected",
                "Potential data exfiltration indicators",
                "Recommended immediate investigation"
            ],
            "confidence_score": 94.2,
            "risk_level": "HIGH",
            "next_actions": ["Deep packet inspection", "Behavioral analysis", "Incident response"]
        }

    def _execute_generic_task(self, task):
        """Execute generic task"""
        time.sleep(1)
        return {
            "type": "generic",
            "status": "completed",
            "message": f"Task {task['id']} processed successfully"
        }

    def submit_task(self, task):
        """Submit task for processing"""
        task["id"] = str(uuid.uuid4())
        task["submitted_at"] = time.time()
        self.task_queue.put(task)
        return task["id"]

    def get_task_status(self, task_id):
        """Get task status"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return None

# Global task processor
task_processor = TaskProcessor()

class XORBWorkerHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/":
            self.send_json_response({
                "service": "XORB Worker Service",
                "version": "2.0.0",
                "status": "operational",
                "message": "Advanced task execution and workflow processing",
                "capabilities": [
                    "distributed_task_processing",
                    "agent_workflow_execution",
                    "ai_powered_analysis",
                    "threat_intelligence_processing",
                    "vulnerability_assessment",
                    "reconnaissance_operations"
                ]
            })

        elif path == "/health":
            active_count = len(task_processor.active_tasks)
            completed_count = len(task_processor.completed_tasks)
            queue_size = task_processor.task_queue.qsize()

            self.send_json_response({
                "status": "healthy",
                "service": "xorb-worker",
                "version": "2.0.0",
                "timestamp": time.time(),
                "uptime": time.time() - start_time,
                "performance": {
                    "active_tasks": active_count,
                    "completed_tasks": completed_count,
                    "queue_size": queue_size,
                    "worker_threads": len(task_processor.worker_threads)
                }
            })

        elif path == "/metrics":
            active_count = len(task_processor.active_tasks)
            completed_count = len(task_processor.completed_tasks)
            queue_size = task_processor.task_queue.qsize()

            metrics_data = f"""# HELP xorb_worker_tasks_active Currently active tasks
# TYPE xorb_worker_tasks_active gauge
xorb_worker_tasks_active {active_count}

# HELP xorb_worker_tasks_completed_total Total completed tasks
# TYPE xorb_worker_tasks_completed_total counter
xorb_worker_tasks_completed_total {completed_count}

# HELP xorb_worker_queue_size Current queue size
# TYPE xorb_worker_queue_size gauge
xorb_worker_queue_size {queue_size}

# HELP xorb_worker_uptime_seconds Worker uptime in seconds
# TYPE xorb_worker_uptime_seconds gauge
xorb_worker_uptime_seconds {time.time() - start_time}

# HELP xorb_worker_health Worker health status
# TYPE xorb_worker_health gauge
xorb_worker_health 1
"""
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(metrics_data.encode())

        elif path == "/api/v1/worker/status":
            self.send_json_response({
                "worker": {
                    "status": "operational",
                    "version": "2.0.0",
                    "environment": os.getenv("ENVIRONMENT", "production")
                },
                "performance": {
                    "active_tasks": len(task_processor.active_tasks),
                    "completed_tasks": len(task_processor.completed_tasks),
                    "queue_size": task_processor.task_queue.qsize(),
                    "worker_threads": len(task_processor.worker_threads),
                    "processing_capacity": "HIGH",
                    "avg_task_time": "2.5s"
                },
                "capabilities": {
                    "reconnaissance": True,
                    "vulnerability_scanning": True,
                    "threat_hunting": True,
                    "ai_analysis": True,
                    "distributed_processing": True,
                    "real_time_processing": True
                }
            })

        elif path.startswith("/api/v1/worker/tasks/"):
            task_id = path.split("/")[-1]
            task_status = task_processor.get_task_status(task_id)

            if task_status:
                self.send_json_response(task_status)
            else:
                self.send_json_response({"error": "Task not found"}, status=404)

        elif path == "/api/v1/worker/tasks":
            all_tasks = {
                "active_tasks": list(task_processor.active_tasks.values()),
                "completed_tasks": list(task_processor.completed_tasks.values())[-10:],  # Last 10
                "queue_size": task_processor.task_queue.qsize(),
                "total_active": len(task_processor.active_tasks),
                "total_completed": len(task_processor.completed_tasks)
            }
            self.send_json_response(all_tasks)

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/api/v1/worker/tasks":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                task_data = json.loads(post_data.decode('utf-8'))

                # Validate task data
                if "type" not in task_data:
                    self.send_json_response({"error": "Task type required"}, status=400)
                    return

                # Submit task for processing
                task_id = task_processor.submit_task(task_data)

                response = {
                    "message": "Task submitted successfully",
                    "task_id": task_id,
                    "task_type": task_data["type"],
                    "estimated_completion": time.time() + 30,  # 30 seconds estimate
                    "status_endpoint": f"/api/v1/worker/tasks/{task_id}"
                }

                self.send_json_response(response, status=201)

            except Exception as e:
                self.send_json_response({"error": str(e)}, status=400)

        else:
            self.send_error(404, "Endpoint not found")

    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WORKER: {format % args}")

# Global state
start_time = time.time()

def main():
    """Start the XORB Worker server"""
    global start_time
    start_time = time.time()

    # Start task processor workers
    task_processor.start_workers(num_workers=4)

    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "9000"))

    server = HTTPServer((host, port), XORBWorkerHandler)

    print(f"⚡ XORB Worker Service starting on {host}:{port}")
    print(f"📊 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print("🔧 Worker threads: 4")
    print(f"📋 Health check: http://{host}:{port}/health")
    print(f"📈 Metrics: http://{host}:{port}/metrics")
    print(f"⚙️  Task management: http://{host}:{port}/api/v1/worker/tasks")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("🛑 Shutting down XORB Worker service...")
        task_processor.running = False
        server.shutdown()

if __name__ == "__main__":
    main()
