#!/usr/bin/env python3
"""
Simple XORB Orchestrator Service
Campaign management and agent orchestration
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


class XORBOrchestratorHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/":
            self.send_json_response({
                "service": "XORB Orchestrator Service",
                "version": "2.0.0",
                "status": "operational",
                "message": "Campaign management and agent orchestration",
                "capabilities": [
                    "campaign_orchestration",
                    "agent_coordination",
                    "workflow_management",
                    "resource_optimization"
                ]
            })

        elif path == "/health":
            self.send_json_response({
                "status": "healthy",
                "service": "xorb-orchestrator",
                "version": "2.0.0",
                "timestamp": time.time(),
                "uptime": time.time() - start_time,
                "active_campaigns": len(active_campaigns),
                "total_executions": execution_count
            })

        elif path == "/metrics":
            metrics_data = f"""# HELP xorb_orchestrator_campaigns_total Total campaigns managed
# TYPE xorb_orchestrator_campaigns_total counter
xorb_orchestrator_campaigns_total {len(active_campaigns)}

# HELP xorb_orchestrator_executions_total Total agent executions
# TYPE xorb_orchestrator_executions_total counter
xorb_orchestrator_executions_total {execution_count}

# HELP xorb_orchestrator_uptime_seconds Orchestrator uptime in seconds
# TYPE xorb_orchestrator_uptime_seconds gauge
xorb_orchestrator_uptime_seconds {time.time() - start_time}

# HELP xorb_orchestrator_health Orchestrator health status
# TYPE xorb_orchestrator_health gauge
xorb_orchestrator_health 1
"""
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(metrics_data.encode())

        elif path == "/api/v1/orchestrator/status":
            self.send_json_response({
                "orchestrator": {
                    "status": "operational",
                    "version": "2.0.0",
                    "environment": os.getenv("ENVIRONMENT", "production")
                },
                "performance": {
                    "max_concurrent_agents": 32,
                    "active_campaigns": len(active_campaigns),
                    "total_executions": execution_count,
                    "avg_execution_time": "0.75s",
                    "success_rate": "98.5%"
                },
                "resources": {
                    "cpu_usage": "15%",
                    "memory_usage": "512MB",
                    "available_agents": 3,
                    "queue_depth": 0
                }
            })

        elif path == "/api/v1/orchestrator/campaigns":
            self.send_json_response({
                "campaigns": list(active_campaigns.values()),
                "total_count": len(active_campaigns),
                "running_count": len([c for c in active_campaigns.values() if c["status"] == "running"]),
                "completed_count": len([c for c in active_campaigns.values() if c["status"] == "completed"])
            })

        elif path == "/api/v1/orchestrator/agents":
            self.send_json_response({
                "available_agents": [
                    {
                        "id": "orchestrator-recon-001",
                        "name": "Orchestrated ReconAgent",
                        "type": "reconnaissance",
                        "status": "idle",
                        "current_campaign": None,
                        "executions_today": 15,
                        "success_rate": "99.2%"
                    },
                    {
                        "id": "orchestrator-vuln-001",
                        "name": "Orchestrated VulnAgent",
                        "type": "vulnerability_assessment",
                        "status": "busy",
                        "current_campaign": "campaign-002",
                        "executions_today": 8,
                        "success_rate": "97.8%"
                    },
                    {
                        "id": "orchestrator-threat-001",
                        "name": "Orchestrated ThreatHuntAgent",
                        "type": "threat_hunting",
                        "status": "idle",
                        "current_campaign": None,
                        "executions_today": 22,
                        "success_rate": "98.9%"
                    }
                ],
                "agent_pool_status": "optimal",
                "load_balancing": "active"
            })

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/api/v1/orchestrator/campaigns":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                campaign_data = json.loads(post_data.decode('utf-8'))
                campaign_id = f"orchestrator-campaign-{int(time.time())}"

                new_campaign = {
                    "id": campaign_id,
                    "name": campaign_data.get("name", "Orchestrated Campaign"),
                    "description": campaign_data.get("description", ""),
                    "status": "pending",
                    "created_at": time.time(),
                    "targets": campaign_data.get("targets", []),
                    "agent_requirements": campaign_data.get("agent_requirements", []),
                    "orchestration_config": {
                        "parallel_execution": True,
                        "max_retries": 3,
                        "timeout": 3600,
                        "priority": "normal"
                    }
                }

                active_campaigns[campaign_id] = new_campaign

                # Simulate campaign start
                threading.Thread(target=self.simulate_campaign_execution, args=(campaign_id,)).start()

                response = {
                    "message": "Campaign orchestrated successfully",
                    "campaign": new_campaign,
                    "orchestration_id": campaign_id
                }
                self.send_json_response(response, status=201)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=400)
        else:
            self.send_error(404, "Endpoint not found")

    def simulate_campaign_execution(self, campaign_id):
        """Simulate campaign execution in background"""
        global execution_count

        if campaign_id not in active_campaigns:
            return

        campaign = active_campaigns[campaign_id]
        campaign["status"] = "running"
        campaign["started_at"] = time.time()

        # Simulate execution phases
        phases = ["initialization", "agent_deployment", "execution", "analysis", "completion"]

        for i, phase in enumerate(phases):
            time.sleep(2)  # Simulate work
            campaign["current_phase"] = phase
            campaign["progress"] = int((i + 1) / len(phases) * 100)
            execution_count += 1

        campaign["status"] = "completed"
        campaign["completed_at"] = time.time()
        campaign["execution_time"] = campaign["completed_at"] - campaign["started_at"]
        campaign["findings"] = 12 + (len(campaign["targets"]) * 3)

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
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ORCHESTRATOR: {format % args}")

# Global state
start_time = time.time()
execution_count = 0
active_campaigns = {
    "campaign-demo-001": {
        "id": "campaign-demo-001",
        "name": "Production Demo Campaign",
        "status": "completed",
        "created_at": time.time() - 7200,
        "completed_at": time.time() - 5400,
        "targets": ["demo.xorb.local", "test.xorb.local"],
        "findings": 18,
        "execution_time": 1800
    }
}

def main():
    """Start the XORB Orchestrator server"""
    global start_time
    start_time = time.time()

    host = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
    port = int(os.getenv("ORCHESTRATOR_PORT", "8080"))

    server = HTTPServer((host, port), XORBOrchestratorHandler)

    print(f"🎭 XORB Orchestrator Service starting on {host}:{port}")
    print(f"📊 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print("🤖 Max concurrent agents: 32")
    print(f"📋 Health check: http://{host}:{port}/health")
    print(f"📈 Metrics: http://{host}:{port}/metrics")
    print(f"🎯 Campaign management: http://{host}:{port}/api/v1/orchestrator/campaigns")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("🛑 Shutting down XORB Orchestrator service...")
        server.shutdown()

if __name__ == "__main__":
    main()
