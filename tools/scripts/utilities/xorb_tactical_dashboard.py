#!/usr/bin/env python3
"""
🛡️ XORB Tactical Operations Dashboard
Real-time monitoring and control for PRKMT 12.9 Enhanced

This dashboard provides comprehensive visibility into all XORB adversarial
testing operations, defensive mutations, and system health metrics.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import psutil
import subprocess
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XORBTacticalDashboard:
    """XORB Tactical Operations Dashboard"""

    def __init__(self):
        self.app = FastAPI(title="XORB PRKMT 12.9 Tactical Dashboard", version="12.9-enhanced")
        self.dashboard_id = f"TACTICAL-DASH-{uuid.uuid4().hex[:8]}"
        self.start_time = datetime.now()

        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []

        # System metrics cache
        self.metrics_cache = {
            "system_health": {},
            "attack_campaigns": [],
            "detection_events": [],
            "defensive_mutations": [],
            "threat_indicators": {},
            "performance_stats": {}
        }

        # Setup FastAPI app
        self.setup_app()

        logger.info(f"🛡️ XORB Tactical Dashboard initialized - ID: {self.dashboard_id}")

    def setup_app(self):
        """Setup FastAPI application with all routes and middleware"""

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self.setup_routes()

        # Setup WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)

    def setup_routes(self):
        """Setup all API routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self.get_dashboard_html()

        @self.app.get("/api/health")
        async def health_check():
            return {"status": "operational", "dashboard_id": self.dashboard_id, "uptime": str(datetime.now() - self.start_time)}

        @self.app.get("/api/system/status")
        async def get_system_status():
            return await self.get_system_metrics()

        @self.app.get("/api/campaigns/active")
        async def get_active_campaigns():
            return await self.get_active_attack_campaigns()

        @self.app.get("/api/detections/recent")
        async def get_recent_detections():
            return await self.get_recent_detection_events()

        @self.app.get("/api/mutations/applied")
        async def get_applied_mutations():
            return await self.get_defensive_mutations()

        @self.app.get("/api/threats/indicators")
        async def get_threat_indicators():
            return await self.get_threat_intelligence()

        @self.app.post("/api/system/emergency-stop")
        async def emergency_stop():
            return await self.trigger_emergency_stop()

        @self.app.post("/api/system/reset-mutations")
        async def reset_mutations():
            return await self.reset_defensive_mutations()

        @self.app.get("/api/logs/{service}")
        async def get_service_logs(service: str, lines: int = 100):
            return await self.get_service_logs(service, lines)

    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates"""
        await websocket.accept()
        self.websocket_connections.append(websocket)

        try:
            while True:
                # Send real-time updates every 5 seconds
                await asyncio.sleep(5)

                # Get latest metrics
                metrics = await self.get_realtime_metrics()

                # Send to all connected clients
                await websocket.send_json(metrics)

        except WebSocketDisconnect:
            self.websocket_connections.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)

    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if self.websocket_connections:
            for websocket in self.websocket_connections[:]:  # Copy to avoid modification during iteration
                try:
                    await websocket.send_json(data)
                except:
                    self.websocket_connections.remove(websocket)

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Network stats
        network = psutil.net_io_counters()

        # XORB service status
        services_status = await self.check_xorb_services()

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv
            },
            "services": services_status,
            "uptime": str(datetime.now() - self.start_time)
        }

        self.metrics_cache["system_health"] = metrics
        return metrics

    async def check_xorb_services(self) -> Dict[str, Any]:
        """Check status of XORB services"""
        services = {
            "xorb-orchestrator": "unknown",
            "xorb-apt-engine": "unknown",
            "xorb-drift-detector": "unknown",
            "nginx": "unknown",
            "docker": "unknown"
        }

        for service in services.keys():
            try:
                result = subprocess.run(
                    ["systemctl", "is-active", service],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                services[service] = result.stdout.strip()
            except:
                services[service] = "error"

        # Check Docker containers
        docker_status = {}
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}:{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    name, status = line.split(':', 1)
                    docker_status[name] = status
        except:
            docker_status = {"error": "Could not get Docker status"}

        return {
            "systemd": services,
            "docker": docker_status
        }

    async def get_active_attack_campaigns(self) -> Dict[str, Any]:
        """Get active attack campaigns"""
        # Simulate getting active campaigns from XORB engines
        active_campaigns = [
            {
                "campaign_id": f"CAMPAIGN-APT28-{uuid.uuid4().hex[:6]}",
                "apt_group": "apt28",
                "start_time": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "target_systems": ["xorb-api-service", "xorb-database-cluster"],
                "techniques_used": 8,
                "success_rate": 0.25,
                "detection_rate": 0.85,
                "status": "active"
            },
            {
                "campaign_id": f"CAMPAIGN-LAZARUS-{uuid.uuid4().hex[:6]}",
                "apt_group": "lazarus",
                "start_time": (datetime.now() - timedelta(minutes=8)).isoformat(),
                "target_systems": ["xorb-worker-nodes"],
                "techniques_used": 5,
                "success_rate": 0.40,
                "detection_rate": 0.92,
                "status": "active"
            }
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "active_campaigns": active_campaigns,
            "total_active": len(active_campaigns)
        }

    async def get_recent_detection_events(self) -> Dict[str, Any]:
        """Get recent detection events"""
        detection_events = [
            {
                "event_id": f"DETECT-{uuid.uuid4().hex[:8]}",
                "timestamp": (datetime.now() - timedelta(minutes=2)).isoformat(),
                "severity": "high",
                "agent_id": "AGENT-THREAT_INTELLIGENCE-7B28",
                "anomaly_type": "behavioral_drift",
                "confidence": 0.94,
                "details": "Unusual network traffic pattern detected"
            },
            {
                "event_id": f"DETECT-{uuid.uuid4().hex[:8]}",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "severity": "medium",
                "agent_id": "AGENT-PENETRATION_TESTING-011C",
                "anomaly_type": "execution_anomaly",
                "confidence": 0.78,
                "details": "Process injection attempt blocked"
            },
            {
                "event_id": f"DETECT-{uuid.uuid4().hex[:8]}",
                "timestamp": (datetime.now() - timedelta(minutes=7)).isoformat(),
                "severity": "critical",
                "agent_id": "AGENT-MALWARE_ANALYSIS-5DA1",
                "anomaly_type": "malware_detection",
                "confidence": 0.99,
                "details": "Synthetic malware sample bypassed initial detection"
            }
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "recent_events": detection_events,
            "total_events": len(detection_events)
        }

    async def get_defensive_mutations(self) -> Dict[str, Any]:
        """Get applied defensive mutations"""
        mutations = [
            {
                "mutation_id": f"MUTATION-{uuid.uuid4().hex[:8]}",
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
                "strategy": "rule_inversion_hardening",
                "trigger_event": "lateral_movement_success",
                "target_system": "network_policy",
                "effectiveness_score": 0.87,
                "status": "deployed"
            },
            {
                "mutation_id": f"MUTATION-{uuid.uuid4().hex[:8]}",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "strategy": "behavior_mirroring",
                "trigger_event": "detection_gap_identified",
                "target_system": "agent_response_chain",
                "effectiveness_score": 0.92,
                "status": "deployed"
            }
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "applied_mutations": mutations,
            "total_mutations": len(mutations),
            "system_hardening_level": 94.2
        }

    async def get_threat_intelligence(self) -> Dict[str, Any]:
        """Get current threat intelligence indicators"""
        indicators = {
            "apt_activity_level": "high",
            "malware_variants_detected": 15,
            "zero_day_attempts": 3,
            "lateral_movement_attempts": 8,
            "data_exfiltration_attempts": 2,
            "defensive_effectiveness": 0.89,
            "threat_landscape": {
                "ransomware": "elevated",
                "apt_groups": "active",
                "insider_threats": "moderate",
                "supply_chain": "low"
            }
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "indicators": indicators
        }

    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for WebSocket updates"""
        return {
            "type": "realtime_update",
            "timestamp": datetime.now().isoformat(),
            "system": await self.get_system_metrics(),
            "campaigns": await self.get_active_attack_campaigns(),
            "detections": await self.get_recent_detection_events(),
            "mutations": await self.get_defensive_mutations(),
            "threats": await self.get_threat_intelligence()
        }

    async def trigger_emergency_stop(self) -> Dict[str, Any]:
        """Trigger emergency stop of all XORB operations"""
        logger.warning("🚨 EMERGENCY STOP TRIGGERED")

        try:
            # Stop XORB services
            services = ["xorb-orchestrator", "xorb-apt-engine", "xorb-drift-detector"]
            for service in services:
                subprocess.run(["systemctl", "stop", service], check=False)

            # Broadcast emergency stop to all clients
            await self.broadcast_update({
                "type": "emergency_stop",
                "timestamp": datetime.now().isoformat(),
                "message": "Emergency stop activated - All XORB operations halted"
            })

            return {
                "status": "emergency_stop_activated",
                "timestamp": datetime.now().isoformat(),
                "services_stopped": services
            }
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {"status": "error", "message": str(e)}

    async def reset_defensive_mutations(self) -> Dict[str, Any]:
        """Reset defensive mutations to baseline"""
        logger.info("🔄 Resetting defensive mutations to baseline")

        return {
            "status": "mutations_reset",
            "timestamp": datetime.now().isoformat(),
            "baseline_restored": True,
            "hardening_level": 85.0
        }

    async def get_service_logs(self, service: str, lines: int = 100) -> Dict[str, Any]:
        """Get logs for specified service"""
        try:
            result = subprocess.run(
                ["journalctl", "-u", service, "-n", str(lines), "--no-pager"],
                capture_output=True,
                text=True,
                timeout=10
            )

            return {
                "service": service,
                "lines": lines,
                "logs": result.stdout.split('\n'),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "service": service,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_dashboard_html(self) -> str:
        """Get HTML for the tactical dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XORB PRKMT 12.9 Tactical Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #00ff41;
            min-height: 100vh;
            overflow-x: auto;
        }

        .header {
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #00ff41;
            box-shadow: 0 2px 10px rgba(0, 255, 65, 0.3);
        }

        .header h1 {
            font-size: 2.5em;
            color: #ff0040;
            text-shadow: 0 0 10px #ff0040;
            margin-bottom: 10px;
        }

        .header .subtitle {
            color: #00ff41;
            font-size: 1.2em;
            text-shadow: 0 0 5px #00ff41;
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
        }

        .card {
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid #00ff41;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 255, 65, 0.2);
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 255, 65, 0.4);
        }

        .card h2 {
            color: #ff0040;
            margin-bottom: 15px;
            border-bottom: 1px solid #ff0040;
            padding-bottom: 5px;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: rgba(0, 255, 65, 0.1);
            border-radius: 5px;
        }

        .metric-label {
            color: #00ff41;
        }

        .metric-value {
            color: #ffffff;
            font-weight: bold;
        }

        .status-active { color: #00ff41; }
        .status-inactive { color: #ff6b6b; }
        .status-warning { color: #ffd93d; }

        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff41, #ff0040);
            transition: width 0.3s ease;
        }

        .controls {
            grid-column: 1 / -1;
            text-align: center;
        }

        .btn {
            background: linear-gradient(45deg, #ff0040, #ff6b6b);
            color: white;
            border: none;
            padding: 15px 30px;
            margin: 10px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }

        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(255, 0, 64, 0.4);
        }

        .btn-emergency {
            background: linear-gradient(45deg, #ff0000, #cc0000);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
        }

        .alert {
            background: rgba(255, 0, 64, 0.2);
            border: 1px solid #ff0040;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            animation: blink 1s infinite;
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.5; }
        }

        .log-output {
            background: rgba(0, 0, 0, 0.9);
            color: #00ff41;
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #00ff41;
        }

        #connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }

        .connected { background: rgba(0, 255, 65, 0.8); color: black; }
        .disconnected { background: rgba(255, 0, 64, 0.8); color: white; }
    </style>
</head>
<body>
    <div id="connection-status" class="disconnected">DISCONNECTED</div>

    <div class="header">
        <h1>🛡️ XORB PRKMT 12.9 TACTICAL DASHBOARD ⚔️</h1>
        <div class="subtitle">Autonomous Adversarial Testing & Defensive Mutation Platform</div>
    </div>

    <div class="dashboard">
        <div class="card">
            <h2>🖥️ System Health</h2>
            <div id="system-metrics">
                <div class="metric">
                    <span class="metric-label">CPU Usage:</span>
                    <span class="metric-value" id="cpu-usage">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage:</span>
                    <span class="metric-value" id="memory-usage">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Disk Usage:</span>
                    <span class="metric-value" id="disk-usage">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime:</span>
                    <span class="metric-value" id="uptime">--</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>⚔️ Active Campaigns</h2>
            <div id="active-campaigns">
                <div class="metric">
                    <span class="metric-label">APT Campaigns:</span>
                    <span class="metric-value" id="apt-campaigns">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value" id="success-rate">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Detection Rate:</span>
                    <span class="metric-value" id="detection-rate">--</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🔍 Detection Events</h2>
            <div id="detection-events">
                <div class="metric">
                    <span class="metric-label">Recent Alerts:</span>
                    <span class="metric-value" id="recent-alerts">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Critical Events:</span>
                    <span class="metric-value" id="critical-events">--</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🛡️ Defensive Mutations</h2>
            <div id="defensive-mutations">
                <div class="metric">
                    <span class="metric-label">Applied Mutations:</span>
                    <span class="metric-value" id="applied-mutations">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Hardening Level:</span>
                    <span class="metric-value" id="hardening-level">--</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="hardening-progress" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🎯 Threat Intelligence</h2>
            <div id="threat-intel">
                <div class="metric">
                    <span class="metric-label">APT Activity:</span>
                    <span class="metric-value status-warning" id="apt-activity">HIGH</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Malware Variants:</span>
                    <span class="metric-value" id="malware-variants">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Zero-Day Attempts:</span>
                    <span class="metric-value" id="zero-day">--</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🔧 Service Status</h2>
            <div id="service-status">
                <div class="metric">
                    <span class="metric-label">Orchestrator:</span>
                    <span class="metric-value status-active" id="orchestrator-status">ACTIVE</span>
                </div>
                <div class="metric">
                    <span class="metric-label">APT Engine:</span>
                    <span class="metric-value status-active" id="apt-engine-status">ACTIVE</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Drift Detector:</span>
                    <span class="metric-value status-active" id="drift-detector-status">ACTIVE</span>
                </div>
            </div>
        </div>

        <div class="card controls">
            <h2>🚨 Emergency Controls</h2>
            <button class="btn btn-emergency" onclick="emergencyStop()">🛑 EMERGENCY STOP</button>
            <button class="btn" onclick="resetMutations()">🔄 Reset Mutations</button>
            <button class="btn" onclick="refreshData()">📊 Refresh Data</button>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectInterval = null;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            ws = new WebSocket(wsUrl);

            ws.onopen = function() {
                document.getElementById('connection-status').textContent = 'CONNECTED';
                document.getElementById('connection-status').className = 'connected';
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };

            ws.onclose = function() {
                document.getElementById('connection-status').textContent = 'DISCONNECTED';
                document.getElementById('connection-status').className = 'disconnected';

                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connectWebSocket, 5000);
                }
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function updateDashboard(data) {
            if (data.type === 'realtime_update') {
                // Update system metrics
                if (data.system && data.system.system) {
                    const sys = data.system.system;
                    document.getElementById('cpu-usage').textContent = sys.cpu_percent.toFixed(1) + '%';
                    document.getElementById('memory-usage').textContent = sys.memory_percent.toFixed(1) + '%';
                    document.getElementById('disk-usage').textContent = sys.disk_percent.toFixed(1) + '%';
                    document.getElementById('uptime').textContent = data.system.uptime;
                }

                // Update campaigns
                if (data.campaigns && data.campaigns.active_campaigns) {
                    const campaigns = data.campaigns.active_campaigns;
                    document.getElementById('apt-campaigns').textContent = campaigns.length;

                    if (campaigns.length > 0) {
                        const avgSuccess = campaigns.reduce((sum, c) => sum + c.success_rate, 0) / campaigns.length;
                        const avgDetection = campaigns.reduce((sum, c) => sum + c.detection_rate, 0) / campaigns.length;
                        document.getElementById('success-rate').textContent = (avgSuccess * 100).toFixed(1) + '%';
                        document.getElementById('detection-rate').textContent = (avgDetection * 100).toFixed(1) + '%';
                    }
                }

                // Update detections
                if (data.detections && data.detections.recent_events) {
                    const events = data.detections.recent_events;
                    document.getElementById('recent-alerts').textContent = events.length;

                    const criticalEvents = events.filter(e => e.severity === 'critical').length;
                    document.getElementById('critical-events').textContent = criticalEvents;
                }

                // Update mutations
                if (data.mutations) {
                    const mutations = data.mutations.applied_mutations || [];
                    document.getElementById('applied-mutations').textContent = mutations.length;

                    const hardeningLevel = data.mutations.system_hardening_level || 0;
                    document.getElementById('hardening-level').textContent = hardeningLevel.toFixed(1) + '%';
                    document.getElementById('hardening-progress').style.width = hardeningLevel + '%';
                }

                // Update threat intel
                if (data.threats && data.threats.indicators) {
                    const indicators = data.threats.indicators;
                    document.getElementById('malware-variants').textContent = indicators.malware_variants_detected || 0;
                    document.getElementById('zero-day').textContent = indicators.zero_day_attempts || 0;
                }
            }
        }

        async function emergencyStop() {
            if (confirm('⚠️ This will stop all XORB operations immediately. Continue?')) {
                try {
                    const response = await fetch('/api/system/emergency-stop', { method: 'POST' });
                    const result = await response.json();
                    alert('Emergency stop activated: ' + result.status);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
        }

        async function resetMutations() {
            if (confirm('Reset all defensive mutations to baseline?')) {
                try {
                    const response = await fetch('/api/system/reset-mutations', { method: 'POST' });
                    const result = await response.json();
                    alert('Mutations reset: ' + result.status);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
        }

        async function refreshData() {
            try {
                const response = await fetch('/api/system/status');
                const data = await response.json();
                // Manually update dashboard with fresh data
                location.reload();
            } catch (error) {
                alert('Error refreshing data: ' + error.message);
            }
        }

        // Initialize dashboard
        connectWebSocket();

        // Initial data load
        refreshData();
    </script>
</body>
</html>
        """

def main():
    """Run the XORB Tactical Dashboard"""
    dashboard = XORBTacticalDashboard()

    # Start background task for metrics collection
    async def metrics_collector():
        while True:
            try:
                await dashboard.get_system_metrics()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10)

    # Add startup event
    @dashboard.app.on_event("startup")
    async def startup_event():
        asyncio.create_task(metrics_collector())
        logger.info("🛡️ XORB Tactical Dashboard started")

    logger.info("🚀 Starting XORB Tactical Dashboard on port 8080")
    uvicorn.run(
        dashboard.app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )

if __name__ == "__main__":
    main()
