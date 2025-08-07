#!/usr/bin/env python3
"""
XORB Unified Orchestration Hub
Central command and control system integrating all XORB services
"""

import asyncio
import json
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="XORB Unified Orchestration Hub",
    description="Central command and control for all XORB cybersecurity services",
    version="10.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    OFFLINE = "offline"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class XORBService:
    service_id: str
    name: str
    port: int
    health_endpoint: str
    dashboard_endpoint: str
    status: ServiceStatus
    last_health_check: datetime
    uptime: float
    response_time: float
    capabilities: List[str]
    version: str

class UnifiedAlert(BaseModel):
    alert_id: str
    source_service: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: str
    status: str
    metadata: Dict = {}

class OrchestrationTask(BaseModel):
    task_id: str
    task_type: str
    description: str
    services_involved: List[str]
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict] = None

class SystemMetrics(BaseModel):
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict
    active_connections: int
    response_times: Dict[str, float]

class UnifiedOrchestrationHub:
    """Central orchestration system for all XORB services"""
    
    def __init__(self):
        self.services: Dict[str, XORBService] = {}
        self.alerts: List[UnifiedAlert] = []
        self.orchestration_tasks: List[OrchestrationTask] = []
        self.system_metrics: List[SystemMetrics] = []
        self.websocket_connections: List[WebSocket] = []
        self.monitoring_task = None
        
        # Initialize XORB services registry
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize registry of all XORB services"""
        services_config = [
            {
                "service_id": "threat_intelligence",
                "name": "Threat Intelligence Engine",
                "port": 9004,
                "health_endpoint": "/health",
                "dashboard_endpoint": "/threat-intelligence/real-time-demo",
                "capabilities": ["Real-time Threat Feeds", "WebSocket Streaming", "Global Threat Monitoring"]
            },
            {
                "service_id": "collaboration",
                "name": "Real-time Collaboration Platform",
                "port": 9005,
                "health_endpoint": "/health",
                "dashboard_endpoint": "/collaboration/demo",
                "capabilities": ["Multi-analyst Collaboration", "Threat Annotations", "Incident Coordination"]
            },
            {
                "service_id": "analytics",
                "name": "Advanced Analytics Engine",
                "port": 9006,
                "health_endpoint": "/health",
                "dashboard_endpoint": "/analytics/dashboard",
                "capabilities": ["ML-powered Analysis", "Anomaly Detection", "Behavioral Analysis"]
            },
            {
                "service_id": "reporting",
                "name": "Enterprise Reporting System",
                "port": 9007,
                "health_endpoint": "/health",
                "dashboard_endpoint": "/reports/dashboard",
                "capabilities": ["Executive Summaries", "Compliance Reports", "Threat Analysis Reports"]
            },
            {
                "service_id": "hardening",
                "name": "Security Hardening System",
                "port": 9008,
                "health_endpoint": "/health",
                "dashboard_endpoint": "/hardening/dashboard",
                "capabilities": ["Zero-trust Security", "Automated Hardening", "Compliance Assessment"]
            },
            {
                "service_id": "mobile",
                "name": "Mobile-Responsive Interface",
                "port": 9009,
                "health_endpoint": "/health",
                "dashboard_endpoint": "/",
                "capabilities": ["Progressive Web App", "Mobile-first Design", "Offline Support"]
            },
            {
                "service_id": "nvidia_ai",
                "name": "NVIDIA AI Integration",
                "port": 9010,
                "health_endpoint": "/health",
                "dashboard_endpoint": "/",
                "capabilities": ["Advanced AI Analysis", "Qwen3-235B Model", "Streaming Responses"]
            }
        ]
        
        for config in services_config:
            service = XORBService(
                service_id=config["service_id"],
                name=config["name"],
                port=config["port"],
                health_endpoint=config["health_endpoint"],
                dashboard_endpoint=config["dashboard_endpoint"],
                status=ServiceStatus.OFFLINE,
                last_health_check=datetime.now(),
                uptime=0.0,
                response_time=0.0,
                capabilities=config["capabilities"],
                version="unknown"
            )
            self.services[service.service_id] = service
    
    async def _start_monitoring_loop(self):
        """Start continuous monitoring of all services"""
        while True:
            await self._check_all_services_health()
            await self._collect_system_metrics()
            await self._process_alerts()
            await self._broadcast_status_updates()
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _check_all_services_health(self):
        """Check health of all registered services"""
        tasks = []
        for service in self.services.values():
            tasks.append(self._check_service_health(service))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service: XORBService):
        """Check health of individual service"""
        start_time = time.time()
        
        try:
            url = f"http://localhost:{service.port}{service.health_endpoint}"
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        health_data = await response.json()
                        service.status = ServiceStatus.HEALTHY
                        service.response_time = response_time
                        service.version = health_data.get("version", "unknown")
                        
                        # Update uptime calculation
                        if hasattr(service, '_start_time'):
                            service.uptime = time.time() - service._start_time
                        else:
                            service._start_time = time.time()
                            service.uptime = 0.0
                    else:
                        service.status = ServiceStatus.ERROR
                        await self._create_alert(
                            service.service_id,
                            AlertSeverity.HIGH,
                            f"Service Health Check Failed",
                            f"{service.name} returned HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            service.status = ServiceStatus.DEGRADED
            service.response_time = 10.0
            await self._create_alert(
                service.service_id,
                AlertSeverity.MEDIUM,
                "Service Timeout",
                f"{service.name} health check timed out"
            )
        except Exception as e:
            service.status = ServiceStatus.OFFLINE
            service.response_time = 0.0
            await self._create_alert(
                service.service_id,
                AlertSeverity.CRITICAL,
                "Service Offline",
                f"{service.name} is not responding: {str(e)}"
            )
        
        service.last_health_check = datetime.now()
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # Simulate system metrics collection
            # In production, would integrate with system monitoring tools
            import random
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage=random.uniform(10, 80),
                memory_usage=random.uniform(20, 70),
                disk_usage=random.uniform(30, 60),
                network_io={
                    "bytes_sent": random.randint(1000000, 10000000),
                    "bytes_received": random.randint(5000000, 50000000)
                },
                active_connections=len(self.websocket_connections),
                response_times={
                    service.service_id: service.response_time
                    for service in self.services.values()
                }
            )
            
            self.system_metrics.append(metrics)
            
            # Keep only last 100 metrics entries
            if len(self.system_metrics) > 100:
                self.system_metrics = self.system_metrics[-100:]
                
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    async def _create_alert(self, source_service: str, severity: AlertSeverity, title: str, description: str):
        """Create new system alert"""
        alert = UnifiedAlert(
            alert_id=f"alert_{int(time.time())}_{len(self.alerts)}",
            source_service=source_service,
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now().isoformat(),
            status="active"
        )
        
        self.alerts.append(alert)
        
        # Keep only last 500 alerts
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]
        
        # Broadcast alert to connected WebSocket clients
        await self._broadcast_alert(alert)
    
    async def _process_alerts(self):
        """Process and potentially auto-resolve alerts"""
        current_time = datetime.now()
        
        for alert in self.alerts:
            if alert.status == "active":
                alert_time = datetime.fromisoformat(alert.timestamp)
                
                # Auto-resolve alerts older than 1 hour if service is now healthy
                if current_time - alert_time > timedelta(hours=1):
                    source_service = self.services.get(alert.source_service)
                    if source_service and source_service.status == ServiceStatus.HEALTHY:
                        alert.status = "resolved"
                        alert.metadata["auto_resolved"] = True
                        alert.metadata["resolved_at"] = current_time.isoformat()
    
    async def _broadcast_status_updates(self):
        """Broadcast status updates to connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        status_update = {
            "type": "status_update",
            "data": {
                "services": {
                    service.service_id: {
                        "status": service.status.value,
                        "response_time": round(service.response_time, 3),
                        "uptime": round(service.uptime, 2)
                    }
                    for service in self.services.values()
                },
                "system_health": self._calculate_system_health(),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Broadcast to all connected clients
        disconnected_clients = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(status_update))
            except:
                disconnected_clients.append(websocket)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_connections.remove(client)
    
    async def _broadcast_alert(self, alert: UnifiedAlert):
        """Broadcast alert to WebSocket clients"""
        if not self.websocket_connections:
            return
        
        alert_message = {
            "type": "alert",
            "data": alert.dict()
        }
        
        disconnected_clients = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(alert_message))
            except:
                disconnected_clients.append(websocket)
        
        for client in disconnected_clients:
            self.websocket_connections.remove(client)
    
    def _calculate_system_health(self) -> Dict:
        """Calculate overall system health score"""
        if not self.services:
            return {"score": 0, "status": "unknown"}
        
        healthy_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.HEALTHY)
        total_services = len(self.services)
        health_percentage = (healthy_services / total_services) * 100
        
        if health_percentage >= 90:
            status = "excellent"
        elif health_percentage >= 75:
            status = "good"
        elif health_percentage >= 50:
            status = "degraded"
        else:
            status = "critical"
        
        avg_response_time = sum(s.response_time for s in self.services.values()) / total_services
        
        return {
            "score": round(health_percentage, 1),
            "status": status,
            "healthy_services": healthy_services,
            "total_services": total_services,
            "average_response_time": round(avg_response_time, 3)
        }
    
    async def execute_orchestration_task(self, task_type: str, description: str, services: List[str], parameters: Dict = None) -> OrchestrationTask:
        """Execute cross-service orchestration task"""
        task_id = f"task_{int(time.time())}_{len(self.orchestration_tasks)}"
        
        task = OrchestrationTask(
            task_id=task_id,
            task_type=task_type,
            description=description,
            services_involved=services,
            status="running",
            created_at=datetime.now().isoformat()
        )
        
        self.orchestration_tasks.append(task)
        
        try:
            # Execute task based on type
            if task_type == "security_assessment":
                result = await self._execute_security_assessment(services, parameters or {})
            elif task_type == "incident_response":
                result = await self._execute_incident_response(services, parameters or {})
            elif task_type == "threat_analysis":
                result = await self._execute_threat_analysis(services, parameters or {})
            else:
                result = {"error": f"Unknown task type: {task_type}"}
            
            task.status = "completed"
            task.completed_at = datetime.now().isoformat()
            task.result = result
            
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
        
        return task
    
    async def _execute_security_assessment(self, services: List[str], parameters: Dict) -> Dict:
        """Execute comprehensive security assessment across services"""
        results = {}
        
        # Hardening assessment
        if "hardening" in services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"http://localhost:9008/hardening/assess") as response:
                        if response.status == 200:
                            results["hardening"] = await response.json()
            except Exception as e:
                results["hardening"] = {"error": str(e)}
        
        # Analytics assessment
        if "analytics" in services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"http://localhost:9006/analytics/anomaly-detection") as response:
                        if response.status == 200:
                            results["analytics"] = await response.json()
            except Exception as e:
                results["analytics"] = {"error": str(e)}
        
        # Threat intelligence assessment
        if "threat_intelligence" in services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:9004/health") as response:
                        if response.status == 200:
                            results["threat_intelligence"] = await response.json()
            except Exception as e:
                results["threat_intelligence"] = {"error": str(e)}
        
        return {
            "assessment_type": "comprehensive_security",
            "services_assessed": len(results),
            "results": results,
            "overall_score": self._calculate_system_health()["score"]
        }
    
    async def _execute_incident_response(self, services: List[str], parameters: Dict) -> Dict:
        """Execute coordinated incident response across services"""
        incident_id = parameters.get("incident_id", f"inc_{int(time.time())}")
        
        # Create collaboration session for incident
        collaboration_result = {}
        if "collaboration" in services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://localhost:9005/collaboration/sessions",
                        params={
                            "session_name": f"Incident Response - {incident_id}",
                            "session_type": "incident_response",
                            "created_by": "orchestration_hub",
                            "incident_id": incident_id
                        }
                    ) as response:
                        if response.status == 200:
                            collaboration_result = await response.json()
            except Exception as e:
                collaboration_result = {"error": str(e)}
        
        # Generate incident report
        report_result = {}
        if "reporting" in services:
            try:
                async with aiohttp.ClientSession() as session:
                    report_request = {
                        "report_type": "incident_response",
                        "time_range": "24h",
                        "format": "json",
                        "filters": {"incident_id": incident_id}
                    }
                    async with session.post(
                        f"http://localhost:9007/reports/generate",
                        json=report_request
                    ) as response:
                        if response.status == 200:
                            report_result = await response.json()
            except Exception as e:
                report_result = {"error": str(e)}
        
        return {
            "incident_id": incident_id,
            "response_type": "coordinated_incident_response",
            "collaboration_session": collaboration_result,
            "incident_report": report_result,
            "services_involved": services
        }
    
    async def _execute_threat_analysis(self, services: List[str], parameters: Dict) -> Dict:
        """Execute coordinated threat analysis across services"""
        threat_data = parameters.get("threat_data", "Unknown threat indicators")
        
        # AI-powered analysis
        ai_result = {}
        if "nvidia_ai" in services:
            try:
                async with aiohttp.ClientSession() as session:
                    ai_request = {
                        "task_type": "threat_analysis",
                        "query": threat_data,
                        "context": parameters.get("context", {}),
                        "temperature": 0.7,
                        "max_tokens": 2048
                    }
                    async with session.post(
                        f"http://localhost:9010/ai/analyze",
                        json=ai_request
                    ) as response:
                        if response.status == 200:
                            ai_result = await response.json()
            except Exception as e:
                ai_result = {"error": str(e)}
        
        # Analytics correlation
        analytics_result = {}
        if "analytics" in services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"http://localhost:9006/analytics/threat-prediction") as response:
                        if response.status == 200:
                            analytics_result = await response.json()
            except Exception as e:
                analytics_result = {"error": str(e)}
        
        return {
            "threat_analysis_type": "coordinated_multi_service",
            "ai_analysis": ai_result,
            "analytics_correlation": analytics_result,
            "threat_data": threat_data,
            "services_used": services
        }

# Initialize orchestration hub
orchestration_hub = UnifiedOrchestrationHub()

@app.on_event("startup")
async def startup_event():
    """Start monitoring when FastAPI starts"""
    orchestration_hub.monitoring_task = asyncio.create_task(orchestration_hub._start_monitoring_loop())

@app.websocket("/ws/orchestration")
async def websocket_orchestration(websocket: WebSocket):
    """WebSocket endpoint for real-time orchestration updates"""
    await websocket.accept()
    orchestration_hub.websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        if websocket in orchestration_hub.websocket_connections:
            orchestration_hub.websocket_connections.remove(websocket)

@app.get("/orchestration/status")
async def get_orchestration_status():
    """Get overall orchestration status"""
    return {
        "system_health": orchestration_hub._calculate_system_health(),
        "services": {
            service.service_id: {
                "name": service.name,
                "status": service.status.value,
                "port": service.port,
                "response_time": round(service.response_time, 3),
                "uptime": round(service.uptime, 2),
                "version": service.version,
                "last_check": service.last_health_check.isoformat()
            }
            for service in orchestration_hub.services.values()
        },
        "active_alerts": len([a for a in orchestration_hub.alerts if a.status == "active"]),
        "total_tasks": len(orchestration_hub.orchestration_tasks),
        "websocket_connections": len(orchestration_hub.websocket_connections)
    }

@app.get("/orchestration/services")
async def get_services():
    """Get all registered services"""
    return {
        "total_services": len(orchestration_hub.services),
        "services": [
            {
                "service_id": service.service_id,
                "name": service.name,
                "port": service.port,
                "status": service.status.value,
                "capabilities": service.capabilities,
                "dashboard_url": f"http://188.245.101.102:{service.port}{service.dashboard_endpoint}",
                "health_url": f"http://localhost:{service.port}{service.health_endpoint}"
            }
            for service in orchestration_hub.services.values()
        ]
    }

@app.get("/orchestration/alerts")
async def get_alerts(active_only: bool = False, limit: int = 50):
    """Get system alerts"""
    alerts = orchestration_hub.alerts
    
    if active_only:
        alerts = [a for a in alerts if a.status == "active"]
    
    alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    return {
        "total_alerts": len(orchestration_hub.alerts),
        "active_alerts": len([a for a in orchestration_hub.alerts if a.status == "active"]),
        "alerts": [alert.dict() for alert in alerts]
    }

@app.post("/orchestration/tasks/execute")
async def execute_task(task_type: str, description: str, services: List[str], parameters: Dict = None):
    """Execute orchestration task across multiple services"""
    task = await orchestration_hub.execute_orchestration_task(task_type, description, services, parameters)
    return task.dict()

@app.get("/orchestration/tasks")
async def get_tasks(limit: int = 20):
    """Get orchestration tasks"""
    recent_tasks = orchestration_hub.orchestration_tasks[-limit:]
    return {
        "total_tasks": len(orchestration_hub.orchestration_tasks),
        "tasks": [task.dict() for task in recent_tasks]
    }

@app.get("/orchestration/metrics")
async def get_system_metrics(limit: int = 50):
    """Get system metrics"""
    recent_metrics = orchestration_hub.system_metrics[-limit:]
    return {
        "total_metrics": len(orchestration_hub.system_metrics),
        "metrics": [metric.dict() for metric in recent_metrics]
    }

@app.get("/", response_class=HTMLResponse)
async def orchestration_dashboard():
    """Unified Orchestration Dashboard"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>XORB Unified Orchestration Hub</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0d1117; color: #f0f6fc; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .orchestration-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .card-title { font-size: 1.2em; font-weight: 600; color: #58a6ff; }
        .system-health { text-align: center; margin: 20px 0; }
        .health-score { font-size: 3em; font-weight: bold; margin: 10px 0; }
        .health-excellent { color: #2ea043; }
        .health-good { color: #d29922; }
        .health-degraded { color: #f85149; }
        .services-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
        .service-item { background: #0d1117; padding: 15px; border-radius: 6px; text-align: center; border-left: 4px solid #58a6ff; }
        .service-healthy { border-left-color: #2ea043; }
        .service-degraded { border-left-color: #d29922; }
        .service-error { border-left-color: #f85149; }
        .service-offline { border-left-color: #6e7681; }
        .service-name { font-weight: 600; margin-bottom: 5px; }
        .service-status { font-size: 0.8em; text-transform: uppercase; margin-bottom: 5px; }
        .service-metrics { font-size: 0.7em; color: #8b949e; }
        .alert-list { max-height: 300px; overflow-y: auto; }
        .alert-item { background: #0d1117; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid #58a6ff; }
        .alert-critical { border-left-color: #f85149; }
        .alert-high { border-left-color: #d29922; }
        .alert-medium { border-left-color: #58a6ff; }
        .alert-low { border-left-color: #2ea043; }
        .alert-title { font-weight: 600; margin-bottom: 5px; }
        .alert-description { font-size: 0.9em; color: #8b949e; margin-bottom: 5px; }
        .alert-meta { font-size: 0.8em; color: #6e7681; }
        .task-controls { display: flex; gap: 10px; margin: 15px 0; }
        .task-btn { background: #238636; border: none; color: white; padding: 8px 16px; border-radius: 6px; cursor: pointer; }
        .task-btn:hover { background: #2ea043; }
        .task-btn.secondary { background: #21262d; border: 1px solid #30363d; }
        .metrics-chart { background: #0d1117; padding: 15px; border-radius: 6px; margin-top: 15px; }
        .connection-status { padding: 4px 8px; border-radius: 12px; font-size: 0.8em; }
        .status-connected { background: #2ea043; color: white; }
        .status-disconnected { background: #f85149; color: white; }
        .loading { text-align: center; color: #8b949e; padding: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 XORB UNIFIED ORCHESTRATION HUB</h1>
        <p>Central Command & Control for All XORB Cybersecurity Services</p>
        <div id="connection-status" class="connection-status status-disconnected">Connecting...</div>
    </div>
    
    <div class="dashboard-grid">
        <!-- System Health Card -->
        <div class="orchestration-card">
            <div class="card-header">
                <span class="card-title">🏥 System Health</span>
            </div>
            <div class="system-health">
                <div class="health-score" id="health-score">-</div>
                <div id="health-status">Loading...</div>
                <div style="font-size: 0.9em; color: #8b949e; margin-top: 10px;">
                    <span id="healthy-services">-</span> of <span id="total-services">-</span> services healthy
                </div>
            </div>
        </div>
        
        <!-- Services Status Card -->
        <div class="orchestration-card">
            <div class="card-header">
                <span class="card-title">🔧 Services Status</span>
            </div>
            <div class="services-grid" id="services-grid">
                <div class="loading">Loading services...</div>
            </div>
        </div>
        
        <!-- Active Alerts Card -->
        <div class="orchestration-card">
            <div class="card-header">
                <span class="card-title">🚨 Active Alerts</span>
                <span id="alert-count">0</span>
            </div>
            <div class="alert-list" id="alert-list">
                <div class="loading">Loading alerts...</div>
            </div>
        </div>
        
        <!-- Orchestration Tasks Card -->
        <div class="orchestration-card">
            <div class="card-header">
                <span class="card-title">⚡ Orchestration Tasks</span>
            </div>
            <div class="task-controls">
                <button class="task-btn" onclick="executeSecurityAssessment()">🔍 Security Assessment</button>
                <button class="task-btn" onclick="executeIncidentResponse()">🚨 Incident Response</button>
                <button class="task-btn secondary" onclick="executeThreatAnalysis()">🎯 Threat Analysis</button>
            </div>
            <div id="task-status" style="color: #8b949e; font-size: 0.9em;"></div>
            <div id="recent-tasks" style="margin-top: 15px;">
                <div class="loading">Loading recent tasks...</div>
            </div>
        </div>
    </div>
    
    <!-- System Metrics Chart -->
    <div class="orchestration-card" style="margin-top: 20px;">
        <div class="card-header">
            <span class="card-title">📊 System Metrics</span>
        </div>
        <div class="metrics-chart" id="metrics-chart">
            <div class="loading">Loading system metrics...</div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let orchestrationData = null;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://188.245.101.102:9011/ws/orchestration`);
            
            ws.onopen = function() {
                document.getElementById('connection-status').textContent = '✅ Connected';
                document.getElementById('connection-status').className = 'connection-status status-connected';
                console.log('WebSocket connected to orchestration hub');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'status_update') {
                    updateSystemStatus(data.data);
                } else if (data.type === 'alert') {
                    addNewAlert(data.data);
                }
            };
            
            ws.onclose = function() {
                document.getElementById('connection-status').textContent = '❌ Disconnected';
                document.getElementById('connection-status').className = 'connection-status status-disconnected';
                console.log('WebSocket disconnected, attempting reconnect...');
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
            
            // Send heartbeat every 30 seconds
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
        }
        
        async function loadOrchestrationData() {
            try {
                // Load orchestration status
                const statusResponse = await fetch('/orchestration/status');
                orchestrationData = await statusResponse.json();
                
                updateSystemHealth(orchestrationData.system_health);
                updateServicesGrid(orchestrationData.services);
                
                // Load alerts
                const alertsResponse = await fetch('/orchestration/alerts?active_only=true&limit=10');
                const alertsData = await alertsResponse.json();
                updateAlertsList(alertsData.alerts);
                
                // Load recent tasks
                const tasksResponse = await fetch('/orchestration/tasks?limit=5');
                const tasksData = await tasksResponse.json();
                updateRecentTasks(tasksData.tasks);
                
                // Load system metrics
                const metricsResponse = await fetch('/orchestration/metrics?limit=20');
                const metricsData = await metricsResponse.json();
                updateMetricsChart(metricsData.metrics);
                
            } catch (error) {
                console.error('Error loading orchestration data:', error);
            }
        }
        
        function updateSystemHealth(health) {
            const scoreElement = document.getElementById('health-score');
            const statusElement = document.getElementById('health-status');
            
            scoreElement.textContent = health.score + '%';
            statusElement.textContent = health.status.toUpperCase();
            
            // Update color based on health
            scoreElement.className = 'health-score';
            if (health.score >= 90) {
                scoreElement.classList.add('health-excellent');
            } else if (health.score >= 75) {
                scoreElement.classList.add('health-good');
            } else {
                scoreElement.classList.add('health-degraded');
            }
            
            document.getElementById('healthy-services').textContent = health.healthy_services;
            document.getElementById('total-services').textContent = health.total_services;
        }
        
        function updateServicesGrid(services) {
            const grid = document.getElementById('services-grid');
            grid.innerHTML = '';
            
            Object.values(services).forEach(service => {
                const serviceDiv = document.createElement('div');
                serviceDiv.className = `service-item service-${service.status}`;
                
                serviceDiv.innerHTML = `
                    <div class="service-name">${service.name}</div>
                    <div class="service-status">${service.status}</div>
                    <div class="service-metrics">
                        Response: ${(service.response_time * 1000).toFixed(0)}ms<br>
                        Uptime: ${(service.uptime / 3600).toFixed(1)}h
                    </div>
                `;
                
                // Add click handler to open service dashboard
                serviceDiv.style.cursor = 'pointer';
                serviceDiv.onclick = () => {
                    const port = getServicePort(service.name);
                    if (port) {
                        window.open(`http://188.245.101.102:${port}`, '_blank');
                    }
                };
                
                grid.appendChild(serviceDiv);
            });
        }
        
        function getServicePort(serviceName) {
            const portMap = {
                'Threat Intelligence Engine': 9004,
                'Real-time Collaboration Platform': 9005,
                'Advanced Analytics Engine': 9006,
                'Enterprise Reporting System': 9007,
                'Security Hardening System': 9008,
                'Mobile-Responsive Interface': 9009,
                'NVIDIA AI Integration': 9010
            };
            return portMap[serviceName];
        }
        
        function updateAlertsList(alerts) {
            const alertList = document.getElementById('alert-list');
            const alertCount = document.getElementById('alert-count');
            
            alertCount.textContent = alerts.length;
            
            if (alerts.length === 0) {
                alertList.innerHTML = '<div class="loading">No active alerts</div>';
                return;
            }
            
            alertList.innerHTML = '';
            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert-item alert-${alert.severity}`;
                
                const timestamp = new Date(alert.timestamp).toLocaleString();
                
                alertDiv.innerHTML = `
                    <div class="alert-title">${alert.title}</div>
                    <div class="alert-description">${alert.description}</div>
                    <div class="alert-meta">
                        ${alert.source_service} • ${alert.severity.toUpperCase()} • ${timestamp}
                    </div>
                `;
                
                alertList.appendChild(alertDiv);
            });
        }
        
        function updateRecentTasks(tasks) {
            const container = document.getElementById('recent-tasks');
            
            if (tasks.length === 0) {
                container.innerHTML = '<div class="loading">No recent tasks</div>';
                return;
            }
            
            container.innerHTML = '';
            tasks.forEach(task => {
                const taskDiv = document.createElement('div');
                taskDiv.style.cssText = 'background: #0d1117; padding: 10px; margin: 5px 0; border-radius: 4px; font-size: 0.9em;';
                
                const createdAt = new Date(task.created_at).toLocaleString();
                const statusColor = task.status === 'completed' ? '#2ea043' : 
                                   task.status === 'failed' ? '#f85149' : '#d29922';
                
                taskDiv.innerHTML = `
                    <div style="font-weight: 600;">${task.task_type.replace('_', ' ').toUpperCase()}</div>
                    <div style="color: #8b949e; margin: 3px 0;">${task.description}</div>
                    <div style="color: ${statusColor}; font-size: 0.8em;">
                        ${task.status.toUpperCase()} • ${createdAt}
                    </div>
                `;
                
                container.appendChild(taskDiv);
            });
        }
        
        function updateMetricsChart(metrics) {
            const chart = document.getElementById('metrics-chart');
            
            if (metrics.length === 0) {
                chart.innerHTML = '<div class="loading">No metrics data</div>';
                return;
            }
            
            // Simple metrics display - in production would use proper charting library
            const latest = metrics[metrics.length - 1];
            
            chart.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 1.5em; color: #58a6ff;">${latest.cpu_usage.toFixed(1)}%</div>
                        <div style="color: #8b949e;">CPU Usage</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5em; color: #58a6ff;">${latest.memory_usage.toFixed(1)}%</div>
                        <div style="color: #8b949e;">Memory Usage</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5em; color: #58a6ff;">${latest.disk_usage.toFixed(1)}%</div>
                        <div style="color: #8b949e;">Disk Usage</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5em; color: #58a6ff;">${latest.active_connections}</div>
                        <div style="color: #8b949e;">Active Connections</div>
                    </div>
                </div>
            `;
        }
        
        function updateSystemStatus(data) {
            if (data.system_health) {
                updateSystemHealth(data.system_health);
            }
            if (data.services) {
                updateServicesGrid(data.services);
            }
        }
        
        function addNewAlert(alertData) {
            console.log('New alert received:', alertData);
            // Refresh alerts list
            setTimeout(() => {
                fetch('/orchestration/alerts?active_only=true&limit=10')
                    .then(r => r.json())
                    .then(data => updateAlertsList(data.alerts));
            }, 1000);
        }
        
        async function executeSecurityAssessment() {
            const statusDiv = document.getElementById('task-status');
            statusDiv.innerHTML = '🔄 Executing security assessment...';
            
            try {
                const response = await fetch('/orchestration/tasks/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task_type: 'security_assessment',
                        description: 'Comprehensive security assessment across all services',
                        services: ['hardening', 'analytics', 'threat_intelligence']
                    })
                });
                
                const result = await response.json();
                statusDiv.innerHTML = `✅ Security assessment completed: ${result.task_id}`;
                
                // Refresh tasks
                setTimeout(loadOrchestrationData, 2000);
                
            } catch (error) {
                statusDiv.innerHTML = `❌ Assessment failed: ${error.message}`;
            }
        }
        
        async function executeIncidentResponse() {
            const statusDiv = document.getElementById('task-status');
            statusDiv.innerHTML = '🚨 Initiating incident response...';
            
            try {
                const response = await fetch('/orchestration/tasks/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task_type: 'incident_response',
                        description: 'Coordinated incident response across services',
                        services: ['collaboration', 'reporting', 'nvidia_ai'],
                        parameters: { incident_id: `inc_${Date.now()}` }
                    })
                });
                
                const result = await response.json();
                statusDiv.innerHTML = `✅ Incident response initiated: ${result.task_id}`;
                
                setTimeout(loadOrchestrationData, 2000);
                
            } catch (error) {
                statusDiv.innerHTML = `❌ Incident response failed: ${error.message}`;
            }
        }
        
        async function executeThreatAnalysis() {
            const statusDiv = document.getElementById('task-status');
            statusDiv.innerHTML = '🎯 Running threat analysis...';
            
            try {
                const response = await fetch('/orchestration/tasks/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task_type: 'threat_analysis',
                        description: 'Multi-service threat analysis and correlation',
                        services: ['nvidia_ai', 'analytics', 'threat_intelligence'],
                        parameters: { 
                            threat_data: 'Suspicious network activity detected from multiple sources',
                            context: { urgency: 'high', scope: 'enterprise' }
                        }
                    })
                });
                
                const result = await response.json();
                statusDiv.innerHTML = `✅ Threat analysis completed: ${result.task_id}`;
                
                setTimeout(loadOrchestrationData, 2000);
                
            } catch (error) {
                statusDiv.innerHTML = `❌ Threat analysis failed: ${error.message}`;
            }
        }
        
        // Initialize dashboard
        connectWebSocket();
        loadOrchestrationData();
        
        // Auto-refresh every 60 seconds
        setInterval(loadOrchestrationData, 60000);
    </script>
</body>
</html>
    """

@app.get("/health")
async def health_check():
    """Orchestration hub health check"""
    return {
        "status": "healthy",
        "service": "xorb_unified_orchestration_hub",
        "version": "10.0.0",
        "capabilities": [
            "Service Orchestration",
            "Health Monitoring", 
            "Alert Management",
            "Task Automation",
            "WebSocket Streaming",
            "Cross-service Coordination",
            "System Metrics Collection"
        ],
        "orchestration_stats": {
            "registered_services": len(orchestration_hub.services),
            "healthy_services": len([s for s in orchestration_hub.services.values() if s.status == ServiceStatus.HEALTHY]),
            "active_alerts": len([a for a in orchestration_hub.alerts if a.status == "active"]),
            "total_tasks": len(orchestration_hub.orchestration_tasks),
            "websocket_connections": len(orchestration_hub.websocket_connections),
            "system_health": orchestration_hub._calculate_system_health()
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9011)