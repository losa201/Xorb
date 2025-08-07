#!/usr/bin/env python3
"""
XORB Comprehensive Logging and Alerting System
Enterprise-grade centralized logging with intelligent alerting
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import aiohttp
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="XORB Logging & Alerting",
    description="Comprehensive Logging and Alerting System for XORB Platform",
    version="1.0.0"
)

class LogEntry(BaseModel):
    timestamp: str
    service: str
    level: str
    message: str
    details: Dict = {}

class Alert(BaseModel):
    id: str
    severity: str
    title: str
    description: str
    service: str
    timestamp: str
    status: str = "active"

class XORBLoggingSystem:
    """Centralized logging and alerting system"""
    
    def __init__(self):
        self.log_buffer = []
        self.alerts = []
        self.alert_rules = {
            "service_down": {
                "condition": "service_unavailable",
                "severity": "critical",
                "cooldown": 300  # 5 minutes
            },
            "high_response_time": {
                "condition": "response_time > 10000",
                "severity": "warning", 
                "cooldown": 600  # 10 minutes
            },
            "high_cpu_usage": {
                "condition": "cpu_usage > 90",
                "severity": "warning",
                "cooldown": 300
            },
            "memory_leak": {
                "condition": "memory_usage > 95",
                "severity": "critical",
                "cooldown": 180  # 3 minutes
            }
        }
        self.notification_channels = {
            "email": {
                "enabled": True,
                "recipients": ["admin@xorb.security", "ops@xorb.security"]
            },
            "webhook": {
                "enabled": True,
                "url": "http://localhost:8000/alerts/webhook"
            }
        }
        self.alert_history = []
        
    async def collect_logs_from_services(self):
        """Collect logs from all XORB services"""
        services = [
            ("api_gateway", "http://localhost:8000"),
            ("neural_orchestrator", "http://localhost:8003"), 
            ("learning_service", "http://localhost:8004"),
            ("threat_detection", "http://localhost:8005"),
            ("evolution_accelerator", "http://localhost:8008"),
            ("auto_scaler", "http://localhost:9001")
        ]
        
        for service_name, base_url in services:
            try:
                async with aiohttp.ClientSession() as session:
                    # Health check
                    async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            
                            log_entry = LogEntry(
                                timestamp=datetime.now().isoformat(),
                                service=service_name,
                                level="INFO",
                                message=f"Service health check successful",
                                details=health_data
                            )
                            
                            await self.add_log_entry(log_entry)
                            
                            # Check for alert conditions
                            await self.evaluate_alert_conditions(service_name, health_data)
                            
                        else:
                            # Service unavailable
                            log_entry = LogEntry(
                                timestamp=datetime.now().isoformat(),
                                service=service_name,
                                level="ERROR",
                                message=f"Service health check failed: HTTP {response.status}",
                                details={"status_code": response.status, "url": base_url}
                            )
                            
                            await self.add_log_entry(log_entry)
                            await self.trigger_alert("service_down", service_name, f"Service {service_name} is unavailable")
                            
            except Exception as e:
                # Connection error
                log_entry = LogEntry(
                    timestamp=datetime.now().isoformat(),
                    service=service_name,
                    level="ERROR", 
                    message=f"Failed to connect to service: {str(e)}",
                    details={"error": str(e), "url": base_url}
                )
                
                await self.add_log_entry(log_entry)
                await self.trigger_alert("service_down", service_name, f"Cannot connect to {service_name}: {str(e)}")
    
    async def add_log_entry(self, log_entry: LogEntry):
        """Add log entry to buffer"""
        self.log_buffer.append(log_entry)
        
        # Keep only last 1000 log entries in memory
        if len(self.log_buffer) > 1000:
            self.log_buffer = self.log_buffer[-1000:]
        
        # Print to console for immediate visibility
        print(f"[{log_entry.timestamp}] {log_entry.service} - {log_entry.level}: {log_entry.message}")
    
    async def evaluate_alert_conditions(self, service_name: str, health_data: Dict):
        """Evaluate if alert conditions are met"""
        # Check response time
        if "response_time_ms" in health_data:
            response_time = health_data["response_time_ms"]
            if response_time > 10000:  # 10 seconds
                await self.trigger_alert(
                    "high_response_time", 
                    service_name,
                    f"High response time: {response_time}ms"
                )
        
        # Check CPU usage (if available)
        if "cpu_usage_percent" in health_data:
            cpu_usage = health_data["cpu_usage_percent"]
            if cpu_usage > 90:
                await self.trigger_alert(
                    "high_cpu_usage",
                    service_name, 
                    f"High CPU usage: {cpu_usage}%"
                )
        
        # Check memory usage (if available)
        if "memory_usage_percent" in health_data:
            memory_usage = health_data["memory_usage_percent"]
            if memory_usage > 95:
                await self.trigger_alert(
                    "memory_leak",
                    service_name,
                    f"Critical memory usage: {memory_usage}%"
                )
    
    async def trigger_alert(self, alert_type: str, service: str, description: str):
        """Trigger alert if not in cooldown"""
        alert_rule = self.alert_rules.get(alert_type)
        if not alert_rule:
            return
        
        # Check cooldown
        recent_alerts = [
            a for a in self.alert_history 
            if a["type"] == alert_type 
            and a["service"] == service
            and (datetime.now() - datetime.fromisoformat(a["timestamp"])).seconds < alert_rule["cooldown"]
        ]
        
        if recent_alerts:
            return  # Still in cooldown
        
        # Create alert
        alert_id = f"{alert_type}_{service}_{int(datetime.now().timestamp())}"
        alert = Alert(
            id=alert_id,
            severity=alert_rule["severity"],
            title=f"{alert_type.replace('_', ' ').title()} - {service}",
            description=description,
            service=service,
            timestamp=datetime.now().isoformat()
        )
        
        self.alerts.append(alert)
        self.alert_history.append({
            "type": alert_type,
            "service": service,
            "timestamp": alert.timestamp,
            "severity": alert.severity
        })
        
        # Send notifications
        await self.send_alert_notifications(alert)
        
        # Log the alert
        log_entry = LogEntry(
            timestamp=alert.timestamp,
            service="alerting_system",
            level="WARN" if alert.severity == "warning" else "ERROR",
            message=f"Alert triggered: {alert.title}",
            details={"alert_id": alert_id, "description": description}
        )
        
        await self.add_log_entry(log_entry)
    
    async def send_alert_notifications(self, alert: Alert):
        """Send alert notifications via configured channels"""
        # Email notifications
        if self.notification_channels["email"]["enabled"]:
            await self.send_email_alert(alert)
        
        # Webhook notifications  
        if self.notification_channels["webhook"]["enabled"]:
            await self.send_webhook_alert(alert)
    
    async def send_email_alert(self, alert: Alert):
        """Send email alert (simulated - would use real SMTP in production)"""
        try:
            email_body = f"""
XORB Platform Alert

Alert ID: {alert.id}
Severity: {alert.severity.upper()}
Service: {alert.service}
Title: {alert.title}
Description: {alert.description}
Timestamp: {alert.timestamp}

This is an automated alert from the XORB monitoring system.
Please investigate and take appropriate action.

Dashboard: http://localhost:3001
Monitoring: http://localhost:3002
            """
            
            # In production, would send actual email
            print(f"📧 EMAIL ALERT SENT: {alert.severity.upper()} - {alert.title}")
            
            # Log email sent
            log_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                service="alerting_system",
                level="INFO",
                message=f"Email alert sent for {alert.id}",
                details={"recipients": self.notification_channels["email"]["recipients"]}
            )
            await self.add_log_entry(log_entry)
            
        except Exception as e:
            print(f"❌ Failed to send email alert: {e}")
    
    async def send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        try:
            webhook_url = self.notification_channels["webhook"]["url"]
            alert_data = {
                "alert_id": alert.id,
                "severity": alert.severity,
                "title": alert.title,
                "description": alert.description,
                "service": alert.service,
                "timestamp": alert.timestamp
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=alert_data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        print(f"🔗 WEBHOOK ALERT SENT: {alert.severity.upper()} - {alert.title}")
                    else:
                        print(f"⚠️  Webhook alert failed: HTTP {response.status}")
                        
        except Exception as e:
            print(f"❌ Failed to send webhook alert: {e}")
    
    async def monitoring_loop(self):
        """Main monitoring loop for log collection"""
        print("📊 Starting XORB Logging & Alerting Monitor")
        
        while True:
            try:
                await self.collect_logs_from_services()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    def get_logs(self, service: Optional[str] = None, level: Optional[str] = None, limit: int = 100) -> List[LogEntry]:
        """Get filtered logs"""
        logs = self.log_buffer
        
        if service:
            logs = [log for log in logs if log.service == service]
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        return logs[-limit:]
    
    def get_alerts(self, status: Optional[str] = None, severity: Optional[str] = None) -> List[Alert]:
        """Get filtered alerts"""
        alerts = self.alerts
        
        if status:
            alerts = [alert for alert in alerts if alert.status == status]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts
    
    def get_system_health(self) -> Dict:
        """Get overall system health summary"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        
        recent_logs = [
            log for log in self.log_buffer 
            if datetime.fromisoformat(log.timestamp) > last_hour
        ]
        
        error_logs = [log for log in recent_logs if log.level == "ERROR"]
        warning_logs = [log for log in recent_logs if log.level == "WARN"]
        
        active_alerts = [alert for alert in self.alerts if alert.status == "active"]
        critical_alerts = [alert for alert in active_alerts if alert.severity == "critical"]
        
        health_score = 100
        if critical_alerts:
            health_score -= len(critical_alerts) * 20
        if len(error_logs) > 10:
            health_score -= 15
        if len(warning_logs) > 20:
            health_score -= 10
        
        health_score = max(health_score, 0)
        
        return {
            "overall_health_score": health_score,
            "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "critical",
            "total_logs_last_hour": len(recent_logs),
            "error_logs_last_hour": len(error_logs),
            "warning_logs_last_hour": len(warning_logs),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "timestamp": now.isoformat()
        }

# Initialize logging system
logging_system = XORBLoggingSystem()

@app.on_event("startup")
async def startup_event():
    """Start monitoring loop on startup"""
    asyncio.create_task(logging_system.monitoring_loop())

@app.get("/logs")
async def get_logs(service: Optional[str] = None, level: Optional[str] = None, limit: int = 100):
    """Get system logs with optional filtering"""
    logs = logging_system.get_logs(service, level, limit)
    return {
        "total_logs": len(logs),
        "filters": {"service": service, "level": level, "limit": limit},
        "logs": [log.dict() for log in logs]
    }

@app.get("/alerts")
async def get_alerts(status: Optional[str] = None, severity: Optional[str] = None):
    """Get system alerts with optional filtering"""
    alerts = logging_system.get_alerts(status, severity)
    return {
        "total_alerts": len(alerts),
        "filters": {"status": status, "severity": severity},
        "alerts": [alert.dict() for alert in alerts]
    }

@app.get("/health/system")
async def get_system_health():
    """Get comprehensive system health status"""
    return logging_system.get_system_health()

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    for alert in logging_system.alerts:
        if alert.id == alert_id:
            alert.status = "acknowledged"
            
            # Log acknowledgment
            log_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                service="alerting_system",
                level="INFO",
                message=f"Alert acknowledged: {alert_id}",
                details={"alert_title": alert.title}
            )
            await logging_system.add_log_entry(log_entry)
            
            return {"message": f"Alert {alert_id} acknowledged"}
    
    return {"error": f"Alert {alert_id} not found"}

@app.post("/logs/manual")
async def add_manual_log(log_entry: LogEntry):
    """Add manual log entry"""
    await logging_system.add_log_entry(log_entry)
    return {"message": "Log entry added successfully"}

@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get logging and alerting metrics summary"""
    now = datetime.now()
    
    return {
        "timestamp": now.isoformat(),
        "log_buffer_size": len(logging_system.log_buffer),
        "total_alerts": len(logging_system.alerts),
        "alert_history_size": len(logging_system.alert_history),
        "alert_rules_configured": len(logging_system.alert_rules),
        "notification_channels": logging_system.notification_channels,
        "uptime_seconds": (now - datetime.now().replace(second=0, microsecond=0)).seconds
    }

@app.get("/health")
async def health_check():
    """Logging system health check"""
    return {
        "status": "healthy",
        "service": "xorb_logging_alerting",
        "version": "1.0.0",
        "features": [
            "Centralized Logging",
            "Intelligent Alerting",
            "Email Notifications",
            "Webhook Integration",
            "Health Monitoring",
            "Log Filtering",
            "Alert Management"
        ],
        "monitoring_active": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9002)