#!/usr/bin/env python3
"""
XORB Mobile-Responsive Interface System
Progressive Web App with mobile-first design for cybersecurity operations
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import aiohttp
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="XORB Mobile-Responsive Interface",
    description="Progressive Web App for mobile cybersecurity operations",
    version="8.0.0"
)

class DeviceType(str, Enum):
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"

class UserRole(str, Enum):
    ANALYST = "analyst"
    MANAGER = "manager"
    EXECUTIVE = "executive"

@dataclass
class MobileSession:
    session_id: str
    user_id: str
    device_type: DeviceType
    user_agent: str
    last_activity: datetime
    preferences: Dict

class DashboardWidget(BaseModel):
    widget_id: str
    title: str
    type: str
    data: Dict
    priority: int
    mobile_enabled: bool

class MobileNotification(BaseModel):
    notification_id: str
    title: str
    message: str
    severity: str
    timestamp: str
    read: bool
    action_url: Optional[str] = None

class MobileInterface:
    """Mobile-responsive interface system"""

    def __init__(self):
        self.active_sessions: Dict[str, MobileSession] = {}
        self.dashboard_widgets: List[DashboardWidget] = []
        self.notifications: List[MobileNotification] = []

        # Initialize sample data
        self._initialize_widgets()
        self._initialize_notifications()

    def _initialize_widgets(self):
        """Initialize dashboard widgets"""
        widgets = [
            {
                "widget_id": "security_score",
                "title": "Security Score",
                "type": "metric",
                "data": {"value": 87, "unit": "%", "trend": "up"},
                "priority": 1,
                "mobile_enabled": True
            },
            {
                "widget_id": "active_threats",
                "title": "Active Threats",
                "type": "counter",
                "data": {"count": 12, "critical": 2, "high": 5, "medium": 5},
                "priority": 2,
                "mobile_enabled": True
            },
            {
                "widget_id": "system_status",
                "title": "System Status",
                "type": "status",
                "data": {"status": "operational", "uptime": "99.9%", "services": 8},
                "priority": 3,
                "mobile_enabled": True
            },
            {
                "widget_id": "recent_incidents",
                "title": "Recent Incidents",
                "type": "list",
                "data": {
                    "incidents": [
                        {"id": "INC-001", "severity": "High", "status": "Investigating", "time": "2 min ago"},
                        {"id": "INC-002", "severity": "Medium", "status": "Resolved", "time": "15 min ago"},
                        {"id": "INC-003", "severity": "Low", "status": "Closed", "time": "1 hour ago"}
                    ]
                },
                "priority": 4,
                "mobile_enabled": True
            },
            {
                "widget_id": "threat_map",
                "title": "Threat Map",
                "type": "map",
                "data": {
                    "regions": [
                        {"region": "US-East", "threats": 25, "severity": "medium"},
                        {"region": "EU-West", "threats": 18, "severity": "low"},
                        {"region": "APAC", "threats": 32, "severity": "high"}
                    ]
                },
                "priority": 5,
                "mobile_enabled": False  # Complex visualization, desktop-only
            },
            {
                "widget_id": "compliance_status",
                "title": "Compliance",
                "type": "progress",
                "data": {
                    "frameworks": [
                        {"name": "SOC2", "score": 87, "status": "compliant"},
                        {"name": "ISO27001", "score": 92, "status": "compliant"},
                        {"name": "NIST", "score": 89, "status": "compliant"}
                    ]
                },
                "priority": 6,
                "mobile_enabled": True
            }
        ]

        for widget_data in widgets:
            widget = DashboardWidget(**widget_data)
            self.dashboard_widgets.append(widget)

    def _initialize_notifications(self):
        """Initialize sample notifications"""
        notifications = [
            {
                "notification_id": "notif_001",
                "title": "Critical Alert",
                "message": "Suspicious activity detected from IP 192.168.1.100",
                "severity": "critical",
                "timestamp": datetime.now().isoformat(),
                "read": False,
                "action_url": "/threats/investigate/192.168.1.100"
            },
            {
                "notification_id": "notif_002",
                "title": "System Update",
                "message": "Security patches applied successfully to 15 servers",
                "severity": "info",
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "read": False,
                "action_url": "/systems/updates"
            },
            {
                "notification_id": "notif_003",
                "title": "Compliance Alert",
                "message": "SOC2 audit scheduled for next week - preparation required",
                "severity": "warning",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "read": True,
                "action_url": "/compliance/soc2"
            }
        ]

        for notif_data in notifications:
            notification = MobileNotification(**notif_data)
            self.notifications.append(notification)

    def detect_device_type(self, user_agent: str) -> DeviceType:
        """Detect device type from user agent"""
        user_agent_lower = user_agent.lower()

        if any(mobile in user_agent_lower for mobile in ['mobile', 'android', 'iphone', 'ipod']):
            return DeviceType.MOBILE
        elif any(tablet in user_agent_lower for tablet in ['tablet', 'ipad']):
            return DeviceType.TABLET
        else:
            return DeviceType.DESKTOP

    def get_mobile_widgets(self, device_type: DeviceType) -> List[DashboardWidget]:
        """Get widgets optimized for mobile devices"""
        if device_type == DeviceType.MOBILE:
            # Return only mobile-enabled widgets, sorted by priority
            mobile_widgets = [w for w in self.dashboard_widgets if w.mobile_enabled]
            return sorted(mobile_widgets, key=lambda x: x.priority)[:4]  # Limit to 4 for mobile
        elif device_type == DeviceType.TABLET:
            # Tablets can handle more widgets
            return sorted(self.dashboard_widgets, key=lambda x: x.priority)[:6]
        else:
            # Desktop gets all widgets
            return sorted(self.dashboard_widgets, key=lambda x: x.priority)

    def get_notifications(self, limit: int = 10, unread_only: bool = False) -> List[MobileNotification]:
        """Get notifications with mobile optimization"""
        notifications = self.notifications

        if unread_only:
            notifications = [n for n in notifications if not n.read]

        # Sort by timestamp (newest first) and limit
        notifications.sort(key=lambda x: x.timestamp, reverse=True)
        return notifications[:limit]

    async def fetch_service_data(self, service: str) -> Dict:
        """Fetch data from other XORB services"""
        service_urls = {
            "threat_intelligence": "http://localhost:9004/health",
            "collaboration": "http://localhost:9005/health",
            "analytics": "http://localhost:9006/health",
            "reporting": "http://localhost:9007/health",
            "hardening": "http://localhost:9008/health"
        }

        if service not in service_urls:
            return {"status": "unknown", "error": "Service not found"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(service_urls[service], timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"status": "healthy", "data": data}
                    else:
                        return {"status": "error", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Initialize mobile interface
mobile_interface = MobileInterface()

@app.get("/mobile/detect-device")
async def detect_device(request: Request):
    """Detect device type and capabilities"""
    user_agent = request.headers.get("user-agent", "")
    device_type = mobile_interface.detect_device_type(user_agent)

    return {
        "device_type": device_type,
        "user_agent": user_agent,
        "mobile_optimized": device_type in [DeviceType.MOBILE, DeviceType.TABLET],
        "recommendations": {
            "layout": "single-column" if device_type == DeviceType.MOBILE else "multi-column",
            "widget_size": "compact" if device_type == DeviceType.MOBILE else "standard",
            "navigation": "bottom-tabs" if device_type == DeviceType.MOBILE else "sidebar"
        }
    }

@app.get("/mobile/dashboard")
async def get_mobile_dashboard(request: Request, role: str = "analyst"):
    """Get mobile-optimized dashboard data"""
    user_agent = request.headers.get("user-agent", "")
    device_type = mobile_interface.detect_device_type(user_agent)

    # Get appropriate widgets for device
    widgets = mobile_interface.get_mobile_widgets(device_type)

    # Get recent notifications
    notifications = mobile_interface.get_notifications(limit=5, unread_only=True)

    # Fetch service status
    services_status = {}
    for service in ["threat_intelligence", "collaboration", "analytics", "reporting", "hardening"]:
        services_status[service] = await mobile_interface.fetch_service_data(service)

    return {
        "device_type": device_type,
        "widgets": [widget.dict() for widget in widgets],
        "notifications": [notif.dict() for notif in notifications],
        "services_status": services_status,
        "layout_config": {
            "columns": 1 if device_type == DeviceType.MOBILE else 2 if device_type == DeviceType.TABLET else 3,
            "compact_mode": device_type == DeviceType.MOBILE,
            "show_sidebar": device_type == DeviceType.DESKTOP
        }
    }

@app.get("/mobile/notifications")
async def get_notifications(limit: int = 20, unread_only: bool = False):
    """Get mobile-optimized notifications"""
    notifications = mobile_interface.get_notifications(limit, unread_only)
    return {
        "total_notifications": len(mobile_interface.notifications),
        "unread_count": len([n for n in mobile_interface.notifications if not n.read]),
        "notifications": [notif.dict() for notif in notifications]
    }

@app.post("/mobile/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """Mark notification as read"""
    notification = next((n for n in mobile_interface.notifications if n.notification_id == notification_id), None)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification.read = True
    return {"status": "marked_read", "notification_id": notification_id}

@app.get("/mobile/services/status")
async def get_services_status():
    """Get status of all XORB services"""
    services = ["threat_intelligence", "collaboration", "analytics", "reporting", "hardening"]
    status = {}

    for service in services:
        status[service] = await mobile_interface.fetch_service_data(service)

    # Calculate overall health
    healthy_services = sum(1 for s in status.values() if s.get("status") == "healthy")
    overall_health = (healthy_services / len(services)) * 100

    return {
        "overall_health": round(overall_health, 1),
        "healthy_services": healthy_services,
        "total_services": len(services),
        "services": status
    }

@app.get("/", response_class=HTMLResponse)
async def mobile_app(request: Request):
    """Progressive Web App - Mobile-First Interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>XORB Mobile - Cybersecurity Operations</title>

    <!-- PWA Meta Tags -->
    <meta name="theme-color" content="#0d1117">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="XORB Mobile">

    <!-- PWA Icons -->
    <link rel="icon" sizes="192x192" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTkyIiBoZWlnaHQ9IjE5MiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjNThhNmZmIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMiAyQzEzLjEgMiAxNCAyLjkgMTQgNFY4SDE2VjZIMThWOEgyMFYxMEgxOFYxMkgyMFYxNEgxOFYxNkgyMFYxOEgxOFYyMEgxNlYxOEgxNFYyMEgxMlYxOEgxMFYyMEg4VjE4SDZWMjBINFYxOEg2VjE2SDRWMTRINlYxMkg0VjEwSDZWOEg0VjZINlY4SDhWNkgxMFY4SDEyVjRDMTIgMi45IDEyLjkgMiAxMiAyWiIvPjwvc3ZnPg==">
    <link rel="apple-touch-icon" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTkyIiBoZWlnaHQ9IjE5MiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjNThhNmZmIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMiAyQzEzLjEgMiAxNCAyLjkgMTQgNFY4SDE2VjZIMThWOEgyMFYxMEgxOFYxMkgyMFYxNEgxOFYxNkgyMFYxOEgxOFYyMEgxNlYxOEgxNFYyMEgxMlYxOEgxMFYyMEg4VjE4SDZWMjBINFYxOEg2VjE2SDRWMTRINlYxMkg0VjEwSDZWOEg0VjZINlY4SDhWNkgxMFY4SDEyVjRDMTIgMi45IDEyLjkgMiAxMiAyWiIvPjwvc3ZnPg==">

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #f0f6fc;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* Mobile-first responsive design */
        .container {
            max-width: 100vw;
            padding: 0;
            margin: 0;
        }

        /* Header */
        .header {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 10px 15px;
            position: sticky;
            top: 0;
            z-index: 1000;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 1.2em;
            color: #58a6ff;
        }

        .header-actions {
            display: flex;
            gap: 10px;
        }

        .notification-badge {
            position: relative;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }

        .notification-count {
            position: absolute;
            top: -5px;
            right: -5px;
            background: #f85149;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 0.7em;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Dashboard Grid - Mobile First */
        .dashboard {
            padding: 15px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 15px;
        }

        /* Widget Cards */
        .widget {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .widget:active {
            transform: scale(0.98);
        }

        .widget-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .widget-title {
            font-size: 1em;
            font-weight: 600;
            color: #58a6ff;
        }

        .widget-icon {
            font-size: 1.2em;
        }

        /* Metric Widget */
        .metric-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #f0f6fc;
            text-align: center;
            margin: 10px 0;
        }

        .metric-unit {
            font-size: 0.8em;
            color: #8b949e;
        }

        .metric-trend {
            text-align: center;
            font-size: 0.9em;
            color: #2ea043;
        }

        /* Counter Widget */
        .counter-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }

        .counter-item {
            background: #0d1117;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }

        .counter-value {
            font-size: 1.5em;
            font-weight: bold;
        }

        .counter-critical { color: #f85149; }
        .counter-high { color: #d29922; }
        .counter-medium { color: #58a6ff; }

        /* Status Widget */
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #2ea043;
        }

        .status-dot.warning { background: #d29922; }
        .status-dot.error { background: #f85149; }

        /* List Widget */
        .incident-list {
            list-style: none;
        }

        .incident-item {
            background: #0d1117;
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid #58a6ff;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .incident-high { border-left-color: #f85149; }
        .incident-medium { border-left-color: #d29922; }
        .incident-low { border-left-color: #2ea043; }

        .incident-info {
            flex: 1;
        }

        .incident-id {
            font-weight: 600;
            color: #f0f6fc;
        }

        .incident-status {
            font-size: 0.8em;
            color: #8b949e;
        }

        .incident-time {
            font-size: 0.8em;
            color: #8b949e;
            white-space: nowrap;
        }

        /* Progress Widget */
        .compliance-list {
            list-style: none;
        }

        .compliance-item {
            margin: 12px 0;
        }

        .compliance-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }

        .compliance-name {
            font-weight: 600;
            color: #f0f6fc;
        }

        .compliance-score {
            color: #2ea043;
        }

        .progress-bar {
            background: #21262d;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #f85149, #d29922, #2ea043);
            transition: width 0.3s;
        }

        /* Bottom Navigation */
        .bottom-nav {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #161b22;
            border-top: 1px solid #30363d;
            display: flex;
            justify-content: space-around;
            padding: 10px 0;
            z-index: 1000;
        }

        .nav-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 5px 10px;
            text-decoration: none;
            color: #8b949e;
            transition: color 0.2s;
        }

        .nav-item.active {
            color: #58a6ff;
        }

        .nav-icon {
            font-size: 1.2em;
            margin-bottom: 2px;
        }

        .nav-label {
            font-size: 0.7em;
        }

        /* Notifications Panel */
        .notifications-panel {
            position: fixed;
            top: 0;
            right: -100%;
            width: 100%;
            height: 100vh;
            background: #0d1117;
            z-index: 2000;
            transition: right 0.3s;
            overflow-y: auto;
        }

        .notifications-panel.open {
            right: 0;
        }

        .notifications-header {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .notification-item {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 15px;
            border-left: 4px solid #58a6ff;
        }

        .notification-critical { border-left-color: #f85149; }
        .notification-warning { border-left-color: #d29922; }
        .notification-info { border-left-color: #58a6ff; }

        .notification-title {
            font-weight: 600;
            color: #f0f6fc;
            margin-bottom: 5px;
        }

        .notification-message {
            color: #8b949e;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .notification-time {
            color: #6e7681;
            font-size: 0.8em;
        }

        /* Loading States */
        .loading {
            text-align: center;
            padding: 20px;
            color: #8b949e;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #30363d;
            border-radius: 50%;
            border-top-color: #58a6ff;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Tablet Styles */
        @media (min-width: 768px) {
            .dashboard {
                grid-template-columns: repeat(2, 1fr);
                padding: 20px;
                gap: 20px;
            }

            .notifications-panel {
                width: 400px;
            }

            .bottom-nav {
                display: none;
            }
        }

        /* Desktop Styles */
        @media (min-width: 1024px) {
            .dashboard {
                grid-template-columns: repeat(3, 1fr);
                max-width: 1200px;
                margin: 0 auto;
                padding: 30px;
            }

            .header {
                padding: 15px 30px;
            }

            .header h1 {
                font-size: 1.5em;
            }
        }

        /* Dark mode enhancements */
        @media (prefers-color-scheme: dark) {
            body {
                background: #0d1117;
                color: #f0f6fc;
            }
        }

        /* Reduced motion */
        @media (prefers-reduced-motion: reduce) {
            .widget, .notification-item, .nav-item {
                transition: none;
            }
        }

        /* High contrast mode */
        @media (prefers-contrast: high) {
            .widget {
                border: 2px solid #58a6ff;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🛡️ XORB Mobile</h1>
            <div class="header-actions">
                <div class="notification-badge" onclick="toggleNotifications()">
                    🔔
                    <span class="notification-count" id="notification-count">3</span>
                </div>
            </div>
        </div>

        <!-- Dashboard -->
        <div class="dashboard" id="dashboard">
            <div class="loading">
                <div class="spinner"></div>
                <p>Loading dashboard...</p>
            </div>
        </div>

        <!-- Bottom Navigation (Mobile Only) -->
        <div class="bottom-nav">
            <a href="#" class="nav-item active" onclick="showDashboard()">
                <span class="nav-icon">🏠</span>
                <span class="nav-label">Dashboard</span>
            </a>
            <a href="#" class="nav-item" onclick="showThreats()">
                <span class="nav-icon">⚠️</span>
                <span class="nav-label">Threats</span>
            </a>
            <a href="#" class="nav-item" onclick="showReports()">
                <span class="nav-icon">📊</span>
                <span class="nav-label">Reports</span>
            </a>
            <a href="#" class="nav-item" onclick="showSettings()">
                <span class="nav-icon">⚙️</span>
                <span class="nav-label">Settings</span>
            </a>
        </div>

        <!-- Notifications Panel -->
        <div class="notifications-panel" id="notifications-panel">
            <div class="notifications-header">
                <h2>Notifications</h2>
                <button onclick="toggleNotifications()" style="background: none; border: none; color: #58a6ff; font-size: 1.2em; cursor: pointer;">✕</button>
            </div>
            <div id="notifications-list">
                <div class="loading">Loading notifications...</div>
            </div>
        </div>
    </div>

    <script>
        let deviceInfo = null;
        let dashboardData = null;

        // Initialize app
        async function initApp() {
            try {
                // Detect device capabilities
                deviceInfo = await fetch('/mobile/detect-device').then(r => r.json());
                console.log('Device detected:', deviceInfo);

                // Load dashboard data
                await loadDashboard();

                // Load notifications
                await loadNotifications();

                // Set up periodic refresh
                setInterval(loadDashboard, 30000); // Refresh every 30 seconds
                setInterval(loadNotifications, 60000); // Refresh notifications every minute

                // Register service worker for PWA
                if ('serviceWorker' in navigator) {
                    navigator.serviceWorker.register('/sw.js').catch(console.error);
                }

            } catch (error) {
                console.error('App initialization failed:', error);
                document.getElementById('dashboard').innerHTML = '<div class="loading">❌ Failed to load app</div>';
            }
        }

        async function loadDashboard() {
            try {
                dashboardData = await fetch('/mobile/dashboard').then(r => r.json());
                renderDashboard(dashboardData);
            } catch (error) {
                console.error('Dashboard load failed:', error);
                document.getElementById('dashboard').innerHTML = '<div class="loading">❌ Failed to load dashboard</div>';
            }
        }

        function renderDashboard(data) {
            const dashboard = document.getElementById('dashboard');
            dashboard.innerHTML = '';

            data.widgets.forEach(widget => {
                const widgetElement = createWidget(widget);
                dashboard.appendChild(widgetElement);
            });
        }

        function createWidget(widget) {
            const div = document.createElement('div');
            div.className = 'widget';

            let content = '';

            if (widget.type === 'metric') {
                content = `
                    <div class="widget-header">
                        <span class="widget-title">${widget.title}</span>
                        <span class="widget-icon">📊</span>
                    </div>
                    <div class="metric-value">
                        ${widget.data.value}<span class="metric-unit">${widget.data.unit}</span>
                    </div>
                    <div class="metric-trend">📈 ${widget.data.trend}</div>
                `;
            } else if (widget.type === 'counter') {
                content = `
                    <div class="widget-header">
                        <span class="widget-title">${widget.title}</span>
                        <span class="widget-icon">🚨</span>
                    </div>
                    <div class="counter-grid">
                        <div class="counter-item">
                            <div class="counter-value counter-critical">${widget.data.critical}</div>
                            <div>Critical</div>
                        </div>
                        <div class="counter-item">
                            <div class="counter-value counter-high">${widget.data.high}</div>
                            <div>High</div>
                        </div>
                        <div class="counter-item">
                            <div class="counter-value counter-medium">${widget.data.medium}</div>
                            <div>Medium</div>
                        </div>
                        <div class="counter-item">
                            <div class="counter-value">${widget.data.count}</div>
                            <div>Total</div>
                        </div>
                    </div>
                `;
            } else if (widget.type === 'status') {
                content = `
                    <div class="widget-header">
                        <span class="widget-title">${widget.title}</span>
                        <span class="widget-icon">✅</span>
                    </div>
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span>${widget.data.status.toUpperCase()}</span>
                    </div>
                    <div>Uptime: ${widget.data.uptime}</div>
                    <div>Services: ${widget.data.services}</div>
                `;
            } else if (widget.type === 'list') {
                const incidents = widget.data.incidents || [];
                content = `
                    <div class="widget-header">
                        <span class="widget-title">${widget.title}</span>
                        <span class="widget-icon">📋</span>
                    </div>
                    <ul class="incident-list">
                        ${incidents.map(incident => `
                            <li class="incident-item incident-${incident.severity.toLowerCase()}">
                                <div class="incident-info">
                                    <div class="incident-id">${incident.id}</div>
                                    <div class="incident-status">${incident.status}</div>
                                </div>
                                <div class="incident-time">${incident.time}</div>
                            </li>
                        `).join('')}
                    </ul>
                `;
            } else if (widget.type === 'progress') {
                const frameworks = widget.data.frameworks || [];
                content = `
                    <div class="widget-header">
                        <span class="widget-title">${widget.title}</span>
                        <span class="widget-icon">✅</span>
                    </div>
                    <ul class="compliance-list">
                        ${frameworks.map(framework => `
                            <li class="compliance-item">
                                <div class="compliance-header">
                                    <span class="compliance-name">${framework.name}</span>
                                    <span class="compliance-score">${framework.score}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${framework.score}%"></div>
                                </div>
                            </li>
                        `).join('')}
                    </ul>
                `;
            }

            div.innerHTML = content;
            return div;
        }

        async function loadNotifications() {
            try {
                const data = await fetch('/mobile/notifications?limit=10').then(r => r.json());
                updateNotificationBadge(data.unread_count);
                renderNotifications(data.notifications);
            } catch (error) {
                console.error('Notifications load failed:', error);
            }
        }

        function updateNotificationBadge(count) {
            const badge = document.getElementById('notification-count');
            badge.textContent = count;
            badge.style.display = count > 0 ? 'flex' : 'none';
        }

        function renderNotifications(notifications) {
            const list = document.getElementById('notifications-list');

            if (notifications.length === 0) {
                list.innerHTML = '<div class="loading">No notifications</div>';
                return;
            }

            list.innerHTML = notifications.map(notif => `
                <div class="notification-item notification-${notif.severity}" onclick="handleNotificationClick('${notif.notification_id}', '${notif.action_url || ''}')">
                    <div class="notification-title">${notif.title}</div>
                    <div class="notification-message">${notif.message}</div>
                    <div class="notification-time">${new Date(notif.timestamp).toLocaleTimeString()}</div>
                </div>
            `).join('');
        }

        function toggleNotifications() {
            const panel = document.getElementById('notifications-panel');
            panel.classList.toggle('open');
        }

        async function handleNotificationClick(notificationId, actionUrl) {
            try {
                // Mark as read
                await fetch(`/mobile/notifications/${notificationId}/read`, { method: 'POST' });

                // Handle action URL
                if (actionUrl) {
                    // In a real app, this would navigate to the appropriate section
                    console.log('Navigate to:', actionUrl);
                }

                // Refresh notifications
                await loadNotifications();

            } catch (error) {
                console.error('Notification click failed:', error);
            }
        }

        // Navigation handlers
        function showDashboard() {
            updateActiveNav(0);
            loadDashboard();
        }

        function showThreats() {
            updateActiveNav(1);
            document.getElementById('dashboard').innerHTML = '<div class="loading">🚨 Threat Intelligence Interface Coming Soon</div>';
        }

        function showReports() {
            updateActiveNav(2);
            document.getElementById('dashboard').innerHTML = '<div class="loading">📊 Mobile Reports Interface Coming Soon</div>';
        }

        function showSettings() {
            updateActiveNav(3);
            document.getElementById('dashboard').innerHTML = '<div class="loading">⚙️ Mobile Settings Interface Coming Soon</div>';
        }

        function updateActiveNav(activeIndex) {
            const navItems = document.querySelectorAll('.nav-item');
            navItems.forEach((item, index) => {
                item.classList.toggle('active', index === activeIndex);
            });
        }

        // Handle offline/online status
        window.addEventListener('online', () => {
            console.log('App is online');
            loadDashboard();
        });

        window.addEventListener('offline', () => {
            console.log('App is offline');
        });

        // Handle visibility change (app resume/pause)
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                loadDashboard();
                loadNotifications();
            }
        });

        // Initialize app when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initApp);
        } else {
            initApp();
        }
    </script>
</body>
</html>
    """

@app.get("/sw.js")
async def service_worker():
    """Service Worker for PWA capabilities"""
    return """
// XORB Mobile Service Worker
const CACHE_NAME = 'xorb-mobile-v1';
const urlsToCache = [
    '/',
    '/mobile/dashboard',
    '/mobile/notifications'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});

// Handle push notifications
self.addEventListener('push', (event) => {
    const options = {
        body: event.data ? event.data.text() : 'New security alert',
        icon: '/icon-192x192.png',
        badge: '/badge-72x72.png',
        vibrate: [100, 50, 100],
        data: {
            url: '/'
        }
    };

    event.waitUntil(
        self.registration.showNotification('XORB Security Alert', options)
    );
});

self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    event.waitUntil(
        clients.openWindow(event.notification.data.url)
    );
});
    """

@app.get("/manifest.json")
async def web_app_manifest():
    """PWA Web App Manifest"""
    return {
        "name": "XORB Mobile - Cybersecurity Operations",
        "short_name": "XORB Mobile",
        "description": "Mobile cybersecurity operations dashboard",
        "start_url": "/",
        "display": "standalone",
        "theme_color": "#0d1117",
        "background_color": "#0d1117",
        "orientation": "portrait",
        "categories": ["productivity", "security"],
        "icons": [
            {
                "src": "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTkyIiBoZWlnaHQ9IjE5MiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjNThhNmZmIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMiAyQzEzLjEgMiAxNCAyLjkgMTQgNFY4SDE2VjZIMThWOEgyMFYxMEgxOFYxMkgyMFYxNEgxOFYxNkgyMFYxOEgxOFYyMEgxNlYxOEgxNFYyMEgxMlYxOEgxMFYyMEg4VjE4SDZWMjBINFYxOEg2VjE2SDRWMTRINlYxMkg0VjEwSDZWOEg0VjZINlY4SDhWNkgxMFY4SDEyVjRDMTIgMi45IDEyLjkgMiAxMiAyWiIvPjwvc3ZnPg==",
                "sizes": "192x192",
                "type": "image/svg+xml"
            },
            {
                "src": "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNTEyIiBoZWlnaHQ9IjUxMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjNThhNmZmIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMiAyQzEzLjEgMiAxNCAyLjkgMTQgNFY4SDE2VjZIMThWOEgyMFYxMEgxOFYxMkgyMFYxNEgxOFYxNkgyMFYxOEgxOFYyMEgxNlYxOEgxNFYyMEgxMlYxOEgxMFYyMEg4VjE4SDZWMjBINFYxOEg2VjE2SDRWMTRINlYxMkg0VjEwSDZWOEg0VjZINlY4SDhWNkgxMFY4SDEyVjRDMTIgMi45IDEyLjkgMiAxMiAyWiIvPjwvc3ZnPg==",
                "sizes": "512x512",
                "type": "image/svg+xml"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Mobile interface system health check"""
    return {
        "status": "healthy",
        "service": "xorb_mobile_interface",
        "version": "8.0.0",
        "capabilities": [
            "Progressive Web App",
            "Mobile-First Design",
            "Responsive Layout",
            "Offline Support",
            "Push Notifications",
            "Touch Optimized",
            "Dark Mode"
        ],
        "interface_stats": {
            "active_sessions": len(mobile_interface.active_sessions),
            "total_widgets": len(mobile_interface.dashboard_widgets),
            "mobile_widgets": len([w for w in mobile_interface.dashboard_widgets if w.mobile_enabled]),
            "notifications": len(mobile_interface.notifications),
            "unread_notifications": len([n for n in mobile_interface.notifications if not n.read])
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9009)
