#!/usr/bin/env python3
"""
XORB Dynamic Dashboard Generator
Real-time executive and operational dashboards with advanced visualization
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('XORB-DASHBOARD')

@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    widget_type: str  # chart, metric, table, alert, map
    title: str
    data_source: str
    refresh_interval: int  # seconds
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    dashboard_id: str
    dashboard_name: str
    dashboard_type: str  # executive, operational, security, compliance
    widgets: List[DashboardWidget]
    refresh_interval: int = 30
    auto_refresh: bool = True
    access_level: str = "admin"  # admin, analyst, viewer

class ExecutiveDashboardGenerator:
    """Generate executive-level dashboards with KPIs and strategic metrics."""
    
    def __init__(self):
        self.generator_id = f"EXEC-DASH-{str(uuid.uuid4())[:8].upper()}"
        
    def generate_executive_layout(self) -> DashboardLayout:
        """Generate executive dashboard layout."""
        widgets = [
            # KPI Summary Row
            DashboardWidget(
                widget_id="kpi-operations",
                widget_type="metric",
                title="Total Operations (24h)",
                data_source="operations_counter",
                refresh_interval=60,
                position={"x": 0, "y": 0, "width": 3, "height": 2},
                config={"format": "number", "trend": True, "color": "blue"}
            ),
            DashboardWidget(
                widget_id="kpi-performance",
                widget_type="metric",
                title="Avg Performance",
                data_source="performance_gauge",
                refresh_interval=30,
                position={"x": 3, "y": 0, "width": 3, "height": 2},
                config={"format": "percentage", "threshold": 0.85, "color": "green"}
            ),
            DashboardWidget(
                widget_id="kpi-threats",
                widget_type="metric",
                title="Threats Detected",
                data_source="threat_counter",
                refresh_interval=120,
                position={"x": 6, "y": 0, "width": 3, "height": 2},
                config={"format": "number", "alert_threshold": 10, "color": "red"}
            ),
            DashboardWidget(
                widget_id="kpi-roi",
                widget_type="metric",
                title="ROI Percentage",
                data_source="roi_calculator",
                refresh_interval=300,
                position={"x": 9, "y": 0, "width": 3, "height": 2},
                config={"format": "percentage", "prefix": "+", "color": "gold"}
            ),
            
            # Performance Trends
            DashboardWidget(
                widget_id="performance-trend",
                widget_type="chart",
                title="Performance Trend (7 Days)",
                data_source="performance_timeseries",
                refresh_interval=300,
                position={"x": 0, "y": 2, "width": 6, "height": 4},
                config={"chart_type": "line", "y_axis": "percentage", "smoothing": True}
            ),
            DashboardWidget(
                widget_id="threat-landscape",
                widget_type="chart",
                title="Threat Landscape",
                data_source="threat_categories",
                refresh_interval=600,
                position={"x": 6, "y": 2, "width": 6, "height": 4},
                config={"chart_type": "pie", "show_percentages": True, "3d": True}
            ),
            
            # Operational Overview
            DashboardWidget(
                widget_id="system-health",
                widget_type="chart",
                title="System Health Matrix",
                data_source="system_metrics",
                refresh_interval=60,
                position={"x": 0, "y": 6, "width": 8, "height": 3},
                config={"chart_type": "heatmap", "metrics": ["cpu", "memory", "network", "agents"]}
            ),
            DashboardWidget(
                widget_id="critical-alerts",
                widget_type="alert",
                title="Critical Alerts",
                data_source="alert_feed",
                refresh_interval=30,
                position={"x": 8, "y": 6, "width": 4, "height": 3},
                config={"max_items": 5, "severity_filter": ["critical", "high"]}
            )
        ]
        
        return DashboardLayout(
            dashboard_id="executive-main",
            dashboard_name="Executive Command Center",
            dashboard_type="executive",
            widgets=widgets,
            refresh_interval=30,
            auto_refresh=True,
            access_level="executive"
        )

class OperationalDashboardGenerator:
    """Generate operational dashboards for SOC and technical teams."""
    
    def __init__(self):
        self.generator_id = f"OPS-DASH-{str(uuid.uuid4())[:8].upper()}"
    
    def generate_operational_layout(self) -> DashboardLayout:
        """Generate operational dashboard layout."""
        widgets = [
            # Agent Status Row
            DashboardWidget(
                widget_id="agent-overview",
                widget_type="table",
                title="Agent Status Overview",
                data_source="agent_registry",
                refresh_interval=30,
                position={"x": 0, "y": 0, "width": 8, "height": 4},
                config={"columns": ["agent_id", "type", "status", "performance", "last_seen"], "sortable": True}
            ),
            DashboardWidget(
                widget_id="agent-performance",
                widget_type="chart",
                title="Agent Performance Distribution",
                data_source="agent_performance",
                refresh_interval=60,
                position={"x": 8, "y": 0, "width": 4, "height": 4},
                config={"chart_type": "histogram", "bins": 20, "color": "blue"}
            ),
            
            # Resource Utilization
            DashboardWidget(
                widget_id="cpu-utilization",
                widget_type="chart",
                title="CPU Utilization (Real-time)",
                data_source="cpu_metrics",
                refresh_interval=15,
                position={"x": 0, "y": 4, "width": 4, "height": 3},
                config={"chart_type": "gauge", "min": 0, "max": 100, "warning": 80, "critical": 90}
            ),
            DashboardWidget(
                widget_id="memory-utilization",
                widget_type="chart",
                title="Memory Utilization",
                data_source="memory_metrics",
                refresh_interval=15,
                position={"x": 4, "y": 4, "width": 4, "height": 3},
                config={"chart_type": "gauge", "min": 0, "max": 100, "warning": 70, "critical": 85}
            ),
            DashboardWidget(
                widget_id="network-throughput",
                widget_type="chart",
                title="Network Throughput",
                data_source="network_metrics",
                refresh_interval=30,
                position={"x": 8, "y": 4, "width": 4, "height": 3},
                config={"chart_type": "area", "unit": "Mbps", "stacked": False}
            ),
            
            # Campaign Operations
            DashboardWidget(
                widget_id="active-campaigns",
                widget_type="table",
                title="Active Campaigns",
                data_source="campaign_registry",
                refresh_interval=60,
                position={"x": 0, "y": 7, "width": 12, "height": 3},
                config={"columns": ["campaign_id", "status", "progress", "targets", "duration"], "progress_bars": True}
            )
        ]
        
        return DashboardLayout(
            dashboard_id="operational-soc",
            dashboard_name="SOC Operations Center",
            dashboard_type="operational",
            widgets=widgets,
            refresh_interval=15,
            auto_refresh=True,
            access_level="analyst"
        )

class SecurityDashboardGenerator:
    """Generate security-focused dashboards for threat monitoring."""
    
    def __init__(self):
        self.generator_id = f"SEC-DASH-{str(uuid.uuid4())[:8].upper()}"
    
    def generate_security_layout(self) -> DashboardLayout:
        """Generate security dashboard layout."""
        widgets = [
            # Threat Intelligence Row
            DashboardWidget(
                widget_id="threat-feed",
                widget_type="table",
                title="Live Threat Intelligence Feed",
                data_source="threat_intelligence",
                refresh_interval=60,
                position={"x": 0, "y": 0, "width": 8, "height": 4},
                config={"columns": ["timestamp", "threat_type", "severity", "source", "confidence"], "auto_scroll": True}
            ),
            DashboardWidget(
                widget_id="threat-severity",
                widget_type="chart",
                title="Threat Severity Distribution",
                data_source="threat_severity",
                refresh_interval=300,
                position={"x": 8, "y": 0, "width": 4, "height": 4},
                config={"chart_type": "donut", "colors": ["#ff4444", "#ff8800", "#ffcc00", "#88cc00"]}
            ),
            
            # Attack Patterns
            DashboardWidget(
                widget_id="attack-timeline",
                widget_type="chart",
                title="Attack Pattern Timeline",
                data_source="attack_events",
                refresh_interval=120,
                position={"x": 0, "y": 4, "width": 12, "height": 3},
                config={"chart_type": "timeline", "event_types": ["reconnaissance", "exploitation", "persistence", "exfiltration"]}
            ),
            
            # Vulnerability Management
            DashboardWidget(
                widget_id="vulnerability-status",
                widget_type="chart",
                title="Vulnerability Lifecycle Status",
                data_source="vulnerability_lifecycle",
                refresh_interval=600,
                position={"x": 0, "y": 7, "width": 6, "height": 3},
                config={"chart_type": "stacked_bar", "categories": ["discovered", "triaged", "patched", "verified"]}
            ),
            DashboardWidget(
                widget_id="mitigation-effectiveness",
                widget_type="chart",
                title="Mitigation Effectiveness",
                data_source="mitigation_metrics",
                refresh_interval=600,
                position={"x": 6, "y": 7, "width": 6, "height": 3},
                config={"chart_type": "radar", "metrics": ["detection", "prevention", "response", "recovery"]}
            )
        ]
        
        return DashboardLayout(
            dashboard_id="security-threat",
            dashboard_name="Threat Monitoring Center",
            dashboard_type="security",
            widgets=widgets,
            refresh_interval=60,
            auto_refresh=True,
            access_level="security"
        )

class DashboardDataProvider:
    """Provides real-time data for dashboard widgets."""
    
    def __init__(self):
        self.provider_id = f"DATA-{str(uuid.uuid4())[:8].upper()}"
        self.data_cache = {}
        self.last_update = {}
    
    async def get_widget_data(self, data_source: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for a specific widget."""
        current_time = time.time()
        
        # Simulate different data sources
        if data_source == "operations_counter":
            return {"value": np.random.randint(150, 300), "trend": "up", "change": "+12%"}
        
        elif data_source == "performance_gauge":
            return {"value": 0.87 + np.random.uniform(-0.05, 0.05), "status": "good", "threshold": 0.85}
        
        elif data_source == "threat_counter":
            return {"value": np.random.randint(5, 25), "severity_breakdown": {"high": 3, "medium": 8, "low": 14}}
        
        elif data_source == "roi_calculator":
            return {"value": 285.5 + np.random.uniform(-20, 30), "trend": "up", "benchmark": 250.0}
        
        elif data_source == "performance_timeseries":
            # Generate 7 days of performance data
            days = 7
            data_points = []
            for i in range(days * 24):  # Hourly data
                timestamp = current_time - (days * 24 * 3600) + (i * 3600)
                value = 0.85 + 0.1 * np.sin(i * 0.1) + np.random.uniform(-0.05, 0.05)
                data_points.append({"timestamp": timestamp, "value": value})
            return {"data": data_points, "trend": "stable"}
        
        elif data_source == "threat_categories":
            categories = ["APT", "Malware", "Phishing", "Ransomware", "Zero-day"]
            return {
                "data": [{"category": cat, "count": np.random.randint(5, 20)} for cat in categories],
                "total": sum(np.random.randint(5, 20) for _ in categories)
            }
        
        elif data_source == "system_metrics":
            metrics = ["cpu", "memory", "network", "agents"]
            return {
                "data": [
                    {"metric": metric, "value": np.random.uniform(0.3, 0.9), "status": "normal"}
                    for metric in metrics
                ]
            }
        
        elif data_source == "alert_feed":
            alerts = []
            for i in range(5):
                alerts.append({
                    "id": f"ALERT-{str(uuid.uuid4())[:8].upper()}",
                    "severity": np.random.choice(["critical", "high", "medium"]),
                    "message": f"Alert message {i+1}",
                    "timestamp": current_time - np.random.randint(0, 3600)
                })
            return {"alerts": alerts}
        
        elif data_source == "agent_registry":
            agents = []
            for i in range(10):
                agents.append({
                    "agent_id": f"AGENT-{i+1:03d}",
                    "type": np.random.choice(["security", "red_team", "blue_team"]),
                    "status": np.random.choice(["active", "idle", "learning"]),
                    "performance": np.random.uniform(0.75, 0.98),
                    "last_seen": current_time - np.random.randint(0, 300)
                })
            return {"agents": agents}
        
        elif data_source == "cpu_metrics":
            return {"value": 75.3 + np.random.uniform(-15, 15), "status": "normal"}
        
        elif data_source == "memory_metrics":
            return {"value": 42.8 + np.random.uniform(-10, 10), "status": "normal"}
        
        elif data_source == "network_metrics":
            return {"value": 245.7 + np.random.uniform(-50, 50), "unit": "Mbps"}
        
        else:
            return {"error": f"Unknown data source: {data_source}"}

class ComprehensiveDashboardOrchestrator:
    """Orchestrates comprehensive dashboard generation and management."""
    
    def __init__(self):
        self.orchestrator_id = f"DASH-ORCH-{str(uuid.uuid4())[:8].upper()}"
        self.exec_generator = ExecutiveDashboardGenerator()
        self.ops_generator = OperationalDashboardGenerator()
        self.sec_generator = SecurityDashboardGenerator()
        self.data_provider = DashboardDataProvider()
        self.active_dashboards = {}
        self.is_running = False
        
        logger.info(f"📊 Dashboard Orchestrator initialized: {self.orchestrator_id}")
    
    def initialize_dashboards(self):
        """Initialize all dashboard layouts."""
        # Generate dashboard layouts
        exec_layout = self.exec_generator.generate_executive_layout()
        ops_layout = self.ops_generator.generate_operational_layout()
        sec_layout = self.sec_generator.generate_security_layout()
        
        self.active_dashboards = {
            exec_layout.dashboard_id: exec_layout,
            ops_layout.dashboard_id: ops_layout,
            sec_layout.dashboard_id: sec_layout
        }
        
        logger.info(f"📊 Initialized {len(self.active_dashboards)} dashboards")
        for dashboard_id, layout in self.active_dashboards.items():
            logger.info(f"   {layout.dashboard_name}: {len(layout.widgets)} widgets")
    
    async def refresh_dashboard_data(self, dashboard_id: str):
        """Refresh data for all widgets in a dashboard."""
        if dashboard_id not in self.active_dashboards:
            return
        
        dashboard = self.active_dashboards[dashboard_id]
        current_time = time.time()
        
        for widget in dashboard.widgets:
            if not widget.enabled:
                continue
            
            # Check if widget needs refresh
            last_refresh = getattr(widget, 'last_refresh', 0)
            if current_time - last_refresh < widget.refresh_interval:
                continue
            
            # Get fresh data
            try:
                widget_data = await self.data_provider.get_widget_data(widget.data_source, widget.config)
                widget.data = widget_data
                widget.last_refresh = current_time
                
                logger.debug(f"🔄 Refreshed widget: {widget.title}")
            except Exception as e:
                logger.error(f"❌ Failed to refresh widget {widget.title}: {e}")
    
    async def dashboard_refresh_loop(self):
        """Continuous dashboard refresh loop."""
        while self.is_running:
            refresh_tasks = []
            
            for dashboard_id in self.active_dashboards.keys():
                task = self.refresh_dashboard_data(dashboard_id)
                refresh_tasks.append(task)
            
            # Refresh all dashboards concurrently
            await asyncio.gather(*refresh_tasks, return_exceptions=True)
            
            # Log dashboard status
            logger.info("📊 Dashboard refresh cycle completed")
            for dashboard_id, layout in self.active_dashboards.items():
                active_widgets = len([w for w in layout.widgets if w.enabled])
                logger.info(f"   {layout.dashboard_name}: {active_widgets} active widgets")
            
            await asyncio.sleep(30)  # Global refresh every 30 seconds
    
    def export_dashboard_config(self, dashboard_id: str) -> Dict[str, Any]:
        """Export dashboard configuration as JSON."""
        if dashboard_id not in self.active_dashboards:
            return {"error": "Dashboard not found"}
        
        dashboard = self.active_dashboards[dashboard_id]
        
        config = {
            "dashboard_id": dashboard.dashboard_id,
            "dashboard_name": dashboard.dashboard_name,
            "dashboard_type": dashboard.dashboard_type,
            "refresh_interval": dashboard.refresh_interval,
            "auto_refresh": dashboard.auto_refresh,
            "access_level": dashboard.access_level,
            "widgets": []
        }
        
        for widget in dashboard.widgets:
            widget_config = {
                "widget_id": widget.widget_id,
                "widget_type": widget.widget_type,
                "title": widget.title,
                "data_source": widget.data_source,
                "refresh_interval": widget.refresh_interval,
                "position": widget.position,
                "config": widget.config,
                "enabled": widget.enabled
            }
            config["widgets"].append(widget_config)
        
        return config
    
    async def start_dashboard_orchestration(self):
        """Start comprehensive dashboard orchestration."""
        logger.info("🚀 Starting Comprehensive Dashboard Orchestration")
        
        # Initialize dashboards
        self.initialize_dashboards()
        
        self.is_running = True
        
        try:
            await self.dashboard_refresh_loop()
        except Exception as e:
            logger.error(f"❌ Dashboard orchestration error: {e}")
        finally:
            logger.info("🏁 Dashboard orchestration stopped")
    
    def generate_dashboard_summary(self) -> Dict[str, Any]:
        """Generate summary of all active dashboards."""
        summary = {
            "orchestrator_id": self.orchestrator_id,
            "total_dashboards": len(self.active_dashboards),
            "dashboard_details": {},
            "generation_time": time.time()
        }
        
        for dashboard_id, layout in self.active_dashboards.items():
            total_widgets = len(layout.widgets)
            active_widgets = len([w for w in layout.widgets if w.enabled])
            
            summary["dashboard_details"][dashboard_id] = {
                "name": layout.dashboard_name,
                "type": layout.dashboard_type,
                "total_widgets": total_widgets,
                "active_widgets": active_widgets,
                "refresh_interval": layout.refresh_interval,
                "access_level": layout.access_level
            }
        
        return summary

async def main():
    """Main execution for dashboard orchestration."""
    dashboard_orchestrator = ComprehensiveDashboardOrchestrator()
    
    print(f"\n📊 XORB COMPREHENSIVE DASHBOARD ORCHESTRATOR ACTIVATED")
    print(f"🆔 Orchestrator ID: {dashboard_orchestrator.orchestrator_id}")
    print(f"📈 Dashboard Types: Executive, Operational, Security")
    print(f"🔄 Real-time Data: Auto-refresh, Live Updates")
    print(f"📱 Multi-level Access: Executive, Analyst, Viewer")
    print(f"\n🔥 DASHBOARD ORCHESTRATION STARTING...\n")
    
    try:
        await dashboard_orchestrator.start_dashboard_orchestration()
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard orchestration interrupted by user")
        
        # Export final dashboard configurations
        for dashboard_id in dashboard_orchestrator.active_dashboards.keys():
            config = dashboard_orchestrator.export_dashboard_config(dashboard_id)
            filename = f"dashboard_config_{dashboard_id}.json"
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"💾 Dashboard config exported: {filename}")
        
        # Generate final summary
        summary = dashboard_orchestrator.generate_dashboard_summary()
        print(f"\n📊 DASHBOARD ORCHESTRATION SUMMARY:")
        print(f"   Total Dashboards: {summary['total_dashboards']}")
        for dashboard_id, details in summary['dashboard_details'].items():
            print(f"   {details['name']}: {details['active_widgets']} widgets active")
    
    except Exception as e:
        logger.error(f"Dashboard orchestration failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())