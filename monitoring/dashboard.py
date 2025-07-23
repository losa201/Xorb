#!/usr/bin/env python3

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from threading import Thread

import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import curses
import aiohttp


@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_campaigns: int
    active_agents: int
    knowledge_atoms: int
    findings_generated: int
    api_requests: int


class PrometheusExporter:
    def __init__(self, port: int = 8000):
        self.port = port
        
        # Define metrics
        self.campaign_counter = Counter('xorb_campaigns_total', 'Total number of campaigns', ['status'])
        self.agent_counter = Counter('xorb_agent_tasks_total', 'Total agent tasks', ['agent_type', 'status'])
        self.knowledge_atoms_gauge = Gauge('xorb_knowledge_atoms', 'Number of knowledge atoms', ['type'])
        self.findings_counter = Counter('xorb_findings_total', 'Total findings', ['severity'])
        self.api_request_counter = Counter('xorb_api_requests_total', 'API requests', ['endpoint', 'method'])
        self.response_time_histogram = Histogram('xorb_response_time_seconds', 'Response time', ['endpoint'])
        
        self.system_cpu_gauge = Gauge('xorb_system_cpu_percent', 'CPU usage percentage')
        self.system_memory_gauge = Gauge('xorb_system_memory_percent', 'Memory usage percentage')
        self.system_disk_gauge = Gauge('xorb_system_disk_percent', 'Disk usage percentage')
        
        self.logger = logging.getLogger(__name__)
        self._server_thread: Optional[Thread] = None

    def start_server(self):
        """Start Prometheus metrics server"""
        try:
            self._server_thread = Thread(target=self._run_server, daemon=True)
            self._server_thread.start()
            self.logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")

    def _run_server(self):
        """Run the Prometheus metrics server"""
        prometheus_client.start_http_server(self.port)

    def record_campaign_event(self, status: str):
        """Record campaign event"""
        self.campaign_counter.labels(status=status).inc()

    def record_agent_task(self, agent_type: str, status: str):
        """Record agent task"""
        self.agent_counter.labels(agent_type=agent_type, status=status).inc()

    def update_knowledge_atoms(self, atom_type: str, count: int):
        """Update knowledge atoms count"""
        self.knowledge_atoms_gauge.labels(type=atom_type).set(count)

    def record_finding(self, severity: str):
        """Record finding"""
        self.findings_counter.labels(severity=severity).inc()

    def record_api_request(self, endpoint: str, method: str, response_time: float):
        """Record API request"""
        self.api_request_counter.labels(endpoint=endpoint, method=method).inc()
        self.response_time_histogram.labels(endpoint=endpoint).observe(response_time)

    def update_system_metrics(self, cpu: float, memory: float, disk: float):
        """Update system metrics"""
        self.system_cpu_gauge.set(cpu)
        self.system_memory_gauge.set(memory)
        self.system_disk_gauge.set(disk)


class TerminalDashboard:
    def __init__(self, refresh_interval: int = 5):
        self.refresh_interval = refresh_interval
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Mock data for demonstration
        self.system_metrics: List[SystemMetrics] = []
        self.max_metrics_history = 60  # Keep 5 minutes of data at 5s intervals

    def start(self):
        """Start the terminal dashboard"""
        try:
            curses.wrapper(self._run_dashboard)
        except KeyboardInterrupt:
            self.running = False
            self.logger.info("Dashboard stopped by user")
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")

    def _run_dashboard(self, stdscr):
        """Main dashboard loop"""
        self.running = True
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        
        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Good status
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)     # Error
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Info
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # Highlight
        
        while self.running:
            try:
                # Update metrics
                self._update_metrics()
                
                # Clear screen
                stdscr.clear()
                height, width = stdscr.getmaxyx()
                
                # Draw dashboard
                self._draw_header(stdscr, width)
                self._draw_system_status(stdscr, 3, width)
                self._draw_campaign_status(stdscr, 10, width)
                self._draw_agent_status(stdscr, 17, width)
                self._draw_knowledge_status(stdscr, 24, width)
                self._draw_footer(stdscr, height - 2, width)
                
                stdscr.refresh()
                
                # Check for quit
                key = stdscr.getch()
                if key == ord('q'):
                    break
                
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Dashboard update error: {e}")
                break
        
        self.running = False

    def _draw_header(self, stdscr, width):
        """Draw dashboard header"""
        title = "XORB Security Platform - Real-time Dashboard"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.color_pair(5) | curses.A_BOLD)
        stdscr.addstr(1, width - len(timestamp) - 2, timestamp, curses.color_pair(4))
        stdscr.addstr(2, 0, "=" * width, curses.color_pair(4))

    def _draw_system_status(self, stdscr, start_row, width):
        """Draw system status section"""
        stdscr.addstr(start_row, 0, "SYSTEM STATUS", curses.color_pair(5) | curses.A_BOLD)
        
        if self.system_metrics:
            latest = self.system_metrics[-1]
            
            # CPU
            cpu_color = self._get_status_color(latest.cpu_usage, 70, 90)
            stdscr.addstr(start_row + 1, 2, f"CPU Usage:    {latest.cpu_usage:5.1f}%", cpu_color)
            
            # Memory  
            mem_color = self._get_status_color(latest.memory_usage, 70, 90)
            stdscr.addstr(start_row + 2, 2, f"Memory Usage: {latest.memory_usage:5.1f}%", mem_color)
            
            # Disk
            disk_color = self._get_status_color(latest.disk_usage, 80, 95)
            stdscr.addstr(start_row + 3, 2, f"Disk Usage:   {latest.disk_usage:5.1f}%", disk_color)
            
            # Uptime (mock)
            uptime = "2d 14h 23m"
            stdscr.addstr(start_row + 4, 2, f"Uptime:       {uptime}", curses.color_pair(1))
            
        else:
            stdscr.addstr(start_row + 1, 2, "No system data available", curses.color_pair(2))

    def _draw_campaign_status(self, stdscr, start_row, width):
        """Draw campaign status section"""
        stdscr.addstr(start_row, 0, "CAMPAIGN STATUS", curses.color_pair(5) | curses.A_BOLD)
        
        # Mock campaign data
        active_campaigns = 3
        completed_campaigns = 45
        total_findings = 127
        
        stdscr.addstr(start_row + 1, 2, f"Active Campaigns:    {active_campaigns:3d}", curses.color_pair(1))
        stdscr.addstr(start_row + 2, 2, f"Completed Campaigns: {completed_campaigns:3d}", curses.color_pair(4))
        stdscr.addstr(start_row + 3, 2, f"Total Findings:      {total_findings:3d}", curses.color_pair(1))
        
        # Recent campaigns
        stdscr.addstr(start_row + 5, 2, "Recent Campaigns:", curses.color_pair(4))
        campaigns = [
            "WebApp-Scan-001      [RUNNING]",
            "Network-Recon-042    [COMPLETED]", 
            "API-Security-Test    [RUNNING]"
        ]
        
        for i, campaign in enumerate(campaigns[:3]):
            color = curses.color_pair(1) if "[RUNNING]" in campaign else curses.color_pair(4)
            stdscr.addstr(start_row + 6 + i, 4, campaign, color)

    def _draw_agent_status(self, start_row, width):
        """Draw agent status section"""
        stdscr.addstr(start_row, 0, "AGENT STATUS", curses.color_pair(5) | curses.A_BOLD)
        
        # Mock agent data
        agents = [
            ("Playwright Agent", "RUNNING", 5, 12),
            ("Recon Agent", "IDLE", 0, 8), 
            ("Vuln Scanner", "RUNNING", 3, 7),
            ("Post-Exploit", "IDLE", 0, 2)
        ]
        
        stdscr.addstr(start_row + 1, 2, "Agent Type        Status    Queue  Completed", curses.color_pair(4))
        stdscr.addstr(start_row + 2, 2, "─" * 45, curses.color_pair(4))
        
        for i, (name, status, queue, completed) in enumerate(agents):
            status_color = curses.color_pair(1) if status == "RUNNING" else curses.color_pair(4)
            stdscr.addstr(start_row + 3 + i, 2, f"{name:<15} {status:<8} {queue:5d}  {completed:9d}", status_color)

    def _draw_knowledge_status(self, stdscr, start_row, width):
        """Draw knowledge fabric status"""
        stdscr.addstr(start_row, 0, "KNOWLEDGE FABRIC", curses.color_pair(5) | curses.A_BOLD)
        
        # Mock knowledge data
        atoms_by_type = {
            "Vulnerabilities": 1247,
            "Techniques": 892,
            "Payloads": 543,
            "Intelligence": 234,
            "Defensive": 156
        }
        
        total_atoms = sum(atoms_by_type.values())
        stdscr.addstr(start_row + 1, 2, f"Total Knowledge Atoms: {total_atoms}", curses.color_pair(1))
        
        stdscr.addstr(start_row + 3, 2, "Knowledge Distribution:", curses.color_pair(4))
        for i, (atom_type, count) in enumerate(atoms_by_type.items()):
            percentage = (count / total_atoms) * 100
            bar_length = int((count / max(atoms_by_type.values())) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            stdscr.addstr(start_row + 4 + i, 4, f"{atom_type:<15} {count:4d} {bar} {percentage:4.1f}%", curses.color_pair(1))

    def _draw_footer(self, stdscr, row, width):
        """Draw dashboard footer"""
        footer_text = "Press 'q' to quit | Refresh: {}s | XORB v1.0.0".format(self.refresh_interval)
        stdscr.addstr(row, 0, "=" * width, curses.color_pair(4))
        stdscr.addstr(row + 1, (width - len(footer_text)) // 2, footer_text, curses.color_pair(4))

    def _update_metrics(self):
        """Update system metrics (mock data for demonstration)"""
        import random
        import psutil
        
        try:
            # Get real system metrics if psutil is available
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            memory_usage = memory.percent
            disk_usage = (disk.used / disk.total) * 100
            
        except ImportError:
            # Fall back to mock data
            cpu_usage = random.uniform(10, 85)
            memory_usage = random.uniform(30, 75)
            disk_usage = random.uniform(20, 60)
        
        # Create new metrics entry
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            active_campaigns=random.randint(2, 5),
            active_agents=random.randint(3, 8),
            knowledge_atoms=random.randint(3000, 3100),
            findings_generated=random.randint(120, 140),
            api_requests=random.randint(500, 800)
        )
        
        self.system_metrics.append(metrics)
        
        # Keep only recent metrics
        if len(self.system_metrics) > self.max_metrics_history:
            self.system_metrics = self.system_metrics[-self.max_metrics_history:]

    def _get_status_color(self, value, warning_threshold, critical_threshold):
        """Get color based on threshold values"""
        if value >= critical_threshold:
            return curses.color_pair(3)  # Red
        elif value >= warning_threshold:
            return curses.color_pair(2)  # Yellow
        else:
            return curses.color_pair(1)  # Green


class HealthChecker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.components = {
            "orchestrator": {"status": "unknown", "last_check": None},
            "knowledge_fabric": {"status": "unknown", "last_check": None},
            "agents": {"status": "unknown", "last_check": None},
            "llm_integration": {"status": "unknown", "last_check": None}
        }

    async def check_all_components(self) -> Dict[str, Any]:
        """Check health of all system components"""
        results = {}
        
        for component in self.components:
            try:
                if component == "orchestrator":
                    status = await self._check_orchestrator()
                elif component == "knowledge_fabric":
                    status = await self._check_knowledge_fabric()
                elif component == "agents":
                    status = await self._check_agents()
                elif component == "llm_integration":
                    status = await self._check_llm_integration()
                else:
                    status = {"healthy": False, "error": "Unknown component"}
                
                self.components[component]["status"] = "healthy" if status.get("healthy") else "unhealthy"
                self.components[component]["last_check"] = datetime.utcnow()
                self.components[component]["details"] = status
                
                results[component] = status
                
            except Exception as e:
                error_status = {"healthy": False, "error": str(e)}
                self.components[component]["status"] = "error"
                self.components[component]["last_check"] = datetime.utcnow()
                self.components[component]["details"] = error_status
                
                results[component] = error_status
        
        # Calculate overall health
        healthy_count = sum(1 for comp in results.values() if comp.get("healthy"))
        overall_health = healthy_count / len(results)
        
        return {
            "overall_healthy": overall_health > 0.7,
            "overall_score": overall_health,
            "components": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _check_orchestrator(self) -> Dict[str, Any]:
        """Check orchestrator health"""
        # Mock health check - in real implementation, this would connect to orchestrator
        return {
            "healthy": True,
            "campaigns_active": 3,
            "campaigns_queued": 1,
            "uptime_seconds": 86400
        }

    async def _check_knowledge_fabric(self) -> Dict[str, Any]:
        """Check knowledge fabric health"""
        # Mock health check
        return {
            "healthy": True,
            "redis_connected": True,
            "sqlite_accessible": True,
            "total_atoms": 3072,
            "cache_hit_rate": 0.85
        }

    async def _check_agents(self) -> Dict[str, Any]:
        """Check agents health"""
        # Mock health check
        return {
            "healthy": True,
            "total_agents": 4,
            "active_agents": 2,
            "idle_agents": 2,
            "error_agents": 0
        }

    async def _check_llm_integration(self) -> Dict[str, Any]:
        """Check LLM integration health"""
        # Mock health check
        return {
            "healthy": True,
            "api_accessible": True,
            "daily_requests": 45,
            "daily_limit": 100,
            "last_request_success": True
        }

    def get_component_status(self, component: str) -> Dict[str, Any]:
        """Get status of specific component"""
        return self.components.get(component, {"status": "unknown"})


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if "--prometheus" in sys.argv:
        # Start Prometheus exporter
        exporter = PrometheusExporter()
        exporter.start_server()
        
        print("Prometheus metrics server running on http://localhost:8000/metrics")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    elif "--health" in sys.argv:
        # Run health check
        async def health_check():
            checker = HealthChecker()
            results = await checker.check_all_components()
            print(json.dumps(results, indent=2, default=str))
        
        asyncio.run(health_check())
    
    else:
        # Start terminal dashboard
        dashboard = TerminalDashboard()
        print("Starting XORB Terminal Dashboard...")
        print("Press 'q' in the dashboard to quit")
        dashboard.start()