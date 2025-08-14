#!/usr/bin/env python3
"""
XORB Unified Platform Orchestration Dashboard
Real-time monitoring and control interface for the complete XORB ecosystem
"""

import asyncio
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status levels"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

class ServiceType(Enum):
    """Service types in XORB platform"""
    NEURAL_ORCHESTRATOR = "neural_orchestrator"
    LEARNING_SERVICE = "learning_service"
    THREAT_DETECTION = "threat_detection"
    AGENT_CLUSTER = "agent_cluster"
    INTELLIGENCE_FUSION = "intelligence_fusion"
    DATABASE = "database"
    MONITORING = "monitoring"
    API_GATEWAY = "api_gateway"

@dataclass
class ServiceHealth:
    """Service health information"""
    service_name: str
    service_type: ServiceType
    status: SystemStatus
    response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    last_check: datetime
    endpoint: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlatformMetrics:
    """Platform-wide metrics"""
    total_services: int
    operational_services: int
    degraded_services: int
    critical_services: int
    offline_services: int
    overall_health_score: float
    response_time_avg: float
    error_rate_avg: float
    uptime_percentage: float
    last_updated: datetime

class XORBPlatformOrchestrationDashboard:
    """Unified orchestration dashboard for XORB platform"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.services: Dict[str, ServiceHealth] = {}
        self.platform_metrics: Optional[PlatformMetrics] = None
        self.historical_data: List[Dict[str, Any]] = []

        # Service endpoints configuration
        self.service_endpoints = {
            ServiceType.NEURAL_ORCHESTRATOR: "http://localhost:8003",
            ServiceType.LEARNING_SERVICE: "http://localhost:8004",
            ServiceType.THREAT_DETECTION: "http://localhost:8005",
            ServiceType.AGENT_CLUSTER: "http://localhost:8006",
            ServiceType.INTELLIGENCE_FUSION: "http://localhost:8007",
            ServiceType.MONITORING: "http://localhost:9092",
            ServiceType.API_GATEWAY: "http://localhost:8000"
        }

        # Database endpoints
        self.database_endpoints = {
            "postgresql": "localhost:5434",
            "neo4j": "localhost:7476",
            "redis": "localhost:6381"
        }

        # Initialize Streamlit configuration
        self._setup_streamlit()

        logger.info("XORB Platform Orchestration Dashboard initialized")

    def _setup_streamlit(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="XORB Platform Orchestration Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .service-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #e1e1e1;
            margin: 0.5rem 0;
        }
        .status-operational { color: #28a745; font-weight: bold; }
        .status-degraded { color: #ffc107; font-weight: bold; }
        .status-critical { color: #dc3545; font-weight: bold; }
        .status-offline { color: #6c757d; font-weight: bold; }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    async def collect_service_health(self):
        """Collect health information from all services"""
        try:
            health_checks = []

            # Check each service endpoint
            for service_type, endpoint in self.service_endpoints.items():
                health_checks.append(self._check_service_health(service_type, endpoint))

            # Check database services
            for db_name, endpoint in self.database_endpoints.items():
                health_checks.append(self._check_database_health(db_name, endpoint))

            # Execute all health checks concurrently
            results = await asyncio.gather(*health_checks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, ServiceHealth):
                    self.services[result.service_name] = result
                elif isinstance(result, Exception):
                    logger.error(f"Health check failed: {result}")

            # Update platform metrics
            await self._update_platform_metrics()

            # Store historical data
            self._store_historical_data()

        except Exception as e:
            logger.error(f"Failed to collect service health: {e}")

    async def _check_service_health(self, service_type: ServiceType, endpoint: str) -> ServiceHealth:
        """Check health of a specific service"""
        try:
            start_time = time.time()

            # Make health check request
            health_endpoint = f"{endpoint}/health"
            response = requests.get(health_endpoint, timeout=5)

            response_time = (time.time() - start_time) * 1000  # ms

            if response.status_code == 200:
                health_data = response.json()
                status = SystemStatus.OPERATIONAL

                # Check for specific health indicators
                if health_data.get('status') == 'degraded':
                    status = SystemStatus.DEGRADED
                elif 'error' in health_data:
                    status = SystemStatus.CRITICAL

                return ServiceHealth(
                    service_name=service_type.value,
                    service_type=service_type,
                    status=status,
                    response_time=response_time,
                    error_rate=health_data.get('error_rate', 0.0),
                    cpu_usage=health_data.get('cpu_usage', 0.0),
                    memory_usage=health_data.get('memory_usage', 0.0),
                    last_check=datetime.now(),
                    endpoint=endpoint,
                    metadata=health_data
                )
            else:
                return ServiceHealth(
                    service_name=service_type.value,
                    service_type=service_type,
                    status=SystemStatus.CRITICAL,
                    response_time=response_time,
                    error_rate=1.0,
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    last_check=datetime.now(),
                    endpoint=endpoint,
                    metadata={'http_status': response.status_code}
                )

        except requests.exceptions.Timeout:
            return ServiceHealth(
                service_name=service_type.value,
                service_type=service_type,
                status=SystemStatus.CRITICAL,
                response_time=5000.0,  # Timeout
                error_rate=1.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                last_check=datetime.now(),
                endpoint=endpoint,
                metadata={'error': 'timeout'}
            )

        except requests.exceptions.ConnectionError:
            return ServiceHealth(
                service_name=service_type.value,
                service_type=service_type,
                status=SystemStatus.OFFLINE,
                response_time=0.0,
                error_rate=1.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                last_check=datetime.now(),
                endpoint=endpoint,
                metadata={'error': 'connection_refused'}
            )

        except Exception as e:
            return ServiceHealth(
                service_name=service_type.value,
                service_type=service_type,
                status=SystemStatus.CRITICAL,
                response_time=0.0,
                error_rate=1.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                last_check=datetime.now(),
                endpoint=endpoint,
                metadata={'error': str(e)}
            )

    async def _check_database_health(self, db_name: str, endpoint: str) -> ServiceHealth:
        """Check health of database services"""
        try:
            import socket

            host, port = endpoint.split(':')
            port = int(port)

            start_time = time.time()

            # Check port connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()

            response_time = (time.time() - start_time) * 1000  # ms

            if result == 0:
                status = SystemStatus.OPERATIONAL
            else:
                status = SystemStatus.OFFLINE

            return ServiceHealth(
                service_name=f"database_{db_name}",
                service_type=ServiceType.DATABASE,
                status=status,
                response_time=response_time,
                error_rate=0.0 if result == 0 else 1.0,
                cpu_usage=0.0,  # Database metrics would need specific queries
                memory_usage=0.0,
                last_check=datetime.now(),
                endpoint=endpoint,
                metadata={'database_type': db_name}
            )

        except Exception as e:
            return ServiceHealth(
                service_name=f"database_{db_name}",
                service_type=ServiceType.DATABASE,
                status=SystemStatus.CRITICAL,
                response_time=0.0,
                error_rate=1.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                last_check=datetime.now(),
                endpoint=endpoint,
                metadata={'error': str(e), 'database_type': db_name}
            )

    async def _update_platform_metrics(self):
        """Update platform-wide metrics"""
        if not self.services:
            return

        total_services = len(self.services)
        operational_services = len([s for s in self.services.values() if s.status == SystemStatus.OPERATIONAL])
        degraded_services = len([s for s in self.services.values() if s.status == SystemStatus.DEGRADED])
        critical_services = len([s for s in self.services.values() if s.status == SystemStatus.CRITICAL])
        offline_services = len([s for s in self.services.values() if s.status == SystemStatus.OFFLINE])

        # Calculate overall health score
        health_weights = {
            SystemStatus.OPERATIONAL: 1.0,
            SystemStatus.DEGRADED: 0.7,
            SystemStatus.CRITICAL: 0.3,
            SystemStatus.OFFLINE: 0.0
        }

        total_weight = sum(health_weights[service.status] for service in self.services.values())
        overall_health_score = total_weight / total_services if total_services > 0 else 0.0

        # Calculate average response time and error rate
        response_times = [s.response_time for s in self.services.values() if s.response_time > 0]
        error_rates = [s.error_rate for s in self.services.values()]

        response_time_avg = np.mean(response_times) if response_times else 0.0
        error_rate_avg = np.mean(error_rates) if error_rates else 0.0

        # Calculate uptime percentage
        uptime_percentage = (operational_services + degraded_services * 0.7) / total_services * 100 if total_services > 0 else 0.0

        self.platform_metrics = PlatformMetrics(
            total_services=total_services,
            operational_services=operational_services,
            degraded_services=degraded_services,
            critical_services=critical_services,
            offline_services=offline_services,
            overall_health_score=overall_health_score,
            response_time_avg=response_time_avg,
            error_rate_avg=error_rate_avg,
            uptime_percentage=uptime_percentage,
            last_updated=datetime.now()
        )

    def _store_historical_data(self):
        """Store current metrics as historical data"""
        if self.platform_metrics:
            historical_point = {
                'timestamp': datetime.now(),
                'overall_health_score': self.platform_metrics.overall_health_score,
                'operational_services': self.platform_metrics.operational_services,
                'response_time_avg': self.platform_metrics.response_time_avg,
                'error_rate_avg': self.platform_metrics.error_rate_avg,
                'uptime_percentage': self.platform_metrics.uptime_percentage
            }

            self.historical_data.append(historical_point)

            # Keep only last 100 data points
            if len(self.historical_data) > 100:
                self.historical_data = self.historical_data[-100:]

    def render_dashboard(self):
        """Render the complete dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🤖 XORB Platform Orchestration Dashboard</h1>', unsafe_allow_html=True)

        # Auto-refresh checkbox
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            auto_refresh = st.checkbox("Auto Refresh", value=True)
        with col2:
            refresh_interval = st.selectbox("Refresh Interval", [5, 10, 30, 60], index=1)
        with col3:
            if st.button("🔄 Refresh Now"):
                st.experimental_rerun()

        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()

        # Platform overview
        self._render_platform_overview()

        # Service status grid
        self._render_service_status_grid()

        # Performance metrics
        self._render_performance_metrics()

        # Historical trends
        self._render_historical_trends()

        # Service details
        self._render_service_details()

        # System alerts
        self._render_system_alerts()

    def _render_platform_overview(self):
        """Render platform overview section"""
        st.header("🌐 Platform Overview")

        if not self.platform_metrics:
            st.warning("No platform metrics available. Please refresh.")
            return

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="service-card">
                <div class="metric-value status-operational">{self.platform_metrics.overall_health_score:.1%}</div>
                <div class="metric-label">Overall Health</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="service-card">
                <div class="metric-value">{self.platform_metrics.operational_services}/{self.platform_metrics.total_services}</div>
                <div class="metric-label">Services Online</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="service-card">
                <div class="metric-value">{self.platform_metrics.response_time_avg:.0f}ms</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="service-card">
                <div class="metric-value">{self.platform_metrics.error_rate_avg:.1%}</div>
                <div class="metric-label">Error Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="service-card">
                <div class="metric-value status-operational">{self.platform_metrics.uptime_percentage:.1f}%</div>
                <div class="metric-label">Uptime</div>
            </div>
            """, unsafe_allow_html=True)

        # Platform status indicator
        if self.platform_metrics.overall_health_score >= 0.9:
            status_color = "🟢"
            status_text = "All Systems Operational"
        elif self.platform_metrics.overall_health_score >= 0.7:
            status_color = "🟡"
            status_text = "Some Services Degraded"
        elif self.platform_metrics.overall_health_score >= 0.3:
            status_color = "🟠"
            status_text = "Critical Issues Detected"
        else:
            status_color = "🔴"
            status_text = "Platform Offline"

        st.markdown(f"### {status_color} {status_text}")
        st.markdown(f"*Last Updated: {self.platform_metrics.last_updated.strftime('%Y-%m-%d %H:%M:%S')}*")

    def _render_service_status_grid(self):
        """Render service status grid"""
        st.header("🔧 Service Status")

        if not self.services:
            st.warning("No service data available.")
            return

        # Create service status grid
        services_per_row = 3
        service_list = list(self.services.values())

        for i in range(0, len(service_list), services_per_row):
            cols = st.columns(services_per_row)

            for j, col in enumerate(cols):
                if i + j < len(service_list):
                    service = service_list[i + j]

                    # Status color and icon
                    if service.status == SystemStatus.OPERATIONAL:
                        status_icon = "✅"
                        status_class = "status-operational"
                    elif service.status == SystemStatus.DEGRADED:
                        status_icon = "⚠️"
                        status_class = "status-degraded"
                    elif service.status == SystemStatus.CRITICAL:
                        status_icon = "🚨"
                        status_class = "status-critical"
                    else:
                        status_icon = "❌"
                        status_class = "status-offline"

                    with col:
                        st.markdown(f"""
                        <div class="service-card">
                            <h4>{status_icon} {service.service_name.replace('_', ' ').title()}</h4>
                            <p class="{status_class}">Status: {service.status.value.title()}</p>
                            <p>Response: {service.response_time:.0f}ms</p>
                            <p>Error Rate: {service.error_rate:.1%}</p>
                            <p><small>Endpoint: {service.endpoint}</small></p>
                            <p><small>Last Check: {service.last_check.strftime('%H:%M:%S')}</small></p>
                        </div>
                        """, unsafe_allow_html=True)

    def _render_performance_metrics(self):
        """Render performance metrics charts"""
        st.header("📊 Performance Metrics")

        if not self.services:
            st.warning("No performance data available.")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Response time chart
            service_names = [s.service_name for s in self.services.values()]
            response_times = [s.response_time for s in self.services.values()]

            fig_response = go.Figure(data=go.Bar(
                x=service_names,
                y=response_times,
                marker_color=['green' if rt < 1000 else 'yellow' if rt < 3000 else 'red' for rt in response_times]
            ))
            fig_response.update_layout(
                title="Service Response Times",
                xaxis_title="Services",
                yaxis_title="Response Time (ms)",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_response, use_container_width=True)

        with col2:
            # Error rate chart
            error_rates = [s.error_rate * 100 for s in self.services.values()]

            fig_errors = go.Figure(data=go.Bar(
                x=service_names,
                y=error_rates,
                marker_color=['green' if er < 1 else 'yellow' if er < 5 else 'red' for er in error_rates]
            ))
            fig_errors.update_layout(
                title="Service Error Rates",
                xaxis_title="Services",
                yaxis_title="Error Rate (%)",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_errors, use_container_width=True)

        # Resource utilization
        if any(s.cpu_usage > 0 or s.memory_usage > 0 for s in self.services.values()):
            col3, col4 = st.columns(2)

            with col3:
                cpu_usage = [s.cpu_usage for s in self.services.values() if s.cpu_usage > 0]
                if cpu_usage:
                    fig_cpu = go.Figure(data=go.Bar(
                        x=[s.service_name for s in self.services.values() if s.cpu_usage > 0],
                        y=cpu_usage,
                        marker_color=['green' if cpu < 70 else 'yellow' if cpu < 90 else 'red' for cpu in cpu_usage]
                    ))
                    fig_cpu.update_layout(
                        title="CPU Usage",
                        xaxis_title="Services",
                        yaxis_title="CPU Usage (%)",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_cpu, use_container_width=True)

            with col4:
                memory_usage = [s.memory_usage for s in self.services.values() if s.memory_usage > 0]
                if memory_usage:
                    fig_memory = go.Figure(data=go.Bar(
                        x=[s.service_name for s in self.services.values() if s.memory_usage > 0],
                        y=memory_usage,
                        marker_color=['green' if mem < 70 else 'yellow' if mem < 90 else 'red' for mem in memory_usage]
                    ))
                    fig_memory.update_layout(
                        title="Memory Usage",
                        xaxis_title="Services",
                        yaxis_title="Memory Usage (%)",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_memory, use_container_width=True)

    def _render_historical_trends(self):
        """Render historical trend charts"""
        st.header("📈 Historical Trends")

        if len(self.historical_data) < 2:
            st.info("Collecting historical data... Check back in a few minutes.")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Health Score', 'Operational Services', 'Response Time', 'Error Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Overall health score
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['overall_health_score'],
                      name='Health Score', line=dict(color='blue')),
            row=1, col=1
        )

        # Operational services
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['operational_services'],
                      name='Operational Services', line=dict(color='green')),
            row=1, col=2
        )

        # Response time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['response_time_avg'],
                      name='Response Time', line=dict(color='orange')),
            row=2, col=1
        )

        # Error rate
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['error_rate_avg'],
                      name='Error Rate', line=dict(color='red')),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=False, title_text="Platform Trends (Last 100 Data Points)")
        st.plotly_chart(fig, use_container_width=True)

    def _render_service_details(self):
        """Render detailed service information"""
        st.header("🔍 Service Details")

        if not self.services:
            st.warning("No service details available.")
            return

        # Service selector
        selected_service = st.selectbox(
            "Select Service for Details",
            options=list(self.services.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )

        if selected_service and selected_service in self.services:
            service = self.services[selected_service]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Service Information")
                st.write(f"**Name:** {service.service_name}")
                st.write(f"**Type:** {service.service_type.value}")
                st.write(f"**Status:** {service.status.value}")
                st.write(f"**Endpoint:** {service.endpoint}")
                st.write(f"**Last Check:** {service.last_check}")

            with col2:
                st.subheader("Performance Metrics")
                st.write(f"**Response Time:** {service.response_time:.2f} ms")
                st.write(f"**Error Rate:** {service.error_rate:.2%}")
                st.write(f"**CPU Usage:** {service.cpu_usage:.1f}%")
                st.write(f"**Memory Usage:** {service.memory_usage:.1f}%")

            # Service metadata
            if service.metadata:
                st.subheader("Additional Information")
                st.json(service.metadata)

    def _render_system_alerts(self):
        """Render system alerts and notifications"""
        st.header("🚨 System Alerts")

        alerts = []

        # Generate alerts based on service status
        for service_name, service in self.services.items():
            if service.status == SystemStatus.CRITICAL:
                alerts.append({
                    'level': 'error',
                    'service': service_name,
                    'message': f"Service {service_name} is in critical state",
                    'timestamp': service.last_check
                })
            elif service.status == SystemStatus.OFFLINE:
                alerts.append({
                    'level': 'error',
                    'service': service_name,
                    'message': f"Service {service_name} is offline",
                    'timestamp': service.last_check
                })
            elif service.status == SystemStatus.DEGRADED:
                alerts.append({
                    'level': 'warning',
                    'service': service_name,
                    'message': f"Service {service_name} is degraded",
                    'timestamp': service.last_check
                })

            # Performance-based alerts
            if service.response_time > 5000:
                alerts.append({
                    'level': 'warning',
                    'service': service_name,
                    'message': f"High response time: {service.response_time:.0f}ms",
                    'timestamp': service.last_check
                })

            if service.error_rate > 0.1:
                alerts.append({
                    'level': 'warning',
                    'service': service_name,
                    'message': f"High error rate: {service.error_rate:.1%}",
                    'timestamp': service.last_check
                })

        if alerts:
            # Sort alerts by timestamp (newest first)
            alerts.sort(key=lambda x: x['timestamp'], reverse=True)

            for alert in alerts[:10]:  # Show top 10 alerts
                if alert['level'] == 'error':
                    st.error(f"🚨 **{alert['service']}**: {alert['message']} (at {alert['timestamp'].strftime('%H:%M:%S')})")
                else:
                    st.warning(f"⚠️ **{alert['service']}**: {alert['message']} (at {alert['timestamp'].strftime('%H:%M:%S')})")
        else:
            st.success("✅ No active alerts - All systems operating normally")

    async def run_dashboard(self):
        """Run the dashboard with periodic updates"""
        try:
            # Initial data collection
            await self.collect_service_health()

            # Render dashboard
            self.render_dashboard()

        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {e}")

# Streamlit app entry point
async def main():
    """Main function for running the dashboard"""
    dashboard = XORBPlatformOrchestrationDashboard()
    await dashboard.run_dashboard()

# Run the dashboard
if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        # This will only work when running with streamlit run
        dashboard = XORBPlatformOrchestrationDashboard()

        # Collect data synchronously for Streamlit
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dashboard.collect_service_health())

        # Render dashboard
        dashboard.render_dashboard()

    except Exception as e:
        print(f"Error running dashboard: {e}")
        # Fallback to CLI mode
        print("Run with: streamlit run xorb_platform_orchestration_dashboard.py")
