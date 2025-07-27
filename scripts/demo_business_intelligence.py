#!/usr/bin/env python3
"""
Business Intelligence and Reporting Demonstration

This script demonstrates the comprehensive business intelligence and reporting
capabilities of the XORB ecosystem, including executive dashboards, campaign
analytics, threat intelligence reporting, and predictive insights.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add the xorb_core package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "xorb_core"))

from intelligence.business_intelligence import (
    BusinessIntelligenceEngine,
    ReportConfiguration,
    ReportType,
    ReportFormat,
    Dashboard,
    DashboardWidget,
    SQLiteDataSource,
    MetricData
)

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class BusinessIntelligenceDemo:
    """Demonstration of business intelligence and reporting capabilities."""
    
    def __init__(self):
        self.bi_engine = BusinessIntelligenceEngine()
        self.demo_results = {}
        self.sample_data_inserted = False
    
    async def setup_demo_environment(self):
        """Set up demo environment with sample data."""
        logger.info("Setting up business intelligence demo environment")
        
        # Initialize data source
        data_source = self.bi_engine.data_sources[0]
        
        # Insert sample campaign data
        await self._insert_sample_campaigns(data_source)
        
        # Insert sample agent performance data
        await self._insert_sample_agents(data_source)
        
        # Insert sample vulnerability data
        await self._insert_sample_vulnerabilities(data_source)
        
        # Insert sample threat intelligence
        await self._insert_sample_threat_intelligence(data_source)
        
        # Insert sample compliance events
        await self._insert_sample_compliance_events(data_source)
        
        self.sample_data_inserted = True
        logger.info("Demo environment setup complete")
    
    async def _insert_sample_campaigns(self, data_source: SQLiteDataSource):
        """Insert sample campaign data."""
        campaigns = [
            {
                "id": "campaign_001",
                "title": "Network Infrastructure Assessment",
                "description": "Comprehensive network security assessment",
                "state": "completed",
                "created_at": time.time() - 86400 * 7,  # 7 days ago
                "started_at": time.time() - 86400 * 7 + 300,
                "completed_at": time.time() - 86400 * 6,
                "coordinator_id": "coordinator_001",
                "total_tasks": 12,
                "completed_tasks": 12,
                "failed_tasks": 0,
                "participating_nodes": ["node_001", "node_002", "node_003"],
                "metadata": {"priority": "high", "classification": "internal"}
            },
            {
                "id": "campaign_002",
                "title": "Web Application Penetration Test",
                "description": "Security testing of customer-facing web applications",
                "state": "completed",
                "created_at": time.time() - 86400 * 5,  # 5 days ago
                "started_at": time.time() - 86400 * 5 + 600,
                "completed_at": time.time() - 86400 * 4,
                "coordinator_id": "coordinator_002",
                "total_tasks": 8,
                "completed_tasks": 7,
                "failed_tasks": 1,
                "participating_nodes": ["node_002", "node_004"],
                "metadata": {"priority": "medium", "classification": "confidential"}
            },
            {
                "id": "campaign_003",
                "title": "Social Engineering Assessment",
                "description": "Phishing and social engineering vulnerability assessment",
                "state": "executing",
                "created_at": time.time() - 86400 * 2,  # 2 days ago
                "started_at": time.time() - 86400 * 2 + 900,
                "completed_at": None,
                "coordinator_id": "coordinator_001",
                "total_tasks": 15,
                "completed_tasks": 8,
                "failed_tasks": 1,
                "participating_nodes": ["node_001", "node_003", "node_005"],
                "metadata": {"priority": "high", "classification": "restricted"}
            },
            {
                "id": "campaign_004",
                "title": "Cloud Infrastructure Security Review",
                "description": "Assessment of cloud infrastructure security controls",
                "state": "pending",
                "created_at": time.time() - 86400,  # 1 day ago
                "started_at": None,
                "completed_at": None,
                "coordinator_id": "coordinator_003",
                "total_tasks": 20,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "participating_nodes": [],
                "metadata": {"priority": "medium", "classification": "internal"}
            }
        ]
        
        for campaign in campaigns:
            await data_source.insert_campaign_data(campaign)
    
    async def _insert_sample_agents(self, data_source: SQLiteDataSource):
        """Insert sample agent performance data."""
        agents = [
            {
                "agent_id": "recon_agent_001",
                "agent_type": "reconnaissance",
                "capabilities": ["port_scanning", "service_detection", "os_fingerprinting"],
                "total_executions": 247,
                "successful_executions": 234,
                "failed_executions": 13,
                "average_execution_time": 145.7,
                "last_execution": time.time() - 3600,
                "metadata": {"version": "2.1.0", "last_updated": "2024-01-15"}
            },
            {
                "agent_id": "exploit_agent_001",
                "agent_type": "exploitation",
                "capabilities": ["buffer_overflow", "sql_injection", "web_exploit"],
                "total_executions": 89,
                "successful_executions": 76,
                "failed_executions": 13,
                "average_execution_time": 342.8,
                "last_execution": time.time() - 7200,
                "metadata": {"version": "1.8.3", "last_updated": "2024-01-12"}
            },
            {
                "agent_id": "stealth_agent_001",
                "agent_type": "stealth",
                "capabilities": ["traffic_obfuscation", "payload_encoding", "anti_forensics"],
                "total_executions": 156,
                "successful_executions": 148,
                "failed_executions": 8,
                "average_execution_time": 67.3,
                "last_execution": time.time() - 1800,
                "metadata": {"version": "3.0.1", "last_updated": "2024-01-18"}
            },
            {
                "agent_id": "analysis_agent_001",
                "agent_type": "analysis",
                "capabilities": ["vulnerability_analysis", "risk_assessment", "report_generation"],
                "total_executions": 312,
                "successful_executions": 298,
                "failed_executions": 14,
                "average_execution_time": 289.4,
                "last_execution": time.time() - 900,
                "metadata": {"version": "2.5.0", "last_updated": "2024-01-16"}
            }
        ]
        
        for agent in agents:
            await data_source.insert_agent_performance(agent)
    
    async def _insert_sample_vulnerabilities(self, data_source: SQLiteDataSource):
        """Insert sample vulnerability data."""
        import sqlite3
        
        vulnerabilities = [
            ("vuln_001", "Critical", 9.8, "Remote Code Execution", time.time() - 86400 * 10, time.time() - 86400 * 7, "resolved", "web_server_001", 72.5),
            ("vuln_002", "High", 8.2, "SQL Injection", time.time() - 86400 * 8, None, "in_progress", "db_server_002", None),
            ("vuln_003", "Medium", 6.5, "Cross-Site Scripting", time.time() - 86400 * 6, time.time() - 86400 * 4, "resolved", "web_app_001", 48.2),
            ("vuln_004", "Low", 3.1, "Information Disclosure", time.time() - 86400 * 5, None, "open", "api_server_001", None),
            ("vuln_005", "Critical", 9.3, "Authentication Bypass", time.time() - 86400 * 3, None, "in_progress", "auth_service_001", None),
            ("vuln_006", "High", 7.8, "Privilege Escalation", time.time() - 86400 * 2, None, "open", "admin_panel_001", None),
            ("vuln_007", "Medium", 5.9, "Directory Traversal", time.time() - 86400 * 1, None, "open", "file_server_001", None)
        ]
        
        with sqlite3.connect(data_source.db_path) as conn:
            conn.executemany("""
                INSERT INTO vulnerabilities 
                (vuln_id, severity, cvss_score, category, discovered_at, resolved_at, status, affected_systems, remediation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, vulnerabilities)
    
    async def _insert_sample_threat_intelligence(self, data_source: SQLiteDataSource):
        """Insert sample threat intelligence data."""
        import sqlite3
        
        threats = [
            ("threat_001", "VirusTotal", "IP", "192.168.1.100", 0.85, "High", time.time() - 86400 * 5, time.time() - 86400 * 1, "malware,c2"),
            ("threat_002", "AlienVault OTX", "Domain", "malicious-site.com", 0.92, "Critical", time.time() - 86400 * 7, time.time() - 86400 * 2, "phishing,banking"),
            ("threat_003", "MISP", "Hash", "d41d8cd98f00b204e9800998ecf8427e", 0.78, "Medium", time.time() - 86400 * 3, time.time() - 86400 * 1, "malware,trojan"),
            ("threat_004", "ThreatFox", "URL", "http://evil-site.net/payload.exe", 0.95, "Critical", time.time() - 86400 * 2, time.time() - 3600, "malware,ransomware"),
            ("threat_005", "Internal", "IP", "10.0.0.50", 0.65, "Medium", time.time() - 86400 * 4, time.time() - 86400 * 1, "suspicious,internal")
        ]
        
        with sqlite3.connect(data_source.db_path) as conn:
            conn.executemany("""
                INSERT INTO threat_intelligence 
                (intel_id, source, indicator_type, indicator_value, confidence, severity, first_seen, last_seen, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, threats)
    
    async def _insert_sample_compliance_events(self, data_source: SQLiteDataSource):
        """Insert sample compliance events data."""
        import sqlite3
        
        compliance_events = [
            ("comp_001", "SOC2", "CC6.1", "compliant", time.time() - 86400 * 30, "Access control audit completed", "No significant findings", "Continue current practices"),
            ("comp_002", "GDPR", "Art. 32", "non_compliant", time.time() - 86400 * 15, "Data encryption assessment", "Unencrypted data found in backup systems", "Implement encryption for backup data"),
            ("comp_003", "HIPAA", "164.312(e)", "compliant", time.time() - 86400 * 20, "Transmission security review", "All transmissions properly encrypted", "Maintain current configuration"),
            ("comp_004", "ISO_27001", "A.12.1.2", "in_progress", time.time() - 86400 * 10, "Change management audit", "Ongoing assessment of change control processes", "Complete documentation review"),
            ("comp_005", "PCI_DSS", "Req. 3", "compliant", time.time() - 86400 * 25, "Cardholder data protection", "Proper encryption and key management verified", "Schedule quarterly review")
        ]
        
        with sqlite3.connect(data_source.db_path) as conn:
            conn.executemany("""
                INSERT INTO compliance_events 
                (event_id, framework, control_id, status, assessment_date, evidence, findings, remediation_plan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, compliance_events)
    
    async def demonstrate_executive_summary(self):
        """Demonstrate executive summary report generation."""
        logger.info("Generating executive summary report")
        
        config = ReportConfiguration(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.JSON,
            title="Executive Security Summary",
            description="High-level security metrics and insights for executive leadership",
            time_range=(datetime.now(timezone.utc) - timedelta(days=30), datetime.now(timezone.utc))
        )
        
        report = await self.bi_engine.generate_report(config)
        
        self.demo_results["executive_summary"] = {
            "status": "success",
            "report_size": len(json.dumps(report)),
            "generation_time": report["report_metadata"]["generation_time_seconds"],
            "key_metrics": report.get("executive_summary", {})
        }
        
        # Save report to file
        report_path = Path(__file__).parent.parent / "executive_summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Executive summary generated", report_path=str(report_path))
    
    async def demonstrate_campaign_analytics(self):
        """Demonstrate campaign analytics report."""
        logger.info("Generating campaign analytics report")
        
        config = ReportConfiguration(
            report_type=ReportType.CAMPAIGN_ANALYTICS,
            format=ReportFormat.HTML,
            title="Campaign Performance Analytics",
            description="Detailed analysis of campaign execution and performance metrics",
            time_range=(datetime.now(timezone.utc) - timedelta(days=14), datetime.now(timezone.utc)),
            include_sections=["trends", "agent_utilization", "duration_analysis"]
        )
        
        report = await self.bi_engine.generate_report(config)
        
        self.demo_results["campaign_analytics"] = {
            "status": "success",
            "format": "HTML",
            "generation_time": report.get("report_metadata", {}).get("generation_time_seconds", 0),
            "content_length": len(report.get("content", ""))
        }
        
        # Save HTML report
        report_path = Path(__file__).parent.parent / "campaign_analytics_report.html"
        with open(report_path, 'w') as f:
            f.write(report.get("content", ""))
        
        logger.info("Campaign analytics generated", report_path=str(report_path))
    
    async def demonstrate_threat_landscape(self):
        """Demonstrate threat landscape analysis."""
        logger.info("Generating threat landscape report")
        
        config = ReportConfiguration(
            report_type=ReportType.THREAT_LANDSCAPE,
            format=ReportFormat.JSON,
            title="Current Threat Landscape Analysis",
            description="Comprehensive analysis of current threat environment and indicators",
            time_range=(datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc))
        )
        
        report = await self.bi_engine.generate_report(config)
        
        self.demo_results["threat_landscape"] = {
            "status": "success",
            "threat_sources": len(report.get("threat_landscape", {}).get("threat_summary", [])),
            "geographic_coverage": len(report.get("threat_landscape", {}).get("geographic_distribution", [])),
            "generation_time": report["report_metadata"]["generation_time_seconds"]
        }
        
        # Save report
        report_path = Path(__file__).parent.parent / "threat_landscape_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Threat landscape report generated", report_path=str(report_path))
    
    async def demonstrate_vulnerability_trends(self):
        """Demonstrate vulnerability trend analysis."""
        logger.info("Generating vulnerability trends report")
        
        config = ReportConfiguration(
            report_type=ReportType.VULNERABILITY_TRENDS,
            format=ReportFormat.JSON,
            title="Vulnerability Discovery and Remediation Trends",
            description="Analysis of vulnerability lifecycle and remediation effectiveness",
            time_range=(datetime.now(timezone.utc) - timedelta(days=30), datetime.now(timezone.utc))
        )
        
        report = await self.bi_engine.generate_report(config)
        
        self.demo_results["vulnerability_trends"] = {
            "status": "success",
            "vulnerabilities_analyzed": len(report.get("vulnerability_trends", {}).get("discovery_trends", [])),
            "categories_covered": len(report.get("vulnerability_trends", {}).get("category_analysis", [])),
            "sla_metrics_included": bool(report.get("vulnerability_trends", {}).get("sla_performance"))
        }
        
        # Save report
        report_path = Path(__file__).parent.parent / "vulnerability_trends_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Vulnerability trends report generated", report_path=str(report_path))
    
    async def demonstrate_compliance_reporting(self):
        """Demonstrate compliance status reporting."""
        logger.info("Generating compliance status report")
        
        config = ReportConfiguration(
            report_type=ReportType.COMPLIANCE_STATUS,
            format=ReportFormat.JSON,
            title="Regulatory Compliance Status Assessment",
            description="Current compliance posture across multiple regulatory frameworks",
            time_range=(datetime.now(timezone.utc) - timedelta(days=90), datetime.now(timezone.utc)),
            filters={"frameworks": ["SOC2", "GDPR", "HIPAA", "ISO_27001"]}
        )
        
        report = await self.bi_engine.generate_report(config)
        
        self.demo_results["compliance_status"] = {
            "status": "success",
            "frameworks_assessed": len(report.get("compliance_status", {}).get("framework_summary", {})),
            "risk_areas_identified": len(report.get("compliance_status", {}).get("risk_areas", [])),
            "remediation_items": len(report.get("compliance_status", {}).get("remediation_plan", []))
        }
        
        # Save report
        report_path = Path(__file__).parent.parent / "compliance_status_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Compliance status report generated", report_path=str(report_path))
    
    async def demonstrate_predictive_insights(self):
        """Demonstrate predictive analytics and insights."""
        logger.info("Generating predictive insights report")
        
        config = ReportConfiguration(
            report_type=ReportType.PREDICTIVE_INSIGHTS,
            format=ReportFormat.JSON,
            title="Security Predictive Analytics and Forecasting",
            description="ML-powered predictions and capacity planning insights",
            time_range=(datetime.now(timezone.utc) - timedelta(days=60), datetime.now(timezone.utc))
        )
        
        report = await self.bi_engine.generate_report(config)
        
        self.demo_results["predictive_insights"] = {
            "status": "success",
            "threat_predictions": len(report.get("predictive_insights", {}).get("threat_forecast", {}).get("next_30_days", {}).get("high_probability_threats", [])),
            "capacity_planning": bool(report.get("predictive_insights", {}).get("capacity_planning")),
            "anomaly_predictions": len(report.get("predictive_insights", {}).get("anomaly_predictions", []))
        }
        
        # Save report
        report_path = Path(__file__).parent.parent / "predictive_insights_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Predictive insights report generated", report_path=str(report_path))
    
    async def demonstrate_dashboard_creation(self):
        """Demonstrate interactive dashboard creation."""
        logger.info("Creating interactive dashboards")
        
        # Create security operations dashboard
        security_ops_dashboard = Dashboard(
            dashboard_id="security_operations_center",
            title="Security Operations Center",
            description="Real-time security operations monitoring and metrics",
            widgets=[
                DashboardWidget(
                    widget_id="active_campaigns_metric",
                    widget_type="metric",
                    title="Active Campaigns",
                    data_source="sqlite",
                    query="SELECT COUNT(*) as count FROM campaigns WHERE state IN ('pending', 'executing')",
                    visualization_config={"color": "blue", "icon": "campaign"},
                    position={"x": 0, "y": 0, "width": 2, "height": 1}
                ),
                DashboardWidget(
                    widget_id="vulnerability_severity_chart",
                    widget_type="pie_chart",
                    title="Vulnerability Severity Distribution",
                    data_source="sqlite",
                    query="SELECT severity, COUNT(*) as count FROM vulnerabilities WHERE status != 'resolved' GROUP BY severity",
                    visualization_config={"chart_type": "pie", "colors": ["red", "orange", "yellow", "green"]},
                    position={"x": 2, "y": 0, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    widget_id="threat_intelligence_feed",
                    widget_type="table",
                    title="Recent Threat Intelligence",
                    data_source="sqlite",
                    query="SELECT indicator_value, severity, confidence, source FROM threat_intelligence ORDER BY last_seen DESC LIMIT 10",
                    visualization_config={"sortable": True, "searchable": True},
                    position={"x": 0, "y": 1, "width": 5, "height": 2}
                ),
                DashboardWidget(
                    widget_id="agent_performance_chart",
                    widget_type="bar_chart",
                    title="Agent Success Rates",
                    data_source="sqlite",
                    query="SELECT agent_type, AVG(successful_executions * 1.0 / total_executions * 100) as success_rate FROM agents GROUP BY agent_type",
                    visualization_config={"chart_type": "horizontal_bar", "color": "green"},
                    position={"x": 5, "y": 0, "width": 3, "height": 3}
                )
            ],
            auto_refresh=True,
            refresh_interval=60,  # 1 minute
            access_control=["security_analysts", "security_managers", "executives"]
        )
        
        # Create executive dashboard
        executive_dashboard = Dashboard(
            dashboard_id="executive_security_dashboard",
            title="Executive Security Dashboard",
            description="High-level security metrics and KPIs for executive leadership",
            widgets=[
                DashboardWidget(
                    widget_id="security_posture_score",
                    widget_type="gauge",
                    title="Overall Security Posture",
                    data_source="calculated",
                    query="security_posture_calculation",
                    visualization_config={"min": 0, "max": 100, "thresholds": [60, 80]},
                    position={"x": 0, "y": 0, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    widget_id="compliance_status_grid",
                    widget_type="status_grid",
                    title="Compliance Framework Status",
                    data_source="sqlite",
                    query="SELECT framework, status, COUNT(*) FROM compliance_events GROUP BY framework, status",
                    visualization_config={"status_mapping": {"compliant": "green", "non_compliant": "red", "in_progress": "yellow"}},
                    position={"x": 3, "y": 0, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    widget_id="roi_metrics",
                    widget_type="metric_group",
                    title="Security ROI Metrics",
                    data_source="calculated",
                    query="roi_calculation",
                    visualization_config={"metrics": ["investment", "benefits", "roi_percentage"]},
                    position={"x": 0, "y": 2, "width": 6, "height": 1}
                )
            ],
            auto_refresh=True,
            refresh_interval=300,  # 5 minutes
            access_control=["executives", "security_leadership"]
        )
        
        # Register dashboards
        soc_dashboard_id = self.bi_engine.create_dashboard(security_ops_dashboard)
        exec_dashboard_id = self.bi_engine.create_dashboard(executive_dashboard)
        
        self.demo_results["dashboard_creation"] = {
            "status": "success",
            "dashboards_created": 2,
            "soc_dashboard_id": soc_dashboard_id,
            "executive_dashboard_id": exec_dashboard_id,
            "total_widgets": len(security_ops_dashboard.widgets) + len(executive_dashboard.widgets)
        }
        
        # Save dashboard configurations
        dashboards_config = {
            "security_operations_center": {
                "title": security_ops_dashboard.title,
                "description": security_ops_dashboard.description,
                "widgets": [
                    {
                        "widget_id": widget.widget_id,
                        "title": widget.title,
                        "type": widget.widget_type,
                        "position": widget.position
                    }
                    for widget in security_ops_dashboard.widgets
                ]
            },
            "executive_security_dashboard": {
                "title": executive_dashboard.title,
                "description": executive_dashboard.description,
                "widgets": [
                    {
                        "widget_id": widget.widget_id,
                        "title": widget.title,
                        "type": widget.widget_type,
                        "position": widget.position
                    }
                    for widget in executive_dashboard.widgets
                ]
            }
        }
        
        config_path = Path(__file__).parent.parent / "dashboard_configurations.json"
        with open(config_path, 'w') as f:
            json.dump(dashboards_config, f, indent=2)
        
        logger.info("Dashboards created", 
                   soc_dashboard=soc_dashboard_id,
                   executive_dashboard=exec_dashboard_id,
                   config_path=str(config_path))
    
    async def demonstrate_scheduled_reporting(self):
        """Demonstrate scheduled report generation."""
        logger.info("Setting up scheduled reports")
        
        # Schedule weekly executive summary
        weekly_exec_config = ReportConfiguration(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.PDF,
            title="Weekly Executive Security Summary",
            description="Weekly high-level security metrics for executive team",
            time_range=(datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc)),
            recipients=["ceo@company.com", "ciso@company.com", "coo@company.com"],
            schedule="0 9 * * MON"  # Every Monday at 9 AM
        )
        
        # Schedule daily operational metrics
        daily_ops_config = ReportConfiguration(
            report_type=ReportType.OPERATIONAL_METRICS,
            format=ReportFormat.HTML,
            title="Daily Security Operations Report",
            description="Daily operational metrics for security operations team",
            time_range=(datetime.now(timezone.utc) - timedelta(days=1), datetime.now(timezone.utc)),
            recipients=["security-team@company.com"],
            schedule="0 8 * * *"  # Every day at 8 AM
        )
        
        # Schedule monthly compliance report
        monthly_compliance_config = ReportConfiguration(
            report_type=ReportType.COMPLIANCE_STATUS,
            format=ReportFormat.EXCEL,
            title="Monthly Compliance Assessment",
            description="Monthly compliance status across all frameworks",
            time_range=(datetime.now(timezone.utc) - timedelta(days=30), datetime.now(timezone.utc)),
            recipients=["compliance@company.com", "legal@company.com"],
            schedule="0 10 1 * *"  # First day of month at 10 AM
        )
        
        # Schedule reports
        weekly_report_id = await self.bi_engine.schedule_report(weekly_exec_config)
        daily_report_id = await self.bi_engine.schedule_report(daily_ops_config)
        monthly_report_id = await self.bi_engine.schedule_report(monthly_compliance_config)
        
        self.demo_results["scheduled_reporting"] = {
            "status": "success",
            "reports_scheduled": 3,
            "weekly_executive_id": weekly_report_id,
            "daily_operations_id": daily_report_id,
            "monthly_compliance_id": monthly_report_id
        }
        
        logger.info("Scheduled reports configured",
                   weekly_id=weekly_report_id,
                   daily_id=daily_report_id,
                   monthly_id=monthly_report_id)
    
    async def generate_demonstration_summary(self):
        """Generate comprehensive demonstration summary."""
        logger.info("Generating demonstration summary")
        
        summary = {
            "demonstration_overview": {
                "timestamp": time.time(),
                "total_demonstrations": len(self.demo_results),
                "successful_demonstrations": len([r for r in self.demo_results.values() if r.get("status") == "success"]),
                "sample_data_inserted": self.sample_data_inserted
            },
            "business_intelligence_capabilities": {
                "report_types_demonstrated": list(self.demo_results.keys()),
                "output_formats_used": ["JSON", "HTML", "PDF", "EXCEL"],
                "dashboards_created": self.demo_results.get("dashboard_creation", {}).get("dashboards_created", 0),
                "scheduled_reports": self.demo_results.get("scheduled_reporting", {}).get("reports_scheduled", 0)
            },
            "demonstration_results": self.demo_results,
            "generated_artifacts": [
                "executive_summary_report.json",
                "campaign_analytics_report.html", 
                "threat_landscape_report.json",
                "vulnerability_trends_report.json",
                "compliance_status_report.json",
                "predictive_insights_report.json",
                "dashboard_configurations.json"
            ],
            "analytics_database": {
                "location": self.bi_engine.data_sources[0].db_path,
                "tables_populated": ["campaigns", "agents", "vulnerabilities", "threat_intelligence", "compliance_events"]
            },
            "performance_metrics": {
                "average_report_generation_time": sum(
                    r.get("generation_time", 0) for r in self.demo_results.values() 
                    if "generation_time" in r
                ) / len([r for r in self.demo_results.values() if "generation_time" in r]) if self.demo_results else 0,
                "total_data_points_processed": sum(
                    len(r.get("data_points", [])) for r in self.demo_results.values()
                    if "data_points" in r
                )
            }
        }
        
        # Save summary
        summary_path = Path(__file__).parent.parent / "business_intelligence_demo_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Demonstration summary saved", summary_path=str(summary_path))
        return summary


async def main():
    """Main demonstration function."""
    print("📊 XORB Business Intelligence and Reporting Demonstration")
    print("=" * 65)
    
    demo = BusinessIntelligenceDemo()
    
    try:
        # Setup demo environment
        print("\n🔧 Setting up demonstration environment...")
        await demo.setup_demo_environment()
        
        # Demonstrate report generation
        print("\n📈 Demonstrating executive summary generation...")
        await demo.demonstrate_executive_summary()
        
        print("\n📊 Demonstrating campaign analytics...")
        await demo.demonstrate_campaign_analytics()
        
        print("\n🎯 Demonstrating threat landscape analysis...")
        await demo.demonstrate_threat_landscape()
        
        print("\n🔍 Demonstrating vulnerability trend analysis...")
        await demo.demonstrate_vulnerability_trends()
        
        print("\n✅ Demonstrating compliance reporting...")
        await demo.demonstrate_compliance_reporting()
        
        print("\n🔮 Demonstrating predictive insights...")
        await demo.demonstrate_predictive_insights()
        
        # Demonstrate dashboards
        print("\n📱 Demonstrating dashboard creation...")
        await demo.demonstrate_dashboard_creation()
        
        # Demonstrate scheduling
        print("\n⏰ Demonstrating scheduled reporting...")
        await demo.demonstrate_scheduled_reporting()
        
        # Generate summary
        print("\n📋 Generating demonstration summary...")
        summary = await demo.generate_demonstration_summary()
        
        print("\n✅ Business Intelligence Demonstration Complete!")
        print(f"📊 Total demonstrations: {len(demo.demo_results)}")
        print(f"📈 Reports generated: {summary['demonstration_overview']['total_demonstrations']}")
        print(f"📱 Dashboards created: {summary['business_intelligence_capabilities']['dashboards_created']}")
        print(f"⏰ Scheduled reports: {summary['business_intelligence_capabilities']['scheduled_reports']}")
        
        # Print demonstration results
        print("\n📋 Demonstration Results:")
        for demo_name, results in demo.demo_results.items():
            status = results.get("status", "completed")
            print(f"  • {demo_name.replace('_', ' ').title()}: {status.upper()}")
        
        print(f"\n📄 Generated artifacts saved to project directory")
        print(f"🗄️ Analytics database: {demo.bi_engine.data_sources[0].db_path}")
        
    except Exception as e:
        logger.error("Business Intelligence demonstration failed", error=str(e), exc_info=True)
        print(f"\n❌ Demonstration failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())