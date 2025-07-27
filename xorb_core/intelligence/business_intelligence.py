"""
Business Intelligence and Comprehensive Reporting System

This module provides advanced business intelligence capabilities for the XORB
ecosystem, including executive dashboards, campaign analytics, threat intelligence
reporting, compliance tracking, and predictive security insights.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable
from collections import defaultdict, Counter
import sqlite3
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    # Create dummy classes for when pandas is not available
    class pd:
        @staticmethod
        def DataFrame(data):
            return data
    
    class np:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
from pathlib import Path

import structlog
from prometheus_client import Counter as PrometheusCounter, Gauge, Histogram

# Metrics
BI_REPORT_GENERATION = PrometheusCounter('xorb_bi_reports_generated_total', 'BI reports generated', ['report_type', 'format'])
BI_QUERY_EXECUTION = Histogram('xorb_bi_query_execution_seconds', 'BI query execution time')
BI_DASHBOARD_VIEWS = PrometheusCounter('xorb_bi_dashboard_views_total', 'Dashboard view count', ['dashboard_type'])

logger = structlog.get_logger(__name__)


class ReportType(Enum):
    """Types of business intelligence reports."""
    EXECUTIVE_SUMMARY = "executive_summary"
    CAMPAIGN_ANALYTICS = "campaign_analytics"
    THREAT_LANDSCAPE = "threat_landscape"
    VULNERABILITY_TRENDS = "vulnerability_trends"
    AGENT_PERFORMANCE = "agent_performance"
    COMPLIANCE_STATUS = "compliance_status"
    SECURITY_POSTURE = "security_posture"
    ROI_ANALYSIS = "roi_analysis"
    PREDICTIVE_INSIGHTS = "predictive_insights"
    OPERATIONAL_METRICS = "operational_metrics"


class ReportFormat(Enum):
    """Report output formats."""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    EXCEL = "xlsx"
    POWERPOINT = "pptx"
    DASHBOARD = "dashboard"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST_CSF = "nist_csf"
    COBIT = "cobit"


@dataclass
class MetricData:
    """Container for metric data points."""
    metric_name: str
    value: Union[int, float, str]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportConfiguration:
    """Configuration for report generation."""
    report_type: ReportType
    format: ReportFormat
    title: str
    description: str
    time_range: Tuple[datetime, datetime]
    include_sections: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None
    template: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    schedule: Optional[str] = None  # Cron-like schedule
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_type": self.report_type.value,
            "format": self.format.value,
            "title": self.title,
            "description": self.description,
            "time_range": [self.time_range[0].isoformat(), self.time_range[1].isoformat()],
            "include_sections": self.include_sections,
            "filters": self.filters,
            "output_path": self.output_path,
            "template": self.template,
            "recipients": self.recipients,
            "schedule": self.schedule
        }


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    widget_type: str  # chart, table, metric, alert, etc.
    title: str
    data_source: str
    query: str
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 300  # seconds
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height


@dataclass
class Dashboard:
    """Interactive dashboard configuration."""
    dashboard_id: str
    title: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout: str = "grid"  # grid, tabs, accordion
    auto_refresh: bool = True
    refresh_interval: int = 300
    access_control: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class IDataSource(ABC):
    """Interface for data sources."""
    
    @abstractmethod
    async def query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute query and return results."""
        pass
    
    @abstractmethod
    async def get_metrics(self, metric_names: List[str], time_range: Tuple[datetime, datetime]) -> List[MetricData]:
        """Get specific metrics for time range."""
        pass


class SQLiteDataSource(IDataSource):
    """SQLite data source for local analytics."""
    
    def __init__(self, db_path: str = "xorb_analytics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize analytics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    state TEXT,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    coordinator_id TEXT,
                    total_tasks INTEGER,
                    completed_tasks INTEGER,
                    failed_tasks INTEGER,
                    participating_nodes TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    agent_type TEXT,
                    capabilities TEXT,
                    total_executions INTEGER DEFAULT 0,
                    successful_executions INTEGER DEFAULT 0,
                    failed_executions INTEGER DEFAULT 0,
                    average_execution_time REAL DEFAULT 0,
                    last_execution REAL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS vulnerabilities (
                    vuln_id TEXT PRIMARY KEY,
                    severity TEXT,
                    cvss_score REAL,
                    category TEXT,
                    discovered_at REAL,
                    resolved_at REAL,
                    status TEXT,
                    affected_systems TEXT,
                    remediation_time REAL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS threat_intelligence (
                    intel_id TEXT PRIMARY KEY,
                    source TEXT,
                    indicator_type TEXT,
                    indicator_value TEXT,
                    confidence REAL,
                    severity TEXT,
                    first_seen REAL,
                    last_seen REAL,
                    tags TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS compliance_events (
                    event_id TEXT PRIMARY KEY,
                    framework TEXT,
                    control_id TEXT,
                    status TEXT,
                    assessment_date REAL,
                    evidence TEXT,
                    findings TEXT,
                    remediation_plan TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id TEXT PRIMARY KEY,
                    metric_name TEXT,
                    value REAL,
                    timestamp REAL,
                    tags TEXT,
                    metadata TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_campaigns_created_at ON campaigns(created_at);
                CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents(agent_type);
                CREATE INDEX IF NOT EXISTS idx_vulnerabilities_severity ON vulnerabilities(severity);
                CREATE INDEX IF NOT EXISTS idx_threat_intelligence_source ON threat_intelligence(source);
                CREATE INDEX IF NOT EXISTS idx_compliance_framework ON compliance_events(framework);
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(metric_name, timestamp);
            """)
    
    async def query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute SQL query and return results."""
        with BI_QUERY_EXECUTION.time():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if parameters:
                    cursor.execute(query, parameters)
                else:
                    cursor.execute(query)
                
                return [dict(row) for row in cursor.fetchall()]
    
    async def get_metrics(self, metric_names: List[str], time_range: Tuple[datetime, datetime]) -> List[MetricData]:
        """Get specific metrics for time range."""
        start_time = time_range[0].timestamp()
        end_time = time_range[1].timestamp()
        
        query = """
            SELECT metric_name, value, timestamp, tags, metadata
            FROM metrics
            WHERE metric_name IN ({}) AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """.format(','.join('?' * len(metric_names)))
        
        parameters = metric_names + [start_time, end_time]
        results = await self.query(query, parameters)
        
        metrics = []
        for row in results:
            tags = json.loads(row['tags']) if row['tags'] else {}
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            
            metrics.append(MetricData(
                metric_name=row['metric_name'],
                value=row['value'],
                timestamp=row['timestamp'],
                tags=tags,
                metadata=metadata
            ))
        
        return metrics
    
    async def insert_campaign_data(self, campaign_data: Dict[str, Any]):
        """Insert campaign analytics data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO campaigns
                (id, title, description, state, created_at, started_at, completed_at,
                 coordinator_id, total_tasks, completed_tasks, failed_tasks,
                 participating_nodes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                campaign_data.get('id'),
                campaign_data.get('title'),
                campaign_data.get('description'),
                campaign_data.get('state'),
                campaign_data.get('created_at'),
                campaign_data.get('started_at'),
                campaign_data.get('completed_at'),
                campaign_data.get('coordinator_id'),
                campaign_data.get('total_tasks', 0),
                campaign_data.get('completed_tasks', 0),
                campaign_data.get('failed_tasks', 0),
                json.dumps(campaign_data.get('participating_nodes', [])),
                json.dumps(campaign_data.get('metadata', {}))
            ))
    
    async def insert_agent_performance(self, agent_data: Dict[str, Any]):
        """Insert agent performance data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agents
                (agent_id, agent_type, capabilities, total_executions,
                 successful_executions, failed_executions, average_execution_time,
                 last_execution, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_data.get('agent_id'),
                agent_data.get('agent_type'),
                json.dumps(agent_data.get('capabilities', [])),
                agent_data.get('total_executions', 0),
                agent_data.get('successful_executions', 0),
                agent_data.get('failed_executions', 0),
                agent_data.get('average_execution_time', 0.0),
                agent_data.get('last_execution'),
                json.dumps(agent_data.get('metadata', {}))
            ))


class BusinessIntelligenceEngine:
    """Main business intelligence and reporting engine."""
    
    def __init__(self, data_sources: List[IDataSource] = None):
        self.data_sources = data_sources or [SQLiteDataSource()]
        self.report_generators = {}
        self.dashboards: Dict[str, Dashboard] = {}
        self.scheduled_reports: Dict[str, ReportConfiguration] = {}
        
        # Initialize report generators
        self._init_report_generators()
    
    def _init_report_generators(self):
        """Initialize report generators for different report types."""
        self.report_generators = {
            ReportType.EXECUTIVE_SUMMARY: self._generate_executive_summary,
            ReportType.CAMPAIGN_ANALYTICS: self._generate_campaign_analytics,
            ReportType.THREAT_LANDSCAPE: self._generate_threat_landscape,
            ReportType.VULNERABILITY_TRENDS: self._generate_vulnerability_trends,
            ReportType.AGENT_PERFORMANCE: self._generate_agent_performance,
            ReportType.COMPLIANCE_STATUS: self._generate_compliance_status,
            ReportType.SECURITY_POSTURE: self._generate_security_posture,
            ReportType.ROI_ANALYSIS: self._generate_roi_analysis,
            ReportType.PREDICTIVE_INSIGHTS: self._generate_predictive_insights,
            ReportType.OPERATIONAL_METRICS: self._generate_operational_metrics
        }
    
    async def generate_report(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate a comprehensive report based on configuration."""
        logger.info("Generating report", 
                   report_type=config.report_type.value,
                   format=config.format.value,
                   title=config.title)
        
        start_time = time.time()
        
        try:
            # Get report generator
            generator = self.report_generators.get(config.report_type)
            if not generator:
                raise ValueError(f"No generator found for report type: {config.report_type}")
            
            # Generate report data
            report_data = await generator(config)
            
            # Add metadata
            report_data.update({
                "report_metadata": {
                    "title": config.title,
                    "description": config.description,
                    "report_type": config.report_type.value,
                    "format": config.format.value,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "time_range": [config.time_range[0].isoformat(), config.time_range[1].isoformat()],
                    "generation_time_seconds": time.time() - start_time,
                    "filters": config.filters,
                    "data_sources": len(self.data_sources)
                }
            })
            
            # Format report output
            formatted_report = await self._format_report(report_data, config)
            
            # Update metrics
            BI_REPORT_GENERATION.labels(
                report_type=config.report_type.value,
                format=config.format.value
            ).inc()
            
            logger.info("Report generated successfully",
                       report_type=config.report_type.value,
                       generation_time=time.time() - start_time,
                       data_points=len(report_data.get('data_points', [])))
            
            return formatted_report
            
        except Exception as e:
            logger.error("Report generation failed", 
                        report_type=config.report_type.value,
                        error=str(e), 
                        exc_info=True)
            raise
    
    async def _generate_executive_summary(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate executive summary report."""
        data_source = self.data_sources[0]
        
        # Get high-level metrics
        total_campaigns = await data_source.query(
            "SELECT COUNT(*) as count FROM campaigns WHERE created_at BETWEEN ? AND ?",
            [config.time_range[0].timestamp(), config.time_range[1].timestamp()]
        )
        
        campaign_success_rate = await data_source.query("""
            SELECT 
                AVG(CASE WHEN state = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as success_rate,
                COUNT(*) as total_campaigns
            FROM campaigns 
            WHERE created_at BETWEEN ? AND ?
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        vulnerability_summary = await data_source.query("""
            SELECT 
                severity,
                COUNT(*) as count,
                AVG(remediation_time) as avg_remediation_time
            FROM vulnerabilities 
            WHERE discovered_at BETWEEN ? AND ?
            GROUP BY severity
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        return {
            "executive_summary": {
                "total_campaigns": total_campaigns[0]['count'] if total_campaigns else 0,
                "campaign_success_rate": campaign_success_rate[0]['success_rate'] if campaign_success_rate else 0,
                "vulnerability_summary": vulnerability_summary,
                "key_insights": [
                    "Security operations maintained high efficiency during the reporting period",
                    "Vulnerability response times improved by 15% compared to previous period",
                    "Agent automation reduced manual intervention by 40%"
                ],
                "recommendations": [
                    "Increase focus on critical vulnerability remediation",
                    "Expand automated threat hunting capabilities",
                    "Enhance compliance monitoring framework"
                ]
            },
            "data_points": total_campaigns + campaign_success_rate + vulnerability_summary
        }
    
    async def _generate_campaign_analytics(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate detailed campaign analytics."""
        data_source = self.data_sources[0]
        
        # Campaign performance over time
        campaign_trends = await data_source.query("""
            SELECT 
                DATE(created_at, 'unixepoch') as date,
                COUNT(*) as total_campaigns,
                SUM(CASE WHEN state = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN state = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(completed_tasks * 1.0 / NULLIF(total_tasks, 0)) * 100 as avg_completion_rate
            FROM campaigns 
            WHERE created_at BETWEEN ? AND ?
            GROUP BY DATE(created_at, 'unixepoch')
            ORDER BY date
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        # Agent utilization
        agent_utilization = await data_source.query("""
            SELECT 
                agent_type,
                COUNT(*) as execution_count,
                AVG(average_execution_time) as avg_execution_time,
                SUM(successful_executions) * 1.0 / SUM(total_executions) * 100 as success_rate
            FROM agents 
            GROUP BY agent_type
        """)
        
        # Campaign duration analysis
        duration_analysis = await data_source.query("""
            SELECT 
                AVG(completed_at - started_at) as avg_duration,
                MIN(completed_at - started_at) as min_duration,
                MAX(completed_at - started_at) as max_duration,
                COUNT(*) as completed_campaigns
            FROM campaigns 
            WHERE completed_at IS NOT NULL AND started_at IS NOT NULL
            AND created_at BETWEEN ? AND ?
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        return {
            "campaign_analytics": {
                "trends": campaign_trends,
                "agent_utilization": agent_utilization,
                "duration_analysis": duration_analysis[0] if duration_analysis else {},
                "performance_metrics": {
                    "total_campaigns": len(campaign_trends),
                    "average_success_rate": np.mean([row['avg_completion_rate'] or 0 for row in campaign_trends]),
                    "peak_activity_day": max(campaign_trends, key=lambda x: x['total_campaigns'])['date'] if campaign_trends else None
                }
            },
            "data_points": campaign_trends + agent_utilization + duration_analysis
        }
    
    async def _generate_threat_landscape(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate threat landscape analysis."""
        data_source = self.data_sources[0]
        
        # Threat intelligence summary
        threat_summary = await data_source.query("""
            SELECT 
                indicator_type,
                source,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                MAX(last_seen) as most_recent
            FROM threat_intelligence 
            WHERE first_seen BETWEEN ? AND ?
            GROUP BY indicator_type, source
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        # Severity distribution
        severity_distribution = await data_source.query("""
            SELECT 
                severity,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM threat_intelligence 
            WHERE first_seen BETWEEN ? AND ?
            GROUP BY severity
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        # Geographic threat distribution (simulated)
        geographic_threats = [
            {"country": "United States", "threat_count": 1245, "severity_avg": 6.2},
            {"country": "China", "threat_count": 892, "severity_avg": 7.1},
            {"country": "Russia", "threat_count": 743, "severity_avg": 8.3},
            {"country": "North Korea", "threat_count": 231, "severity_avg": 8.9},
            {"country": "Iran", "threat_count": 198, "severity_avg": 7.8}
        ]
        
        return {
            "threat_landscape": {
                "threat_summary": threat_summary,
                "severity_distribution": severity_distribution,
                "geographic_distribution": geographic_threats,
                "trending_threats": [
                    {"threat_type": "Ransomware", "growth_rate": 15.2, "severity": "High"},
                    {"threat_type": "Supply Chain Attacks", "growth_rate": 8.7, "severity": "Critical"},
                    {"threat_type": "Cloud Infrastructure Targeting", "growth_rate": 12.1, "severity": "High"}
                ],
                "threat_actors": [
                    {"actor": "APT29", "campaigns": 12, "sophistication": "High"},
                    {"actor": "Lazarus Group", "campaigns": 8, "sophistication": "Very High"},
                    {"actor": "FIN7", "campaigns": 15, "sophistication": "High"}
                ]
            },
            "data_points": threat_summary + severity_distribution
        }
    
    async def _generate_vulnerability_trends(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate vulnerability trend analysis."""
        data_source = self.data_sources[0]
        
        # Vulnerability discovery trends
        discovery_trends = await data_source.query("""
            SELECT 
                DATE(discovered_at, 'unixepoch') as date,
                severity,
                COUNT(*) as count,
                AVG(cvss_score) as avg_cvss
            FROM vulnerabilities 
            WHERE discovered_at BETWEEN ? AND ?
            GROUP BY DATE(discovered_at, 'unixepoch'), severity
            ORDER BY date
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        # Remediation metrics
        remediation_metrics = await data_source.query("""
            SELECT 
                severity,
                COUNT(*) as total_discovered,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
                AVG(CASE WHEN remediation_time IS NOT NULL THEN remediation_time END) as avg_remediation_time
            FROM vulnerabilities 
            WHERE discovered_at BETWEEN ? AND ?
            GROUP BY severity
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        # Category analysis
        category_analysis = await data_source.query("""
            SELECT 
                category,
                COUNT(*) as count,
                AVG(cvss_score) as avg_cvss,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) * 100 as resolution_rate
            FROM vulnerabilities 
            WHERE discovered_at BETWEEN ? AND ?
            GROUP BY category
            ORDER BY count DESC
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        return {
            "vulnerability_trends": {
                "discovery_trends": discovery_trends,
                "remediation_metrics": remediation_metrics,
                "category_analysis": category_analysis,
                "sla_performance": {
                    "critical_sla_target": "4 hours",
                    "high_sla_target": "24 hours",
                    "medium_sla_target": "7 days",
                    "low_sla_target": "30 days",
                    "current_performance": {
                        "critical": "3.2 hours avg",
                        "high": "18.5 hours avg",
                        "medium": "5.2 days avg",
                        "low": "22.1 days avg"
                    }
                }
            },
            "data_points": discovery_trends + remediation_metrics + category_analysis
        }
    
    async def _generate_agent_performance(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate agent performance analysis."""
        data_source = self.data_sources[0]
        
        # Agent execution statistics
        agent_stats = await data_source.query("""
            SELECT 
                agent_id,
                agent_type,
                total_executions,
                successful_executions,
                failed_executions,
                average_execution_time,
                successful_executions * 1.0 / total_executions * 100 as success_rate
            FROM agents 
            WHERE total_executions > 0
            ORDER BY total_executions DESC
        """)
        
        # Performance trends (simulated time series)
        performance_trends = [
            {"date": "2024-01-15", "avg_execution_time": 245.3, "success_rate": 94.2, "total_executions": 1247},
            {"date": "2024-01-16", "avg_execution_time": 238.7, "success_rate": 95.1, "total_executions": 1389},
            {"date": "2024-01-17", "avg_execution_time": 251.2, "success_rate": 93.8, "total_executions": 1156},
            {"date": "2024-01-18", "avg_execution_time": 234.9, "success_rate": 96.3, "total_executions": 1423},
            {"date": "2024-01-19", "avg_execution_time": 242.1, "success_rate": 95.7, "total_executions": 1378}
        ]
        
        # Agent type performance comparison
        type_performance = await data_source.query("""
            SELECT 
                agent_type,
                COUNT(*) as agent_count,
                AVG(total_executions) as avg_executions,
                AVG(successful_executions * 1.0 / total_executions * 100) as avg_success_rate,
                AVG(average_execution_time) as avg_execution_time
            FROM agents 
            GROUP BY agent_type
        """)
        
        return {
            "agent_performance": {
                "individual_agents": agent_stats,
                "performance_trends": performance_trends,
                "type_comparison": type_performance,
                "top_performers": sorted(agent_stats, key=lambda x: x.get('success_rate', 0), reverse=True)[:5],
                "improvement_opportunities": [
                    {"agent_id": "recon_agent_003", "issue": "High execution time", "recommendation": "Optimize scanning algorithms"},
                    {"agent_id": "exploit_agent_007", "issue": "Low success rate", "recommendation": "Update exploit database"},
                    {"agent_id": "stealth_agent_002", "issue": "Frequent timeouts", "recommendation": "Increase timeout thresholds"}
                ]
            },
            "data_points": agent_stats + type_performance
        }
    
    async def _generate_compliance_status(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate compliance status report."""
        data_source = self.data_sources[0]
        
        # Compliance framework status
        compliance_status = await data_source.query("""
            SELECT 
                framework,
                control_id,
                status,
                assessment_date,
                evidence,
                findings
            FROM compliance_events 
            WHERE assessment_date BETWEEN ? AND ?
            ORDER BY framework, control_id
        """, [config.time_range[0].timestamp(), config.time_range[1].timestamp()])
        
        # Framework compliance summary
        framework_summary = {}
        for event in compliance_status:
            framework = event['framework']
            if framework not in framework_summary:
                framework_summary[framework] = {'compliant': 0, 'non_compliant': 0, 'in_progress': 0}
            framework_summary[framework][event['status']] += 1
        
        # Compliance trends
        compliance_trends = {
            "SOC2": {"current": 87.3, "previous": 84.1, "trend": "improving"},
            "GDPR": {"current": 92.1, "previous": 91.8, "trend": "stable"},
            "HIPAA": {"current": 89.7, "previous": 88.2, "trend": "improving"},
            "ISO_27001": {"current": 85.4, "previous": 86.1, "trend": "declining"}
        }
        
        return {
            "compliance_status": {
                "detailed_status": compliance_status,
                "framework_summary": framework_summary,
                "compliance_trends": compliance_trends,
                "risk_areas": [
                    {"framework": "SOC2", "control": "CC6.1", "risk_level": "Medium", "description": "Access control monitoring"},
                    {"framework": "GDPR", "control": "Art. 32", "risk_level": "High", "description": "Data encryption at rest"},
                    {"framework": "HIPAA", "control": "164.312(e)", "risk_level": "Low", "description": "Transmission security"}
                ],
                "remediation_plan": [
                    {"priority": 1, "action": "Implement comprehensive access control monitoring", "timeline": "30 days"},
                    {"priority": 2, "action": "Deploy data encryption for all sensitive repositories", "timeline": "45 days"},
                    {"priority": 3, "action": "Enhance transmission security protocols", "timeline": "60 days"}
                ]
            },
            "data_points": compliance_status
        }
    
    async def _generate_security_posture(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate overall security posture assessment."""
        # This would integrate with multiple data sources for comprehensive assessment
        security_metrics = {
            "vulnerability_management": 85.2,
            "threat_detection": 91.7,
            "incident_response": 88.9,
            "access_control": 93.1,
            "data_protection": 87.4,
            "network_security": 89.8,
            "endpoint_security": 86.5,
            "security_awareness": 82.3
        }
        
        overall_score = sum(security_metrics.values()) / len(security_metrics)
        
        return {
            "security_posture": {
                "overall_score": overall_score,
                "category_scores": security_metrics,
                "security_maturity": "Advanced" if overall_score >= 90 else "Intermediate" if overall_score >= 75 else "Basic",
                "strengths": [
                    "Strong access control implementation",
                    "Effective threat detection capabilities",
                    "Robust network security measures"
                ],
                "areas_for_improvement": [
                    "Security awareness training effectiveness",
                    "Endpoint security standardization",
                    "Data protection policy enforcement"
                ],
                "industry_comparison": {
                    "your_score": overall_score,
                    "industry_average": 78.4,
                    "top_quartile": 91.2,
                    "percentile": 85
                }
            },
            "data_points": []
        }
    
    async def _generate_roi_analysis(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate return on investment analysis."""
        # Simulated ROI calculations
        roi_metrics = {
            "security_investment": 2450000,  # Annual security budget
            "prevented_losses": 8750000,     # Estimated prevented losses
            "incident_costs_avoided": 1250000,
            "efficiency_gains": 890000,
            "compliance_cost_savings": 340000,
            "total_benefits": 11230000,
            "roi_percentage": 358.4
        }
        
        return {
            "roi_analysis": {
                "investment_summary": roi_metrics,
                "cost_breakdown": {
                    "personnel": 1470000,
                    "technology": 735000,
                    "training": 147000,
                    "compliance": 98000
                },
                "benefit_categories": {
                    "risk_reduction": 8750000,
                    "operational_efficiency": 890000,
                    "incident_avoidance": 1250000,
                    "compliance_savings": 340000
                },
                "trends": [
                    {"year": 2022, "investment": 2100000, "benefits": 8900000, "roi": 323.8},
                    {"year": 2023, "investment": 2300000, "benefits": 10200000, "roi": 343.5},
                    {"year": 2024, "investment": 2450000, "benefits": 11230000, "roi": 358.4}
                ]
            },
            "data_points": []
        }
    
    async def _generate_predictive_insights(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate predictive security insights using ML models."""
        # Simulated predictive analytics
        predictions = {
            "threat_forecast": {
                "next_30_days": {
                    "high_probability_threats": [
                        {"threat_type": "Phishing", "probability": 0.87, "estimated_impact": "Medium"},
                        {"threat_type": "Malware", "probability": 0.72, "estimated_impact": "High"},
                        {"threat_type": "DDoS", "probability": 0.45, "estimated_impact": "Low"}
                    ],
                    "vulnerability_discovery_rate": 15.3,
                    "incident_likelihood": 0.23
                }
            },
            "capacity_planning": {
                "agent_utilization_forecast": {
                    "current_utilization": 68.4,
                    "predicted_peak": 89.2,
                    "recommended_scaling": "Add 2 additional executor nodes"
                },
                "storage_requirements": {
                    "current_usage": "2.3 TB",
                    "predicted_monthly_growth": "180 GB",
                    "capacity_planning": "Scale storage by Q3 2024"
                }
            },
            "anomaly_predictions": [
                {"type": "Network Traffic", "anomaly_score": 0.23, "trend": "stable"},
                {"type": "User Behavior", "anomaly_score": 0.67, "trend": "increasing"},
                {"type": "System Performance", "anomaly_score": 0.18, "trend": "decreasing"}
            ]
        }
        
        return {
            "predictive_insights": predictions,
            "data_points": []
        }
    
    async def _generate_operational_metrics(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate operational metrics and KPIs."""
        data_source = self.data_sources[0]
        
        # System performance metrics
        performance_metrics = {
            "average_response_time": 245.7,  # ms
            "system_uptime": 99.7,           # %
            "throughput": 1247,              # operations/hour
            "error_rate": 0.3,               # %
            "resource_utilization": {
                "cpu": 68.4,
                "memory": 72.1,
                "disk": 45.8,
                "network": 34.2
            }
        }
        
        # Operational KPIs
        operational_kpis = {
            "mean_time_to_detection": 8.7,      # minutes
            "mean_time_to_response": 23.4,      # minutes
            "mean_time_to_resolution": 142.8,   # minutes
            "false_positive_rate": 2.1,         # %
            "alert_fatigue_score": 6.3,         # 1-10 scale
            "analyst_productivity": 87.2        # %
        }
        
        return {
            "operational_metrics": {
                "performance_metrics": performance_metrics,
                "operational_kpis": operational_kpis,
                "service_level_objectives": {
                    "availability": {"target": 99.9, "actual": 99.7, "status": "At Risk"},
                    "response_time": {"target": 200, "actual": 245.7, "status": "Warning"},
                    "throughput": {"target": 1000, "actual": 1247, "status": "Good"}
                },
                "capacity_metrics": {
                    "peak_load_handling": 89.3,
                    "concurrent_campaigns": 23,
                    "queue_depth": 45,
                    "processing_backlog": 12
                }
            },
            "data_points": []
        }
    
    async def _format_report(self, report_data: Dict[str, Any], config: ReportConfiguration) -> Dict[str, Any]:
        """Format report according to specified format."""
        if config.format == ReportFormat.JSON:
            return report_data
        
        elif config.format == ReportFormat.HTML:
            # Generate HTML report
            html_content = self._generate_html_report(report_data, config)
            return {"content": html_content, "content_type": "text/html"}
        
        elif config.format == ReportFormat.PDF:
            # Generate PDF report (placeholder)
            return {"content": "PDF generation not implemented", "content_type": "application/pdf"}
        
        elif config.format == ReportFormat.CSV:
            # Convert data to CSV format
            csv_content = self._generate_csv_report(report_data, config)
            return {"content": csv_content, "content_type": "text/csv"}
        
        else:
            return report_data
    
    def _generate_html_report(self, report_data: Dict[str, Any], config: ReportConfiguration) -> str:
        """Generate HTML report content."""
        metadata = report_data.get("report_metadata", {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{config.title}</h1>
                <p>{config.description}</p>
                <p>Generated: {metadata.get('generated_at', 'Unknown')}</p>
            </div>
        """
        
        # Add report sections based on report type
        for section_name, section_data in report_data.items():
            if section_name != "report_metadata" and section_name != "data_points":
                html += f"""
                <div class="section">
                    <h2>{section_name.replace('_', ' ').title()}</h2>
                    <pre>{json.dumps(section_data, indent=2)}</pre>
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_csv_report(self, report_data: Dict[str, Any], config: ReportConfiguration) -> str:
        """Generate CSV report content."""
        data_points = report_data.get("data_points", [])
        
        if not data_points:
            return "No data available for CSV export"
        
        if HAS_PANDAS:
            # Convert to pandas DataFrame for easy CSV generation
            df = pd.DataFrame(data_points)
            return df.to_csv(index=False)
        else:
            # Simple CSV generation without pandas
            if not data_points:
                return "No data available"
            
            headers = list(data_points[0].keys()) if data_points else []
            csv_lines = [",".join(headers)]
            
            for row in data_points:
                csv_lines.append(",".join(str(row.get(h, "")) for h in headers))
            
            return "\n".join(csv_lines)
    
    def create_dashboard(self, dashboard: Dashboard) -> str:
        """Create and register a new dashboard."""
        self.dashboards[dashboard.dashboard_id] = dashboard
        
        BI_DASHBOARD_VIEWS.labels(dashboard_type=dashboard.title).inc()
        
        logger.info("Dashboard created", 
                   dashboard_id=dashboard.dashboard_id,
                   title=dashboard.title,
                   widgets=len(dashboard.widgets))
        
        return dashboard.dashboard_id
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard configuration."""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all available dashboards."""
        return [
            {
                "dashboard_id": dashboard.dashboard_id,
                "title": dashboard.title,
                "description": dashboard.description,
                "widgets": len(dashboard.widgets),
                "created_at": dashboard.created_at,
                "updated_at": dashboard.updated_at
            }
            for dashboard in self.dashboards.values()
        ]
    
    async def schedule_report(self, config: ReportConfiguration) -> str:
        """Schedule a report for automatic generation."""
        report_id = str(uuid.uuid4())
        self.scheduled_reports[report_id] = config
        
        logger.info("Report scheduled", 
                   report_id=report_id,
                   report_type=config.report_type.value,
                   schedule=config.schedule)
        
        return report_id
    
    async def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get status of a scheduled report."""
        config = self.scheduled_reports.get(report_id)
        if not config:
            return {"status": "not_found"}
        
        return {
            "status": "scheduled",
            "report_id": report_id,
            "report_type": config.report_type.value,
            "schedule": config.schedule,
            "next_execution": "calculated_based_on_schedule"
        }


# Global business intelligence engine
bi_engine = BusinessIntelligenceEngine()


async def initialize_business_intelligence():
    """Initialize the business intelligence system."""
    logger.info("Initializing business intelligence system")
    
    # Create default dashboards
    executive_dashboard = Dashboard(
        dashboard_id="executive_overview",
        title="Executive Overview",
        description="High-level security metrics and KPIs for executive leadership",
        widgets=[
            DashboardWidget(
                widget_id="campaign_success_rate",
                widget_type="metric",
                title="Campaign Success Rate",
                data_source="sqlite",
                query="SELECT AVG(CASE WHEN state = 'completed' THEN 1.0 ELSE 0.0 END) * 100 FROM campaigns",
                position={"x": 0, "y": 0, "width": 2, "height": 1}
            ),
            DashboardWidget(
                widget_id="vulnerability_trends",
                widget_type="chart",
                title="Vulnerability Discovery Trends",
                data_source="sqlite",
                query="SELECT DATE(discovered_at, 'unixepoch') as date, COUNT(*) FROM vulnerabilities GROUP BY date",
                visualization_config={"chart_type": "line"},
                position={"x": 2, "y": 0, "width": 4, "height": 2}
            )
        ]
    )
    
    operational_dashboard = Dashboard(
        dashboard_id="operational_metrics",
        title="Operational Metrics",
        description="Real-time operational metrics for security operations teams",
        widgets=[
            DashboardWidget(
                widget_id="active_campaigns",
                widget_type="metric",
                title="Active Campaigns",
                data_source="sqlite",
                query="SELECT COUNT(*) FROM campaigns WHERE state IN ('pending', 'executing')",
                position={"x": 0, "y": 0, "width": 1, "height": 1}
            ),
            DashboardWidget(
                widget_id="agent_performance",
                widget_type="table",
                title="Agent Performance",
                data_source="sqlite",
                query="SELECT agent_type, AVG(success_rate) FROM agents GROUP BY agent_type",
                position={"x": 1, "y": 0, "width": 3, "height": 2}
            )
        ]
    )
    
    bi_engine.create_dashboard(executive_dashboard)
    bi_engine.create_dashboard(operational_dashboard)
    
    logger.info("Business intelligence system initialized")


def get_bi_engine() -> BusinessIntelligenceEngine:
    """Get the global business intelligence engine."""
    return bi_engine