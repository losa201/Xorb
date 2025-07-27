"""
Business Intelligence and Comprehensive Reporting System
Provides executive dashboards, compliance reports, and operational analytics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportType(Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    OPERATIONAL_DASHBOARD = "operational_dashboard"
    COMPLIANCE_AUDIT = "compliance_audit"
    SECURITY_POSTURE = "security_posture"
    THREAT_INTELLIGENCE = "threat_intelligence"
    PERFORMANCE_METRICS = "performance_metrics"
    RISK_ASSESSMENT = "risk_assessment"
    CAMPAIGN_ANALYSIS = "campaign_analysis"

class ReportFormat(Enum):
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "xlsx"

@dataclass
class ReportMetric:
    name: str
    value: Union[int, float, str]
    target: Optional[Union[int, float]] = None
    status: str = "normal"  # normal, warning, critical
    trend: Optional[str] = None  # up, down, stable
    description: Optional[str] = None

@dataclass
class ReportSection:
    title: str
    metrics: List[ReportMetric]
    charts: List[Dict[str, Any]]
    narrative: str
    recommendations: List[str]

@dataclass
class ComprehensiveReport:
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    time_period: Dict[str, datetime]
    sections: List[ReportSection]
    executive_summary: str
    key_findings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

class BusinessIntelligenceEngine:
    """Advanced business intelligence and reporting engine"""
    
    def __init__(self):
        self.data_sources: Dict[str, Any] = {}
        self.report_templates: Dict[ReportType, Dict] = {}
        self.scheduled_reports: List[Dict] = []
        self.report_cache: Dict[str, ComprehensiveReport] = {}
        self.setup_templates()
    
    def setup_templates(self):
        """Setup default report templates"""
        self.report_templates = {
            ReportType.EXECUTIVE_SUMMARY: {
                "sections": [
                    "security_posture_overview",
                    "threat_landscape",
                    "operational_efficiency", 
                    "compliance_status",
                    "strategic_recommendations"
                ],
                "kpis": [
                    "security_score",
                    "incident_count",
                    "mean_time_to_detection",
                    "mean_time_to_response",
                    "compliance_percentage"
                ]
            },
            ReportType.OPERATIONAL_DASHBOARD: {
                "sections": [
                    "campaign_performance",
                    "agent_utilization",
                    "system_health",
                    "threat_detection_metrics"
                ],
                "refresh_interval": 300  # 5 minutes
            },
            ReportType.COMPLIANCE_AUDIT: {
                "sections": [
                    "regulatory_compliance",
                    "policy_adherence", 
                    "audit_findings",
                    "remediation_tracking"
                ],
                "standards": ["SOC2", "GDPR", "HIPAA", "PCI-DSS"]
            }
        }
    
    async def register_data_source(self, name: str, source: Any):
        """Register a data source for reporting"""
        self.data_sources[name] = source
        logger.info(f"Registered data source: {name}")
    
    async def generate_report(
        self, 
        report_type: ReportType,
        time_period: Optional[Dict[str, datetime]] = None,
        custom_filters: Optional[Dict] = None
    ) -> ComprehensiveReport:
        """Generate a comprehensive report"""
        
        if time_period is None:
            time_period = {
                "start": datetime.now() - timedelta(days=30),
                "end": datetime.now()
            }
        
        logger.info(f"Generating {report_type.value} report for period {time_period}")
        
        # Generate unique report ID
        report_id = f"{report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Collect data based on report type
        data = await self._collect_report_data(report_type, time_period, custom_filters)
        
        # Generate sections
        sections = await self._generate_report_sections(report_type, data)
        
        # Create executive summary
        executive_summary = await self._generate_executive_summary(report_type, sections)
        
        # Extract key findings and recommendations
        key_findings = self._extract_key_findings(sections)
        recommendations = self._extract_recommendations(sections)
        
        report = ComprehensiveReport(
            report_id=report_id,
            report_type=report_type,
            title=self._get_report_title(report_type),
            generated_at=datetime.now(),
            time_period=time_period,
            sections=sections,
            executive_summary=executive_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            metadata={
                "data_sources": list(self.data_sources.keys()),
                "filters": custom_filters or {},
                "generation_time": datetime.now().isoformat()
            }
        )
        
        # Cache the report
        self.report_cache[report_id] = report
        
        logger.info(f"Generated report {report_id} with {len(sections)} sections")
        return report
    
    async def _collect_report_data(
        self, 
        report_type: ReportType, 
        time_period: Dict[str, datetime],
        custom_filters: Optional[Dict]
    ) -> Dict[str, Any]:
        """Collect data from various sources for report generation"""
        
        data = {}
        
        # Mock data collection - in real implementation, this would query actual data sources
        if report_type == ReportType.EXECUTIVE_SUMMARY:
            data.update({
                "campaigns": await self._get_campaign_data(time_period),
                "threats": await self._get_threat_data(time_period),
                "compliance": await self._get_compliance_data(time_period),
                "performance": await self._get_performance_data(time_period)
            })
        
        elif report_type == ReportType.OPERATIONAL_DASHBOARD:
            data.update({
                "real_time_metrics": await self._get_real_time_metrics(),
                "agent_status": await self._get_agent_status(),
                "system_health": await self._get_system_health()
            })
        
        elif report_type == ReportType.COMPLIANCE_AUDIT:
            data.update({
                "audit_results": await self._get_audit_results(time_period),
                "policy_violations": await self._get_policy_violations(time_period),
                "remediation_status": await self._get_remediation_status(time_period)
            })
        
        return data
    
    async def _generate_report_sections(
        self, 
        report_type: ReportType, 
        data: Dict[str, Any]
    ) -> List[ReportSection]:
        """Generate report sections based on type and data"""
        
        sections = []
        
        if report_type == ReportType.EXECUTIVE_SUMMARY:
            sections.extend([
                await self._create_security_posture_section(data),
                await self._create_threat_landscape_section(data),
                await self._create_operational_efficiency_section(data),
                await self._create_compliance_status_section(data)
            ])
        
        elif report_type == ReportType.OPERATIONAL_DASHBOARD:
            sections.extend([
                await self._create_campaign_performance_section(data),
                await self._create_agent_utilization_section(data),
                await self._create_system_health_section(data)
            ])
        
        elif report_type == ReportType.COMPLIANCE_AUDIT:
            sections.extend([
                await self._create_regulatory_compliance_section(data),
                await self._create_audit_findings_section(data),
                await self._create_remediation_tracking_section(data)
            ])
        
        return sections
    
    async def _create_security_posture_section(self, data: Dict) -> ReportSection:
        """Create security posture overview section"""
        
        # Calculate security metrics
        total_threats = len(data.get("threats", []))
        critical_threats = len([t for t in data.get("threats", []) if t.get("severity") == "critical"])
        detection_rate = 0.92  # Mock calculation
        
        metrics = [
            ReportMetric("Overall Security Score", 85, 90, "warning", "up", 
                        "Composite security score based on threat detection and response"),
            ReportMetric("Critical Threats Detected", critical_threats, None, 
                        "critical" if critical_threats > 5 else "normal"),
            ReportMetric("Threat Detection Rate", f"{detection_rate:.1%}", 0.95, 
                        "warning" if detection_rate < 0.95 else "normal"),
            ReportMetric("Mean Time to Detection", "4.2 hours", 2.0, "warning", "down"),
            ReportMetric("Mean Time to Response", "12.5 hours", 8.0, "critical", "up")
        ]
        
        # Create charts
        charts = [
            self._create_security_score_chart(data),
            self._create_threat_trend_chart(data),
            self._create_detection_timeline_chart(data)
        ]
        
        narrative = f"""
        The organization's security posture shows mixed results this period. While threat detection 
        capabilities remain strong at {detection_rate:.1%}, response times have increased to 12.5 hours, 
        exceeding our target of 8 hours. We detected {total_threats} total threats, including 
        {critical_threats} critical threats requiring immediate attention.
        """
        
        recommendations = [
            "Implement automated response workflows to reduce MTTR",
            "Enhance threat hunting capabilities for proactive detection",
            "Review and optimize security tool configurations",
            "Increase security team training on rapid response procedures"
        ]
        
        return ReportSection(
            title="Security Posture Overview",
            metrics=metrics,
            charts=charts,
            narrative=narrative.strip(),
            recommendations=recommendations
        )
    
    async def _create_threat_landscape_section(self, data: Dict) -> ReportSection:
        """Create threat landscape analysis section"""
        
        threats = data.get("threats", [])
        threat_types = {}
        for threat in threats:
            t_type = threat.get("type", "unknown")
            threat_types[t_type] = threat_types.get(t_type, 0) + 1
        
        metrics = [
            ReportMetric("Total Threats", len(threats)),
            ReportMetric("Unique Attack Vectors", len(threat_types)),
            ReportMetric("Most Common Threat", max(threat_types.keys(), key=threat_types.get) if threat_types else "None"),
            ReportMetric("Threat Intelligence Score", 78, 85, "warning")
        ]
        
        charts = [
            self._create_threat_distribution_chart(threat_types),
            self._create_attack_vector_timeline_chart(data),
            self._create_threat_severity_matrix_chart(data)
        ]
        
        narrative = f"""
        The threat landscape analysis reveals {len(threats)} total threats across {len(threat_types)} 
        different attack vectors. The most prevalent threat type is {max(threat_types.keys(), key=threat_types.get) if threat_types else 'None'}, 
        accounting for {max(threat_types.values()) if threat_types else 0} incidents. 
        Our threat intelligence integration has provided early warning for 78% of detected threats.
        """
        
        recommendations = [
            "Enhance threat intelligence feeds for better coverage",
            "Implement predictive threat modeling",
            "Strengthen defenses against most common attack vectors",
            "Increase threat hunting focus on emerging attack patterns"
        ]
        
        return ReportSection(
            title="Threat Landscape Analysis",
            metrics=metrics,
            charts=charts,
            narrative=narrative.strip(),
            recommendations=recommendations
        )
    
    async def _create_operational_efficiency_section(self, data: Dict) -> ReportSection:
        """Create operational efficiency section"""
        
        campaigns = data.get("campaigns", [])
        successful_campaigns = len([c for c in campaigns if c.get("status") == "completed"])
        
        metrics = [
            ReportMetric("Total Campaigns", len(campaigns)),
            ReportMetric("Success Rate", f"{successful_campaigns/len(campaigns):.1%}" if campaigns else "0%", 0.90),
            ReportMetric("Average Campaign Duration", "2.3 days", 2.0, "warning"),
            ReportMetric("Agent Utilization", "76%", 80, "warning", "stable"),
            ReportMetric("System Uptime", "99.7%", 99.9, "warning")
        ]
        
        charts = [
            self._create_campaign_success_chart(data),
            self._create_resource_utilization_chart(data),
            self._create_performance_trends_chart(data)
        ]
        
        narrative = f"""
        Operational efficiency metrics show solid performance with {len(campaigns)} campaigns executed 
        and a {successful_campaigns/len(campaigns):.1%} success rate. Agent utilization at 76% suggests 
        room for optimization, while system uptime of 99.7% meets enterprise standards but falls 
        slightly below our 99.9% target.
        """
        
        recommendations = [
            "Optimize agent scheduling algorithms for better utilization",
            "Implement predictive maintenance for improved uptime",
            "Review campaign planning processes to reduce duration",
            "Consider scaling agent infrastructure during peak periods"
        ]
        
        return ReportSection(
            title="Operational Efficiency",
            metrics=metrics,
            charts=charts,
            narrative=narrative.strip(),
            recommendations=recommendations
        )
    
    async def _create_compliance_status_section(self, data: Dict) -> ReportSection:
        """Create compliance status section"""
        
        compliance_data = data.get("compliance", {})
        
        metrics = [
            ReportMetric("Overall Compliance Score", "94%", 95, "warning"),
            ReportMetric("SOC2 Compliance", "98%", 95, "normal"),
            ReportMetric("GDPR Compliance", "91%", 95, "warning"),
            ReportMetric("Open Compliance Issues", 7, 5, "warning"),
            ReportMetric("Remediation Progress", "85%", 90, "warning")
        ]
        
        charts = [
            self._create_compliance_dashboard_chart(compliance_data),
            self._create_compliance_trends_chart(data),
            self._create_remediation_progress_chart(data)
        ]
        
        narrative = """
        Compliance metrics show strong performance across most frameworks. SOC2 compliance exceeds 
        targets at 98%, while GDPR compliance at 91% requires attention. Seven open compliance 
        issues remain, with 85% remediation progress indicating steady improvement.
        """
        
        recommendations = [
            "Focus remediation efforts on GDPR compliance gaps",
            "Implement automated compliance monitoring",
            "Schedule quarterly compliance reviews",
            "Enhance data governance procedures"
        ]
        
        return ReportSection(
            title="Compliance Status",
            metrics=metrics,
            charts=charts,
            narrative=narrative.strip(),
            recommendations=recommendations
        )
    
    def _create_security_score_chart(self, data: Dict) -> Dict[str, Any]:
        """Create security score gauge chart"""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 85,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Security Score"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        return {
            "type": "gauge",
            "title": "Security Score",
            "data": fig.to_dict()
        }
    
    def _create_threat_trend_chart(self, data: Dict) -> Dict[str, Any]:
        """Create threat trend line chart"""
        
        # Mock data - in real implementation, query actual threat data
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
        threat_counts = np.random.poisson(5, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=threat_counts,
            mode='lines+markers',
            name='Daily Threats',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Threat Detection Trend",
            xaxis_title="Date",
            yaxis_title="Threats Detected",
            showlegend=True
        )
        
        return {
            "type": "line",
            "title": "Threat Detection Trend",
            "data": fig.to_dict()
        }
    
    def _create_threat_distribution_chart(self, threat_types: Dict) -> Dict[str, Any]:
        """Create threat distribution pie chart"""
        
        if not threat_types:
            threat_types = {"No threats": 1}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(threat_types.keys()),
            values=list(threat_types.values()),
            hole=.3
        )])
        
        fig.update_layout(title_text="Threat Type Distribution")
        
        return {
            "type": "pie",
            "title": "Threat Type Distribution", 
            "data": fig.to_dict()
        }
    
    async def export_report(
        self, 
        report: ComprehensiveReport, 
        format: ReportFormat,
        output_path: Optional[str] = None
    ) -> str:
        """Export report to specified format"""
        
        if output_path is None:
            output_path = f"reports/{report.report_id}.{format.value}"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == ReportFormat.JSON:
            with open(output_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        elif format == ReportFormat.HTML:
            html_content = await self._generate_html_report(report)
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        elif format == ReportFormat.CSV:
            # Export metrics as CSV
            metrics_data = []
            for section in report.sections:
                for metric in section.metrics:
                    metrics_data.append({
                        "section": section.title,
                        "metric": metric.name,
                        "value": metric.value,
                        "target": metric.target,
                        "status": metric.status
                    })
            
            df = pd.DataFrame(metrics_data)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Exported report {report.report_id} to {output_path}")
        return output_path
    
    async def _generate_html_report(self, report: ComprehensiveReport) -> str:
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 3px; }}
                .metric.warning {{ background: #fff3cd; }}
                .metric.critical {{ background: #f8d7da; }}
                .recommendations {{ background: #d4edda; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Period: {report.time_period['start'].strftime('%Y-%m-%d')} to {report.time_period['end'].strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{report.executive_summary}</p>
            </div>
        """
        
        for section in report.sections:
            html += f"""
            <div class="section">
                <h2>{section.title}</h2>
                
                <div class="metrics">
                    <h3>Key Metrics</h3>
            """
            
            for metric in section.metrics:
                status_class = metric.status if metric.status in ['warning', 'critical'] else ''
                html += f"""
                    <div class="metric {status_class}">
                        <strong>{metric.name}</strong><br>
                        {metric.value}
                        {f' (Target: {metric.target})' if metric.target else ''}
                    </div>
                """
            
            html += f"""
                </div>
                
                <div class="narrative">
                    <h3>Analysis</h3>
                    <p>{section.narrative}</p>
                </div>
                
                <div class="recommendations">
                    <h3>Recommendations</h3>
                    <ul>
            """
            
            for rec in section.recommendations:
                html += f"<li>{rec}</li>"
            
            html += """
                    </ul>
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    async def schedule_report(
        self,
        report_type: ReportType,
        schedule: str,  # cron format
        recipients: List[str],
        format: ReportFormat = ReportFormat.HTML
    ):
        """Schedule automatic report generation"""
        
        scheduled_report = {
            "id": f"scheduled_{report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "report_type": report_type,
            "schedule": schedule,
            "recipients": recipients,
            "format": format,
            "created_at": datetime.now(),
            "last_run": None,
            "next_run": None  # Calculate based on schedule
        }
        
        self.scheduled_reports.append(scheduled_report)
        logger.info(f"Scheduled {report_type.value} report with schedule: {schedule}")
    
    # Mock data methods for demonstration
    async def _get_campaign_data(self, time_period: Dict) -> List[Dict]:
        return [
            {"id": "camp1", "status": "completed", "duration_hours": 48},
            {"id": "camp2", "status": "completed", "duration_hours": 36},
            {"id": "camp3", "status": "failed", "duration_hours": 12}
        ]
    
    async def _get_threat_data(self, time_period: Dict) -> List[Dict]:
        return [
            {"id": "threat1", "type": "malware", "severity": "critical"},
            {"id": "threat2", "type": "phishing", "severity": "high"},
            {"id": "threat3", "type": "brute_force", "severity": "medium"}
        ]
    
    async def _get_compliance_data(self, time_period: Dict) -> Dict:
        return {"soc2": 0.98, "gdpr": 0.91, "hipaa": 0.95}
    
    async def _get_performance_data(self, time_period: Dict) -> Dict:
        return {"uptime": 0.997, "response_time": 120}
    
    async def _get_real_time_metrics(self) -> Dict:
        return {"active_campaigns": 5, "agents_online": 12}
    
    async def _get_agent_status(self) -> Dict:
        return {"total": 15, "active": 12, "idle": 3}
    
    async def _get_system_health(self) -> Dict:
        return {"cpu": 65, "memory": 78, "disk": 45}
    
    async def _get_audit_results(self, time_period: Dict) -> List[Dict]:
        return [{"finding": "password policy", "severity": "medium"}]
    
    async def _get_policy_violations(self, time_period: Dict) -> List[Dict]:
        return [{"policy": "data retention", "count": 3}]
    
    async def _get_remediation_status(self, time_period: Dict) -> Dict:
        return {"total": 10, "completed": 8, "in_progress": 2}
    
    # Additional chart creation methods would be implemented here
    async def _create_detection_timeline_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "timeline", "title": "Detection Timeline", "data": {}}
    
    async def _create_attack_vector_timeline_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "timeline", "title": "Attack Vector Timeline", "data": {}}
    
    async def _create_threat_severity_matrix_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "matrix", "title": "Threat Severity Matrix", "data": {}}
    
    async def _create_campaign_performance_section(self, data: Dict) -> ReportSection:
        return ReportSection("Campaign Performance", [], [], "Mock section", [])
    
    async def _create_agent_utilization_section(self, data: Dict) -> ReportSection:
        return ReportSection("Agent Utilization", [], [], "Mock section", [])
    
    async def _create_system_health_section(self, data: Dict) -> ReportSection:
        return ReportSection("System Health", [], [], "Mock section", [])
    
    async def _create_regulatory_compliance_section(self, data: Dict) -> ReportSection:
        return ReportSection("Regulatory Compliance", [], [], "Mock section", [])
    
    async def _create_audit_findings_section(self, data: Dict) -> ReportSection:
        return ReportSection("Audit Findings", [], [], "Mock section", [])
    
    async def _create_remediation_tracking_section(self, data: Dict) -> ReportSection:
        return ReportSection("Remediation Tracking", [], [], "Mock section", [])
    
    async def _create_campaign_success_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "bar", "title": "Campaign Success", "data": {}}
    
    async def _create_resource_utilization_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "gauge", "title": "Resource Utilization", "data": {}}
    
    async def _create_performance_trends_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "line", "title": "Performance Trends", "data": {}}
    
    async def _create_compliance_dashboard_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "dashboard", "title": "Compliance Dashboard", "data": {}}
    
    async def _create_compliance_trends_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "line", "title": "Compliance Trends", "data": {}}
    
    async def _create_remediation_progress_chart(self, data: Dict) -> Dict[str, Any]:
        return {"type": "progress", "title": "Remediation Progress", "data": {}}
    
    async def _generate_executive_summary(self, report_type: ReportType, sections: List[ReportSection]) -> str:
        """Generate executive summary based on sections"""
        return f"Executive summary for {report_type.value} report generated at {datetime.now()}."
    
    def _extract_key_findings(self, sections: List[ReportSection]) -> List[str]:
        """Extract key findings from all sections"""
        findings = []
        for section in sections:
            # Extract critical metrics and important insights
            critical_metrics = [m for m in section.metrics if m.status == "critical"]
            if critical_metrics:
                findings.append(f"{section.title}: {len(critical_metrics)} critical issues identified")
        return findings
    
    def _extract_recommendations(self, sections: List[ReportSection]) -> List[str]:
        """Extract top recommendations from all sections"""
        all_recommendations = []
        for section in sections:
            all_recommendations.extend(section.recommendations[:2])  # Top 2 per section
        return all_recommendations[:10]  # Top 10 overall
    
    def _get_report_title(self, report_type: ReportType) -> str:
        """Get appropriate title for report type"""
        titles = {
            ReportType.EXECUTIVE_SUMMARY: "Executive Security Summary",
            ReportType.OPERATIONAL_DASHBOARD: "Operational Dashboard Report",
            ReportType.COMPLIANCE_AUDIT: "Compliance Audit Report",
            ReportType.SECURITY_POSTURE: "Security Posture Assessment",
            ReportType.THREAT_INTELLIGENCE: "Threat Intelligence Report",
            ReportType.PERFORMANCE_METRICS: "Performance Metrics Report",
            ReportType.RISK_ASSESSMENT: "Risk Assessment Report",
            ReportType.CAMPAIGN_ANALYSIS: "Campaign Analysis Report"
        }
        return titles.get(report_type, "Security Report")


class ReportingDashboard:
    """Interactive dashboard for report management"""
    
    def __init__(self, bi_engine: BusinessIntelligenceEngine):
        self.bi_engine = bi_engine
        self.dashboard_config = {}
    
    async def create_dashboard(self, dashboard_type: str = "executive"):
        """Create interactive dashboard"""
        # This would integrate with web framework for live dashboard
        pass
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": await self._get_live_metrics(),
            "alerts": await self._get_active_alerts(),
            "status": await self._get_system_status()
        }
    
    async def _get_live_metrics(self) -> Dict:
        return {"campaigns_active": 3, "threats_today": 7, "system_health": 95}
    
    async def _get_active_alerts(self) -> List[Dict]:
        return [{"type": "warning", "message": "High CPU usage detected"}]
    
    async def _get_system_status(self) -> Dict:
        return {"overall": "healthy", "components": {"api": "up", "db": "up"}}

# Example usage and testing
async def demo_business_intelligence():
    """Demonstrate business intelligence capabilities"""
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate executive summary report
    report = await bi_engine.generate_report(ReportType.EXECUTIVE_SUMMARY)
    
    print(f"Generated report: {report.title}")
    print(f"Report ID: {report.report_id}")
    print(f"Sections: {len(report.sections)}")
    
    # Export to different formats
    await bi_engine.export_report(report, ReportFormat.JSON)
    await bi_engine.export_report(report, ReportFormat.HTML)
    
    # Schedule recurring reports
    await bi_engine.schedule_report(
        ReportType.OPERATIONAL_DASHBOARD,
        "0 8 * * MON",  # Every Monday at 8 AM
        ["ciso@company.com", "security-team@company.com"]
    )
    
    print("✅ Business intelligence demo completed")

if __name__ == "__main__":
    asyncio.run(demo_business_intelligence())