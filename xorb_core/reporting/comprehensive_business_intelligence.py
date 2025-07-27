#!/usr/bin/env python3
"""
XORB Comprehensive Business Intelligence and Reporting Engine
Enterprise-grade analytics, dashboards, and strategic intelligence
"""

import asyncio
import json
import time
import uuid
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('XORB-BI')

class ReportType(Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    OPERATIONAL_METRICS = "operational_metrics"
    THREAT_INTELLIGENCE = "threat_intelligence"
    CAMPAIGN_ANALYSIS = "campaign_analysis"
    VULNERABILITY_TRENDS = "vulnerability_trends"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    COMPLIANCE_AUDIT = "compliance_audit"
    COST_ANALYSIS = "cost_analysis"
    RISK_ASSESSMENT = "risk_assessment"

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class BusinessMetric:
    """Core business intelligence metric."""
    metric_id: str
    metric_name: str
    metric_type: str  # counter, gauge, histogram, summary
    value: float
    unit: str
    timestamp: float
    category: str
    tags: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    trend: Optional[str] = None  # increasing, decreasing, stable

@dataclass
class ReportConfiguration:
    """Configuration for report generation."""
    report_id: str
    report_type: ReportType
    frequency: str  # hourly, daily, weekly, monthly
    recipients: List[str]
    format: str  # pdf, html, json, csv
    filters: Dict[str, Any] = field(default_factory=dict)
    custom_queries: List[str] = field(default_factory=list)
    enabled: bool = True

@dataclass
class AlertRule:
    """Business intelligence alert rule."""
    rule_id: str
    rule_name: str
    metric_pattern: str
    condition: str  # greater_than, less_than, equals, not_equals
    threshold: float
    severity: AlertSeverity
    action: str  # email, webhook, dashboard
    enabled: bool = True
    cooldown_minutes: int = 60

class DataWarehouse:
    """High-performance data warehouse for XORB analytics."""
    
    def __init__(self, db_path: str = "/tmp/xorb_warehouse.db"):
        self.db_path = db_path
        self.connection_pool = {}
        self._initialize_warehouse()
    
    def _initialize_warehouse(self):
        """Initialize data warehouse schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Business metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                timestamp REAL NOT NULL,
                category TEXT NOT NULL,
                tags TEXT,
                confidence REAL DEFAULT 1.0,
                trend TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Campaign performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS campaign_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                campaign_name TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT NOT NULL,
                targets_scanned INTEGER DEFAULT 0,
                vulnerabilities_found INTEGER DEFAULT 0,
                exploits_successful INTEGER DEFAULT 0,
                stealth_rating REAL DEFAULT 0.0,
                cost_usd REAL DEFAULT 0.0,
                roi_percentage REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Threat intelligence table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                threat_id TEXT NOT NULL,
                threat_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                description TEXT,
                indicators TEXT,
                mitigations TEXT,
                timestamp REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Compliance events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                compliance_framework TEXT NOT NULL,
                control_id TEXT NOT NULL,
                status TEXT NOT NULL,
                evidence TEXT,
                findings TEXT,
                remediation TEXT,
                timestamp REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_metric(self, metric: BusinessMetric):
        """Store business metric in warehouse."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO business_metrics 
            (metric_id, metric_name, metric_type, value, unit, timestamp, 
             category, tags, confidence, trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.metric_id, metric.metric_name, metric.metric_type,
            metric.value, metric.unit, metric.timestamp, metric.category,
            json.dumps(metric.tags), metric.confidence, metric.trend
        ))
        
        conn.commit()
        conn.close()
    
    def query_metrics(self, filters: Dict[str, Any], limit: int = 1000) -> List[Dict[str, Any]]:
        """Query business metrics with filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM business_metrics WHERE 1=1"
        params = []
        
        if filters.get('category'):
            query += " AND category = ?"
            params.append(filters['category'])
        
        if filters.get('metric_type'):
            query += " AND metric_type = ?"
            params.append(filters['metric_type'])
        
        if filters.get('start_time'):
            query += " AND timestamp >= ?"
            params.append(filters['start_time'])
        
        if filters.get('end_time'):
            query += " AND timestamp <= ?"
            params.append(filters['end_time'])
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class BusinessIntelligenceEngine:
    """Core business intelligence processing engine."""
    
    def __init__(self):
        self.engine_id = f"BI-{str(uuid.uuid4())[:8].upper()}"
        self.warehouse = DataWarehouse()
        self.report_configs = {}
        self.alert_rules = {}
        self.active_alerts = {}
        self.is_running = False
        
        logger.info(f"🧠 Business Intelligence Engine initialized: {self.engine_id}")
    
    def register_report_config(self, config: ReportConfiguration):
        """Register report configuration."""
        self.report_configs[config.report_id] = config
        logger.info(f"📊 Report config registered: {config.report_type.value}")
    
    def register_alert_rule(self, rule: AlertRule):
        """Register alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"🚨 Alert rule registered: {rule.rule_name}")
    
    def calculate_kpis(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Calculate key performance indicators."""
        end_time = time.time()
        start_time = end_time - (time_range_hours * 3600)
        
        filters = {'start_time': start_time, 'end_time': end_time}
        metrics = self.warehouse.query_metrics(filters)
        
        if not metrics:
            return {
                "total_operations": 0,
                "avg_performance": 0.0,
                "threat_detection_rate": 0.0,
                "system_uptime_percentage": 0.0,
                "cost_per_operation": 0.0,
                "roi_percentage": 0.0
            }
        
        # Calculate KPIs
        operation_metrics = [m for m in metrics if m['category'] == 'operations']
        performance_metrics = [m for m in metrics if m['category'] == 'performance']
        threat_metrics = [m for m in metrics if m['category'] == 'threats']
        
        total_operations = len(operation_metrics)
        avg_performance = np.mean([m['value'] for m in performance_metrics]) if performance_metrics else 0.0
        threat_detection_rate = np.mean([m['value'] for m in threat_metrics]) if threat_metrics else 0.0
        
        # Simulate additional KPIs
        system_uptime = 99.7 + np.random.uniform(-0.5, 0.3)
        cost_per_op = 0.12 + np.random.uniform(-0.05, 0.03)
        roi_percentage = 285.5 + np.random.uniform(-25, 50)
        
        return {
            "total_operations": total_operations,
            "avg_performance": avg_performance,
            "threat_detection_rate": threat_detection_rate,
            "system_uptime_percentage": system_uptime,
            "cost_per_operation": cost_per_op,
            "roi_percentage": roi_percentage,
            "time_range_hours": time_range_hours,
            "calculation_timestamp": time.time()
        }
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary report."""
        kpis = self.calculate_kpis(24)  # Last 24 hours
        weekly_kpis = self.calculate_kpis(168)  # Last 7 days
        
        # Executive insights
        insights = []
        
        if kpis["threat_detection_rate"] > 0.90:
            insights.append("🎯 Exceptional threat detection performance (>90%)")
        elif kpis["threat_detection_rate"] > 0.75:
            insights.append("✅ Good threat detection performance (>75%)")
        else:
            insights.append("⚠️ Threat detection needs improvement (<75%)")
        
        if kpis["system_uptime_percentage"] > 99.5:
            insights.append("🏆 Outstanding system reliability (>99.5% uptime)")
        elif kpis["system_uptime_percentage"] > 99.0:
            insights.append("✅ Good system reliability (>99% uptime)")
        else:
            insights.append("⚠️ System reliability concerns (<99% uptime)")
        
        if kpis["roi_percentage"] > 250:
            insights.append("💰 Excellent ROI performance (>250%)")
        elif kpis["roi_percentage"] > 150:
            insights.append("📈 Good ROI performance (>150%)")
        else:
            insights.append("📉 ROI improvement opportunities (<150%)")
        
        # Strategic recommendations
        recommendations = [
            "Continue autonomous operation optimization",
            "Expand threat intelligence integration",
            "Enhance agent learning capabilities",
            "Optimize resource utilization patterns"
        ]
        
        if kpis["cost_per_operation"] > 0.15:
            recommendations.append("Review cost optimization opportunities")
        
        return {
            "report_id": f"EXEC-{str(uuid.uuid4())[:8].upper()}",
            "report_type": "executive_summary",
            "generation_time": time.time(),
            "period": "24 hours",
            "kpis": kpis,
            "weekly_comparison": {
                "operations_growth": kpis["total_operations"] - weekly_kpis["total_operations"],
                "performance_trend": "increasing" if kpis["avg_performance"] > weekly_kpis["avg_performance"] else "decreasing",
                "cost_efficiency": kpis["cost_per_operation"] < weekly_kpis["cost_per_operation"]
            },
            "executive_insights": insights,
            "strategic_recommendations": recommendations,
            "risk_assessment": {
                "overall_risk": "low",
                "operational_risk": "minimal",
                "security_risk": "low",
                "compliance_risk": "minimal"
            }
        }
    
    def generate_operational_metrics_report(self) -> Dict[str, Any]:
        """Generate detailed operational metrics report."""
        kpis = self.calculate_kpis(24)
        
        # Agent performance metrics
        agent_metrics = {
            "total_agents": 68,
            "active_agents": 64,
            "agent_utilization": 94.1,
            "avg_agent_performance": kpis["avg_performance"],
            "learning_cycles_completed": np.random.randint(150, 300),
            "evolution_events": np.random.randint(25, 45)
        }
        
        # System resource metrics
        resource_metrics = {
            "cpu_utilization_avg": 75.3 + np.random.uniform(-5, 5),
            "memory_utilization_avg": 42.8 + np.random.uniform(-3, 3),
            "network_throughput_mbps": 245.7 + np.random.uniform(-20, 20),
            "disk_io_ops_per_sec": 1847 + np.random.randint(-200, 200),
            "database_query_performance": 98.2 + np.random.uniform(-2, 1)
        }
        
        # Security operations metrics
        security_metrics = {
            "vulnerabilities_detected": np.random.randint(15, 35),
            "threats_neutralized": np.random.randint(8, 18),
            "false_positive_rate": 0.03 + np.random.uniform(-0.01, 0.02),
            "mean_time_to_detection": 2.4 + np.random.uniform(-0.5, 0.5),
            "mean_time_to_response": 8.7 + np.random.uniform(-2, 2)
        }
        
        return {
            "report_id": f"OPS-{str(uuid.uuid4())[:8].upper()}",
            "report_type": "operational_metrics",
            "generation_time": time.time(),
            "period": "24 hours",
            "agent_metrics": agent_metrics,
            "resource_metrics": resource_metrics,
            "security_metrics": security_metrics,
            "performance_trends": {
                "agent_efficiency": "increasing",
                "resource_optimization": "stable",
                "security_effectiveness": "increasing"
            }
        }
    
    def generate_threat_intelligence_report(self) -> Dict[str, Any]:
        """Generate threat intelligence analysis report."""
        # Simulate threat intelligence data
        threat_categories = ["APT", "Malware", "Phishing", "Ransomware", "Zero-day"]
        threat_sources = ["Internal Detection", "External Feeds", "Partner Intelligence", "OSINT"]
        
        threat_summary = {}
        for category in threat_categories:
            count = np.random.randint(2, 12)
            threat_summary[category] = {
                "total_threats": count,
                "high_severity": np.random.randint(0, max(1, count//3)),
                "mitigated": np.random.randint(count//2, count),
                "active_investigations": np.random.randint(0, max(1, count//4))
            }
        
        # Top threats
        top_threats = [
            {
                "threat_id": f"THR-{str(uuid.uuid4())[:8].upper()}",
                "name": "Advanced Persistent Threat Campaign",
                "severity": "high",
                "confidence": 0.89,
                "affected_systems": 12,
                "mitigation_status": "in_progress"
            },
            {
                "threat_id": f"THR-{str(uuid.uuid4())[:8].upper()}",
                "name": "Novel Ransomware Variant",
                "severity": "critical",
                "confidence": 0.95,
                "affected_systems": 3,
                "mitigation_status": "contained"
            },
            {
                "threat_id": f"THR-{str(uuid.uuid4())[:8].upper()}",
                "name": "Zero-day Exploit Framework",
                "severity": "high",
                "confidence": 0.73,
                "affected_systems": 0,
                "mitigation_status": "monitoring"
            }
        ]
        
        return {
            "report_id": f"THREAT-{str(uuid.uuid4())[:8].upper()}",
            "report_type": "threat_intelligence",
            "generation_time": time.time(),
            "period": "24 hours",
            "threat_summary": threat_summary,
            "top_threats": top_threats,
            "intelligence_sources": {
                source: np.random.randint(5, 25) for source in threat_sources
            },
            "threat_trends": {
                "emerging_threats": np.random.randint(3, 8),
                "threat_evolution_rate": "moderate",
                "geographic_distribution": {
                    "North America": 0.35,
                    "Europe": 0.28,
                    "Asia Pacific": 0.22,
                    "Other": 0.15
                }
            }
        }
    
    def check_alert_rules(self, metrics: List[BusinessMetric]):
        """Check metrics against alert rules."""
        current_time = time.time()
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_alert = self.active_alerts.get(rule.rule_id)
            if last_alert and (current_time - last_alert) < (rule.cooldown_minutes * 60):
                continue
            
            # Find matching metrics
            matching_metrics = [
                m for m in metrics 
                if rule.metric_pattern in m.metric_name.lower()
            ]
            
            for metric in matching_metrics:
                alert_triggered = False
                
                if rule.condition == "greater_than" and metric.value > rule.threshold:
                    alert_triggered = True
                elif rule.condition == "less_than" and metric.value < rule.threshold:
                    alert_triggered = True
                elif rule.condition == "equals" and metric.value == rule.threshold:
                    alert_triggered = True
                
                if alert_triggered:
                    self._trigger_alert(rule, metric)
                    self.active_alerts[rule.rule_id] = current_time
    
    def _trigger_alert(self, rule: AlertRule, metric: BusinessMetric):
        """Trigger business intelligence alert."""
        alert_data = {
            "alert_id": f"ALERT-{str(uuid.uuid4())[:8].upper()}",
            "rule_id": rule.rule_id,
            "rule_name": rule.rule_name,
            "severity": rule.severity.value,
            "metric_name": metric.metric_name,
            "metric_value": metric.value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "timestamp": time.time(),
            "action": rule.action
        }
        
        logger.warning(f"🚨 ALERT: {rule.rule_name} - {metric.metric_name}: {metric.value} {rule.condition} {rule.threshold}")
        
        # Execute alert action
        if rule.action == "email":
            self._send_email_alert(alert_data)
        elif rule.action == "webhook":
            self._send_webhook_alert(alert_data)
        elif rule.action == "dashboard":
            self._update_dashboard_alert(alert_data)
    
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert (simulated)."""
        logger.info(f"📧 Email alert sent: {alert_data['alert_id']}")
    
    def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Send webhook alert (simulated)."""
        logger.info(f"🔗 Webhook alert sent: {alert_data['alert_id']}")
    
    def _update_dashboard_alert(self, alert_data: Dict[str, Any]):
        """Update dashboard with alert (simulated)."""
        logger.info(f"📊 Dashboard alert updated: {alert_data['alert_id']}")

class ComprehensiveReportingOrchestrator:
    """Orchestrates comprehensive business intelligence and reporting."""
    
    def __init__(self):
        self.orchestrator_id = f"REPORT-{str(uuid.uuid4())[:8].upper()}"
        self.bi_engine = BusinessIntelligenceEngine()
        self.is_running = False
        self.report_schedule = {}
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        logger.info(f"📊 Comprehensive Reporting Orchestrator initialized: {self.orchestrator_id}")
    
    def _initialize_default_configs(self):
        """Initialize default report configurations and alert rules."""
        # Executive summary report (daily)
        exec_config = ReportConfiguration(
            report_id="exec-daily",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            frequency="daily",
            recipients=["ceo@company.com", "ciso@company.com"],
            format="pdf"
        )
        self.bi_engine.register_report_config(exec_config)
        
        # Operational metrics (hourly)
        ops_config = ReportConfiguration(
            report_id="ops-hourly",
            report_type=ReportType.OPERATIONAL_METRICS,
            frequency="hourly",
            recipients=["soc@company.com"],
            format="json"
        )
        self.bi_engine.register_report_config(ops_config)
        
        # Threat intelligence (every 4 hours)
        threat_config = ReportConfiguration(
            report_id="threat-4h",
            report_type=ReportType.THREAT_INTELLIGENCE,
            frequency="4hourly",
            recipients=["threat-team@company.com"],
            format="html"
        )
        self.bi_engine.register_report_config(threat_config)
        
        # Critical performance alert
        perf_alert = AlertRule(
            rule_id="critical-performance",
            rule_name="Critical Performance Degradation",
            metric_pattern="performance",
            condition="less_than",
            threshold=0.70,
            severity=AlertSeverity.CRITICAL,
            action="email"
        )
        self.bi_engine.register_alert_rule(perf_alert)
        
        # High CPU utilization alert
        cpu_alert = AlertRule(
            rule_id="high-cpu",
            rule_name="High CPU Utilization",
            metric_pattern="cpu",
            condition="greater_than",
            threshold=90.0,
            severity=AlertSeverity.HIGH,
            action="dashboard"
        )
        self.bi_engine.register_alert_rule(cpu_alert)
    
    async def generate_sample_metrics(self):
        """Generate sample business metrics for demonstration."""
        while self.is_running:
            current_time = time.time()
            
            # Performance metrics
            perf_metric = BusinessMetric(
                metric_id=f"perf-{str(uuid.uuid4())[:8]}",
                metric_name="agent_performance_score",
                metric_type="gauge",
                value=0.85 + np.random.uniform(-0.15, 0.10),
                unit="percentage",
                timestamp=current_time,
                category="performance",
                tags={"source": "agent_monitor"}
            )
            self.bi_engine.warehouse.store_metric(perf_metric)
            
            # CPU utilization metrics
            cpu_metric = BusinessMetric(
                metric_id=f"cpu-{str(uuid.uuid4())[:8]}",
                metric_name="cpu_utilization",
                metric_type="gauge",
                value=75.0 + np.random.uniform(-20, 20),
                unit="percentage",
                timestamp=current_time,
                category="system",
                tags={"source": "system_monitor"}
            )
            self.bi_engine.warehouse.store_metric(cpu_metric)
            
            # Threat detection metrics
            threat_metric = BusinessMetric(
                metric_id=f"threat-{str(uuid.uuid4())[:8]}",
                metric_name="threat_detection_rate",
                metric_type="gauge",
                value=0.88 + np.random.uniform(-0.10, 0.08),
                unit="percentage",
                timestamp=current_time,
                category="threats",
                tags={"source": "threat_hunter"}
            )
            self.bi_engine.warehouse.store_metric(threat_metric)
            
            # Operations metrics
            ops_metric = BusinessMetric(
                metric_id=f"ops-{str(uuid.uuid4())[:8]}",
                metric_name="operations_completed",
                metric_type="counter",
                value=np.random.randint(5, 15),
                unit="count",
                timestamp=current_time,
                category="operations",
                tags={"source": "orchestrator"}
            )
            self.bi_engine.warehouse.store_metric(ops_metric)
            
            # Check alert rules
            metrics = [perf_metric, cpu_metric, threat_metric, ops_metric]
            self.bi_engine.check_alert_rules(metrics)
            
            await asyncio.sleep(30)  # Generate metrics every 30 seconds
    
    async def automated_report_generation(self):
        """Automated report generation based on schedules."""
        while self.is_running:
            current_time = time.time()
            
            # Generate reports every 5 minutes for demo
            logger.info("📊 Generating scheduled business intelligence reports...")
            
            # Executive summary
            exec_report = self.bi_engine.generate_executive_summary()
            logger.info(f"📋 Executive Summary: {exec_report['report_id']}")
            logger.info(f"   Total Operations: {exec_report['kpis']['total_operations']}")
            logger.info(f"   Avg Performance: {exec_report['kpis']['avg_performance']:.1%}")
            logger.info(f"   ROI: {exec_report['kpis']['roi_percentage']:.1f}%")
            
            # Operational metrics
            ops_report = self.bi_engine.generate_operational_metrics_report()
            logger.info(f"🔧 Operational Metrics: {ops_report['report_id']}")
            logger.info(f"   Active Agents: {ops_report['agent_metrics']['active_agents']}")
            logger.info(f"   CPU Utilization: {ops_report['resource_metrics']['cpu_utilization_avg']:.1f}%")
            
            # Threat intelligence
            threat_report = self.bi_engine.generate_threat_intelligence_report()
            logger.info(f"🛡️ Threat Intelligence: {threat_report['report_id']}")
            logger.info(f"   Top Threats: {len(threat_report['top_threats'])}")
            
            await asyncio.sleep(300)  # Generate reports every 5 minutes
    
    async def start_comprehensive_reporting(self):
        """Start comprehensive reporting and business intelligence."""
        logger.info("🚀 Starting Comprehensive Business Intelligence and Reporting")
        
        self.is_running = True
        
        try:
            await asyncio.gather(
                self.generate_sample_metrics(),
                self.automated_report_generation(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"❌ Reporting orchestrator error: {e}")
        finally:
            logger.info("🏁 Comprehensive Reporting stopped")

async def main():
    """Main execution for comprehensive reporting."""
    reporting_orchestrator = ComprehensiveReportingOrchestrator()
    
    print(f"\n📊 XORB COMPREHENSIVE BUSINESS INTELLIGENCE ACTIVATED")
    print(f"🆔 Orchestrator ID: {reporting_orchestrator.orchestrator_id}")
    print(f"📈 Features: Executive Reports, Operational Metrics, Threat Intelligence")
    print(f"🚨 Alerts: Performance, Security, Resource Utilization")
    print(f"📋 Automated Scheduling: Hourly, Daily, Weekly Reports")
    print(f"\n🔥 COMPREHENSIVE REPORTING STARTING...\n")
    
    try:
        await reporting_orchestrator.start_comprehensive_reporting()
    except KeyboardInterrupt:
        logger.info("🛑 Comprehensive reporting interrupted by user")
    except Exception as e:
        logger.error(f"Comprehensive reporting failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())