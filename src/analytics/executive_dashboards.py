"""
Executive Analytics Dashboards for XORB
Provides C-level security and business intelligence
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Types of executive dashboards"""
    SECURITY_OVERVIEW = "security_overview"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE_STATUS = "compliance_status"
    BUSINESS_IMPACT = "business_impact"
    OPERATIONAL_METRICS = "operational_metrics"
    THREAT_LANDSCAPE = "threat_landscape"


class MetricTrend(Enum):
    """Metric trend directions"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    CRITICAL = "critical"


@dataclass
class ExecutiveMetric:
    """Executive-level metric"""
    metric_id: str
    name: str
    description: str
    current_value: float
    target_value: float
    previous_value: float
    unit: str
    trend: MetricTrend
    impact_level: str
    last_updated: datetime
    drill_down_url: Optional[str] = None


@dataclass
class SecurityIncident:
    """High-level security incident for executive view"""
    incident_id: str
    title: str
    severity: str
    status: str
    business_impact: str
    estimated_loss: float
    response_time: int  # minutes
    assigned_team: str
    created_at: datetime
    resolved_at: Optional[datetime] = None


@dataclass
class RiskAssessment:
    """Enterprise risk assessment"""
    risk_id: str
    category: str
    description: str
    probability: float  # 0-100
    impact_score: float  # 0-100
    risk_score: float  # calculated
    mitigation_status: str
    owner: str
    due_date: datetime
    last_reviewed: datetime


class ExecutiveDashboardEngine:
    """Generates executive-level security and business intelligence"""

    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache
        self.dashboard_cache = {}

    async def generate_security_overview(self, tenant_id: str, period_days: int = 30) -> Dict[str, Any]:
        """Generate security overview dashboard for executives"""

        cache_key = f"security_overview_{tenant_id}_{period_days}"
        if self._is_cached(cache_key):
            return self.dashboard_cache[cache_key]

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        # Core security metrics
        security_metrics = await self._get_security_metrics(tenant_id, start_date, end_date)

        # Recent incidents
        incidents = await self._get_recent_incidents(tenant_id, limit=10)

        # Threat intelligence summary
        threat_summary = await self._get_threat_landscape_summary(tenant_id)

        # Security posture score
        posture_score = await self._calculate_security_posture(tenant_id)

        dashboard = {
            "dashboard_type": "security_overview",
            "tenant_id": tenant_id,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "generated_at": datetime.utcnow().isoformat(),

            "executive_summary": {
                "security_posture_score": posture_score,
                "total_incidents": len(incidents),
                "critical_incidents": len([i for i in incidents if i.severity == "critical"]),
                "mean_response_time": sum(i.response_time for i in incidents) / len(incidents) if incidents else 0,
                "estimated_risk_exposure": sum(i.estimated_loss for i in incidents),
                "compliance_score": await self._get_compliance_score(tenant_id)
            },

            "key_metrics": [asdict(metric) for metric in security_metrics],

            "recent_incidents": [
                {
                    "incident_id": inc.incident_id,
                    "title": inc.title,
                    "severity": inc.severity,
                    "status": inc.status,
                    "business_impact": inc.business_impact,
                    "estimated_loss": inc.estimated_loss,
                    "days_open": (datetime.utcnow() - inc.created_at).days,
                    "assigned_team": inc.assigned_team
                } for inc in incidents[:5]
            ],

            "threat_landscape": threat_summary,

            "recommendations": await self._generate_executive_recommendations(tenant_id, security_metrics, incidents),

            "charts": {
                "incident_trend": await self._get_incident_trend_data(tenant_id, start_date, end_date),
                "threat_categories": await self._get_threat_category_distribution(tenant_id),
                "response_time_trend": await self._get_response_time_trend(tenant_id, start_date, end_date),
                "security_score_history": await self._get_security_score_history(tenant_id, start_date, end_date)
            }
        }

        self._cache_dashboard(cache_key, dashboard)
        return dashboard

    async def generate_risk_management_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Generate risk management dashboard"""

        cache_key = f"risk_management_{tenant_id}"
        if self._is_cached(cache_key):
            return self.dashboard_cache[cache_key]

        # Get risk assessments
        risks = await self._get_enterprise_risks(tenant_id)

        # Calculate risk metrics
        total_risks = len(risks)
        high_risks = len([r for r in risks if r.risk_score >= 70])
        overdue_risks = len([r for r in risks if r.due_date < datetime.utcnow()])

        # Risk by category
        risk_categories = {}
        for risk in risks:
            category = risk.category
            if category not in risk_categories:
                risk_categories[category] = {"count": 0, "avg_score": 0, "total_score": 0}
            risk_categories[category]["count"] += 1
            risk_categories[category]["total_score"] += risk.risk_score

        for category in risk_categories:
            risk_categories[category]["avg_score"] = (
                risk_categories[category]["total_score"] / risk_categories[category]["count"]
            )

        dashboard = {
            "dashboard_type": "risk_management",
            "tenant_id": tenant_id,
            "generated_at": datetime.utcnow().isoformat(),

            "risk_summary": {
                "total_risks": total_risks,
                "high_risk_count": high_risks,
                "medium_risk_count": len([r for r in risks if 40 <= r.risk_score < 70]),
                "low_risk_count": len([r for r in risks if r.risk_score < 40]),
                "overdue_mitigations": overdue_risks,
                "average_risk_score": sum(r.risk_score for r in risks) / total_risks if risks else 0
            },

            "top_risks": sorted(risks, key=lambda x: x.risk_score, reverse=True)[:10],

            "risk_by_category": risk_categories,

            "mitigation_status": {
                "completed": len([r for r in risks if r.mitigation_status == "completed"]),
                "in_progress": len([r for r in risks if r.mitigation_status == "in_progress"]),
                "planned": len([r for r in risks if r.mitigation_status == "planned"]),
                "not_started": len([r for r in risks if r.mitigation_status == "not_started"])
            },

            "charts": {
                "risk_heat_map": await self._generate_risk_heat_map(risks),
                "mitigation_timeline": await self._get_mitigation_timeline(risks),
                "risk_trend": await self._get_risk_trend_data(tenant_id)
            }
        }

        self._cache_dashboard(cache_key, dashboard)
        return dashboard

    async def generate_compliance_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Generate compliance status dashboard"""

        # Import compliance dashboard if available
        try:
            from compliance.soc2.compliance_dashboard import compliance_dashboard
            compliance_data = await compliance_dashboard.get_compliance_dashboard_data()
        except ImportError:
            compliance_data = await self._mock_compliance_data()

        dashboard = {
            "dashboard_type": "compliance_status",
            "tenant_id": tenant_id,
            "generated_at": datetime.utcnow().isoformat(),

            "compliance_overview": {
                "overall_score": compliance_data.get("overview", {}).get("overall_compliance_score", 85),
                "controls_tested": compliance_data.get("overview", {}).get("controls_tested", 15),
                "controls_passed": compliance_data.get("overview", {}).get("controls_passed", 13),
                "audit_readiness": "High" if compliance_data.get("overview", {}).get("overall_compliance_score", 0) > 90 else "Medium"
            },

            "frameworks": {
                "soc2": {
                    "status": "In Progress",
                    "completion": 87,
                    "next_audit": "2025-03-15",
                    "gaps": 2
                },
                "iso27001": {
                    "status": "Planning",
                    "completion": 45,
                    "next_audit": "2025-06-01",
                    "gaps": 8
                },
                "gdpr": {
                    "status": "Compliant",
                    "completion": 95,
                    "last_review": "2025-01-15",
                    "gaps": 1
                }
            },

            "criteria_scores": compliance_data.get("criteria_scores", {}),

            "recent_evidence": await self._get_recent_evidence_collection(tenant_id),

            "upcoming_reviews": await self._get_upcoming_compliance_reviews(tenant_id),

            "recommendations": [
                "Complete remaining SOC2 Type II controls",
                "Schedule ISO 27001 gap assessment",
                "Update privacy policy for GDPR compliance",
                "Implement automated evidence collection"
            ]
        }

        return dashboard

    async def generate_business_impact_dashboard(self, tenant_id: str, period_days: int = 90) -> Dict[str, Any]:
        """Generate business impact dashboard"""

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        # Calculate security ROI and business metrics
        security_investment = await self._calculate_security_investment(tenant_id, start_date, end_date)
        avoided_losses = await self._calculate_avoided_losses(tenant_id, start_date, end_date)
        productivity_impact = await self._calculate_productivity_impact(tenant_id, start_date, end_date)

        dashboard = {
            "dashboard_type": "business_impact",
            "tenant_id": tenant_id,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "generated_at": datetime.utcnow().isoformat(),

            "financial_summary": {
                "security_investment": security_investment,
                "losses_avoided": avoided_losses,
                "net_benefit": avoided_losses - security_investment,
                "roi_percentage": ((avoided_losses - security_investment) / security_investment * 100) if security_investment > 0 else 0,
                "cost_per_incident": security_investment / max(await self._get_incident_count(tenant_id, start_date, end_date), 1)
            },

            "productivity_metrics": {
                "uptime_percentage": 99.8,
                "incidents_preventing_work": 3,
                "time_saved_automation": 240,  # hours
                "user_satisfaction_score": 4.2
            },

            "risk_reduction": {
                "vulnerability_reduction": 65,  # percentage
                "compliance_improvement": 23,  # percentage
                "incident_reduction": 40,  # percentage
                "response_time_improvement": 60  # percentage
            },

            "charts": {
                "roi_trend": await self._get_roi_trend_data(tenant_id, start_date, end_date),
                "cost_breakdown": await self._get_security_cost_breakdown(tenant_id),
                "value_creation": await self._get_value_creation_metrics(tenant_id)
            }
        }

        return dashboard

    async def generate_operational_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Generate operational metrics dashboard"""

        dashboard = {
            "dashboard_type": "operational_metrics",
            "tenant_id": tenant_id,
            "generated_at": datetime.utcnow().isoformat(),

            "system_health": {
                "platform_uptime": 99.95,
                "api_response_time": 145,  # ms
                "active_users": await self._get_active_user_count(tenant_id),
                "data_processed_gb": 1240,
                "alerts_generated": 45,
                "false_positive_rate": 3.2
            },

            "team_performance": {
                "incident_response_team": {
                    "avg_response_time": 12,  # minutes
                    "sla_compliance": 94,
                    "escalations": 2,
                    "team_size": 8
                },
                "security_analysts": {
                    "cases_resolved": 156,
                    "avg_resolution_time": 4.2,  # hours
                    "quality_score": 4.6,
                    "training_hours": 24
                }
            },

            "automation_metrics": {
                "automated_responses": 89,
                "manual_interventions": 12,
                "automation_success_rate": 96.4,
                "time_saved_hours": 180
            },

            "capacity_planning": {
                "cpu_utilization": 65,
                "memory_utilization": 72,
                "storage_utilization": 58,
                "network_throughput": 1.2  # Gbps
            }
        }

        return dashboard

    async def _get_security_metrics(self, tenant_id: str, start_date: datetime, end_date: datetime) -> List[ExecutiveMetric]:
        """Get key security metrics for executives"""
        return [
            ExecutiveMetric(
                metric_id="security_posture",
                name="Security Posture Score",
                description="Overall security posture based on controls and compliance",
                current_value=87.5,
                target_value=95.0,
                previous_value=84.2,
                unit="score",
                trend=MetricTrend.IMPROVING,
                impact_level="high",
                last_updated=datetime.utcnow(),
                drill_down_url="/dashboards/security-posture"
            ),
            ExecutiveMetric(
                metric_id="mean_time_to_response",
                name="Mean Time to Response",
                description="Average time to respond to security incidents",
                current_value=15.2,
                target_value=10.0,
                previous_value=18.7,
                unit="minutes",
                trend=MetricTrend.IMPROVING,
                impact_level="medium",
                last_updated=datetime.utcnow(),
                drill_down_url="/dashboards/incident-response"
            ),
            ExecutiveMetric(
                metric_id="vulnerability_exposure",
                name="Critical Vulnerability Exposure",
                description="Number of critical vulnerabilities in production",
                current_value=3,
                target_value=0,
                previous_value=7,
                unit="count",
                trend=MetricTrend.IMPROVING,
                impact_level="high",
                last_updated=datetime.utcnow(),
                drill_down_url="/dashboards/vulnerabilities"
            ),
            ExecutiveMetric(
                metric_id="compliance_score",
                name="Compliance Score",
                description="Overall compliance with security frameworks",
                current_value=92.3,
                target_value=95.0,
                previous_value=89.1,
                unit="percentage",
                trend=MetricTrend.IMPROVING,
                impact_level="high",
                last_updated=datetime.utcnow(),
                drill_down_url="/dashboards/compliance"
            )
        ]

    async def _get_recent_incidents(self, tenant_id: str, limit: int = 10) -> List[SecurityIncident]:
        """Get recent security incidents"""
        # This would query the actual incident database
        return [
            SecurityIncident(
                incident_id="INC-2025-001",
                title="Suspicious API Access Detected",
                severity="medium",
                status="investigating",
                business_impact="Low - No data accessed",
                estimated_loss=5000.0,
                response_time=8,
                assigned_team="Security Operations",
                created_at=datetime.utcnow() - timedelta(hours=2)
            ),
            SecurityIncident(
                incident_id="INC-2025-002",
                title="Failed Login Attempts from Unknown Location",
                severity="low",
                status="resolved",
                business_impact="None - Blocked by MFA",
                estimated_loss=0.0,
                response_time=12,
                assigned_team="SOC Tier 1",
                created_at=datetime.utcnow() - timedelta(hours=6),
                resolved_at=datetime.utcnow() - timedelta(hours=4)
            )
        ]

    async def _calculate_security_posture(self, tenant_id: str) -> float:
        """Calculate overall security posture score"""
        # Composite score based on multiple factors
        factors = {
            "vulnerability_management": 85,
            "access_controls": 92,
            "incident_response": 88,
            "compliance": 90,
            "threat_detection": 87,
            "data_protection": 94
        }

        return sum(factors.values()) / len(factors)

    async def _get_compliance_score(self, tenant_id: str) -> float:
        """Get overall compliance score"""
        return 92.3

    async def _generate_executive_recommendations(
        self,
        tenant_id: str,
        metrics: List[ExecutiveMetric],
        incidents: List[SecurityIncident]
    ) -> List[str]:
        """Generate executive-level recommendations"""
        recommendations = []

        # Analyze metrics for recommendations
        for metric in metrics:
            if metric.current_value < metric.target_value:
                if metric.metric_id == "mean_time_to_response":
                    recommendations.append("Consider investing in automated incident response tools")
                elif metric.metric_id == "vulnerability_exposure":
                    recommendations.append("Implement continuous vulnerability scanning")

        # Analyze incidents for patterns
        if len([i for i in incidents if i.severity == "critical"]) > 2:
            recommendations.append("Review and strengthen threat detection capabilities")

        return recommendations[:5]  # Top 5 recommendations

    def _is_cached(self, cache_key: str) -> bool:
        """Check if dashboard is cached and not expired"""
        if cache_key not in self.dashboard_cache:
            return False

        cached_data = self.dashboard_cache[cache_key]
        cache_time = datetime.fromisoformat(cached_data.get("cached_at", "1970-01-01T00:00:00"))

        return (datetime.utcnow() - cache_time).seconds < self.cache_ttl

    def _cache_dashboard(self, cache_key: str, dashboard: Dict[str, Any]):
        """Cache dashboard data"""
        dashboard["cached_at"] = datetime.utcnow().isoformat()
        self.dashboard_cache[cache_key] = dashboard

    async def _mock_compliance_data(self) -> Dict[str, Any]:
        """Mock compliance data when module not available"""
        return {
            "overview": {
                "overall_compliance_score": 87.5,
                "controls_tested": 15,
                "controls_passed": 13
            },
            "criteria_scores": {
                "security": 89,
                "availability": 91,
                "processing_integrity": 85,
                "confidentiality": 88,
                "privacy": 82
            }
        }

    # Additional helper methods would continue here...
    async def _get_threat_landscape_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get threat landscape summary"""
        return {
            "total_threats_detected": 245,
            "blocked_attacks": 198,
            "top_threat_categories": ["Malware", "Phishing", "Insider Threat"],
            "threat_intelligence_feeds": 12,
            "iocs_processed": 1456
        }

    async def _get_incident_trend_data(self, tenant_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get incident trend data for charts"""
        return [
            {"date": "2025-01-01", "incidents": 5, "critical": 1},
            {"date": "2025-01-02", "incidents": 3, "critical": 0},
            {"date": "2025-01-03", "incidents": 7, "critical": 2}
        ]


# Global dashboard engine
executive_dashboard = ExecutiveDashboardEngine()


async def get_executive_dashboard(tenant_id: str, dashboard_type: str, period_days: int = 30) -> Dict[str, Any]:
    """API endpoint for executive dashboards"""

    if dashboard_type == "security_overview":
        return await executive_dashboard.generate_security_overview(tenant_id, period_days)
    elif dashboard_type == "risk_management":
        return await executive_dashboard.generate_risk_management_dashboard(tenant_id)
    elif dashboard_type == "compliance_status":
        return await executive_dashboard.generate_compliance_dashboard(tenant_id)
    elif dashboard_type == "business_impact":
        return await executive_dashboard.generate_business_impact_dashboard(tenant_id, period_days)
    elif dashboard_type == "operational_metrics":
        return await executive_dashboard.generate_operational_dashboard(tenant_id)
    else:
        raise ValueError(f"Unknown dashboard type: {dashboard_type}")
