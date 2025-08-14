"""
Enterprise Compliance Automation Service
Comprehensive compliance management, reporting, and automation
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib

from .base_service import SecurityService, ServiceHealth, ServiceStatus
from .interfaces import ComplianceService

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    PCI_DSS = "pci-dss"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso-27001"
    GDPR = "gdpr"
    NIST_CSF = "nist-csf"
    FISMA = "fisma"
    FedRAMP = "fedramp"
    SOC2 = "soc2"
    CCPA = "ccpa"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"


class ControlSeverity(Enum):
    """Control severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class RemediationPriority(Enum):
    """Remediation priority levels"""
    IMMEDIATE = "immediate"      # 0-24 hours
    HIGH = "high"               # 1-7 days
    MEDIUM = "medium"           # 1-30 days
    LOW = "low"                 # 30-90 days
    PLANNED = "planned"         # >90 days


@dataclass
class ComplianceControl:
    """Individual compliance control"""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirements: List[str]
    severity: ControlSeverity
    automated_check: bool = False
    check_frequency: str = "quarterly"
    responsible_team: str = "security"
    implementation_guidance: str = ""
    testing_procedures: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)


@dataclass
class ControlAssessment:
    """Assessment of a compliance control"""
    assessment_id: str
    control_id: str
    assessment_date: datetime
    assessor: str
    status: ComplianceStatus
    score: float  # 0.0 - 1.0
    findings: List[str]
    evidence_collected: List[str]
    gaps_identified: List[str]
    recommendations: List[str]
    next_assessment_due: datetime
    remediation_plan: Optional[str] = None


@dataclass
class ComplianceGap:
    """Identified compliance gap"""
    gap_id: str
    control_id: str
    framework: ComplianceFramework
    gap_description: str
    risk_level: ControlSeverity
    business_impact: str
    remediation_steps: List[str]
    remediation_priority: RemediationPriority
    assigned_to: str
    target_completion: datetime
    estimated_effort: str
    cost_estimate: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    status: str = "open"


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    framework: ComplianceFramework
    report_type: str
    generated_at: datetime
    reporting_period: Dict[str, str]
    overall_score: float
    compliance_percentage: float
    executive_summary: Dict[str, Any]
    control_assessments: List[ControlAssessment]
    gaps_summary: List[ComplianceGap]
    recommendations: List[str]
    next_actions: List[str]
    appendices: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RemediationItem:
    """Individual remediation task"""
    item_id: str
    gap_id: str
    title: str
    description: str
    priority: RemediationPriority
    assigned_to: str
    due_date: datetime
    estimated_hours: int
    cost_estimate: Optional[float] = None
    status: str = "pending"
    progress_percentage: int = 0
    dependencies: List[str] = field(default_factory=list)
    updates: List[Dict[str, Any]] = field(default_factory=list)
    completion_date: Optional[datetime] = None


class EnterpriseComplianceAutomationService(SecurityService, ComplianceService):
    """
    Enterprise-grade compliance automation service
    """

    def __init__(self, **kwargs):
        super().__init__(
            service_id="compliance_automation",
            dependencies=["database", "scanner", "threat_intelligence"],
            **kwargs
        )

        # Core components
        self.compliance_controls: Dict[str, ComplianceControl] = {}
        self.control_assessments: Dict[str, ControlAssessment] = {}
        self.compliance_gaps: Dict[str, ComplianceGap] = {}
        self.remediation_items: Dict[str, RemediationItem] = {}
        self.compliance_reports: Dict[str, ComplianceReport] = {}

        # Framework mappings
        self.framework_controls = self._initialize_framework_controls()
        self.control_mappings = self._initialize_control_mappings()

        # Automation components
        self.automated_checks: Dict[str, Any] = {}
        self.assessment_schedules: Dict[str, datetime] = {}
        self.notification_templates: Dict[str, str] = {}

        # Analytics and metrics
        self.compliance_metrics = {
            "total_frameworks": 0,
            "total_controls": 0,
            "automated_controls": 0,
            "compliant_controls": 0,
            "non_compliant_controls": 0,
            "open_gaps": 0,
            "remediation_items": 0,
            "overdue_assessments": 0,
            "reports_generated": 0
        }

        # Background tasks
        self.automation_tasks: List[asyncio.Task] = []

        # Configuration
        self.auto_assessment_enabled = True
        self.continuous_monitoring = True
        self.real_time_alerts = True

    async def initialize(self) -> bool:
        """Initialize the compliance automation service"""
        try:
            logger.info("Initializing Enterprise Compliance Automation Service...")

            # Load compliance frameworks and controls
            await self._load_compliance_frameworks()

            # Initialize automated checks
            await self._initialize_automated_checks()

            # Load existing assessments and gaps
            await self._load_existing_data()

            # Start automation tasks
            if self.auto_assessment_enabled:
                await self._start_automation_tasks()

            # Start continuous monitoring
            if self.continuous_monitoring:
                await self._start_continuous_monitoring()

            logger.info("Compliance Automation Service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize compliance automation service: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the service"""
        try:
            # Cancel automation tasks
            for task in self.automation_tasks:
                task.cancel()

            await asyncio.gather(*self.automation_tasks, return_exceptions=True)

            # Save state
            await self._save_compliance_state()

            logger.info("Compliance Automation Service shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    # ========================================================================
    # COMPLIANCE VALIDATION
    # ========================================================================

    async def validate_compliance(
        self,
        framework: ComplianceFramework,
        scope: Dict[str, Any],
        assessment_type: str = "full"
    ) -> Dict[str, Any]:
        """Validate compliance against specific framework"""

        try:
            validation_id = str(uuid.uuid4())

            logger.info(f"Starting {framework.value} compliance validation")

            validation_result = {
                "validation_id": validation_id,
                "framework": framework.value,
                "assessment_type": assessment_type,
                "scope": scope,
                "started_at": datetime.utcnow().isoformat(),
                "control_results": [],
                "overall_compliance": 0.0,
                "gaps_identified": [],
                "recommendations": [],
                "next_actions": []
            }

            # Get framework controls
            framework_controls = self.framework_controls.get(framework.value, [])

            if not framework_controls:
                raise ValueError(f"No controls defined for framework {framework.value}")

            # Assess each control
            total_score = 0.0
            assessed_controls = 0

            for control_id in framework_controls:
                if control_id in self.compliance_controls:
                    control = self.compliance_controls[control_id]

                    # Perform control assessment
                    assessment_result = await self._assess_control(control, scope)
                    validation_result["control_results"].append(assessment_result)

                    total_score += assessment_result["score"]
                    assessed_controls += 1

                    # Identify gaps
                    if assessment_result["status"] != ComplianceStatus.COMPLIANT.value:
                        gap = await self._identify_control_gap(control, assessment_result)
                        validation_result["gaps_identified"].append(gap)

            # Calculate overall compliance
            if assessed_controls > 0:
                validation_result["overall_compliance"] = total_score / assessed_controls
                validation_result["compliance_percentage"] = validation_result["overall_compliance"] * 100

            # Generate recommendations
            validation_result["recommendations"] = await self._generate_compliance_recommendations(
                validation_result["control_results"],
                validation_result["gaps_identified"]
            )

            # Generate next actions
            validation_result["next_actions"] = await self._generate_next_actions(
                validation_result["gaps_identified"]
            )

            validation_result["completed_at"] = datetime.utcnow().isoformat()

            # Update metrics
            self.compliance_metrics["reports_generated"] += 1

            return validation_result

        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            raise

    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime,
        report_type: str = "executive"
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""

        try:
            report_id = str(uuid.uuid4())

            logger.info(f"Generating {report_type} compliance report for {framework.value}")

            # Collect assessment data for period
            assessments = await self._get_assessments_for_period(
                framework, period_start, period_end
            )

            # Calculate compliance metrics
            compliance_metrics = await self._calculate_compliance_metrics(assessments)

            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                framework, compliance_metrics, assessments
            )

            # Compile gaps and remediation status
            gaps_summary = await self._compile_gaps_summary(framework, period_start, period_end)

            # Generate detailed findings
            detailed_findings = await self._generate_detailed_findings(assessments)

            # Create compliance report
            report = ComplianceReport(
                report_id=report_id,
                framework=framework,
                report_type=report_type,
                generated_at=datetime.utcnow(),
                reporting_period={
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat()
                },
                overall_score=compliance_metrics["overall_score"],
                compliance_percentage=compliance_metrics["compliance_percentage"],
                executive_summary=executive_summary,
                control_assessments=assessments,
                gaps_summary=gaps_summary,
                recommendations=await self._generate_strategic_recommendations(
                    compliance_metrics, gaps_summary
                ),
                next_actions=await self._generate_action_plan(gaps_summary)
            )

            # Add framework-specific appendices
            if report_type == "detailed":
                report.appendices = await self._generate_report_appendices(
                    framework, assessments, gaps_summary
                )

            # Store report
            self.compliance_reports[report_id] = report

            # Update metrics
            self.compliance_metrics["reports_generated"] += 1

            return asdict(report)

        except Exception as e:
            logger.error(f"Compliance report generation failed: {e}")
            raise

    async def get_compliance_gaps(
        self,
        framework: Optional[ComplianceFramework] = None,
        severity_filter: Optional[ControlSeverity] = None
    ) -> List[Dict[str, Any]]:
        """Identify compliance gaps and remediation steps"""

        try:
            gaps = []

            for gap in self.compliance_gaps.values():
                # Apply filters
                if framework and gap.framework != framework:
                    continue
                if severity_filter and gap.risk_level != severity_filter:
                    continue

                # Get remediation items for this gap
                remediation_items = [
                    asdict(item) for item in self.remediation_items.values()
                    if item.gap_id == gap.gap_id
                ]

                # Calculate progress
                total_items = len(remediation_items)
                completed_items = len([item for item in remediation_items if item["status"] == "completed"])
                progress_percentage = (completed_items / total_items * 100) if total_items > 0 else 0

                gap_data = asdict(gap)
                gap_data["remediation_items"] = remediation_items
                gap_data["progress_percentage"] = progress_percentage
                gap_data["days_until_target"] = (gap.target_completion - datetime.utcnow()).days

                gaps.append(gap_data)

            # Sort by priority and risk level
            gaps.sort(key=lambda x: (
                x["remediation_priority"],
                x["risk_level"],
                x["target_completion"]
            ))

            return gaps

        except Exception as e:
            logger.error(f"Failed to get compliance gaps: {e}")
            raise

    async def track_remediation_progress(
        self,
        gap_id: Optional[str] = None,
        framework: Optional[ComplianceFramework] = None
    ) -> Dict[str, Any]:
        """Track progress of compliance remediation efforts"""

        try:
            progress_report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "scope": {
                    "gap_id": gap_id,
                    "framework": framework.value if framework else "all"
                },
                "summary": {},
                "remediation_items": [],
                "timeline": [],
                "metrics": {}
            }

            # Filter remediation items
            filtered_items = []
            for item in self.remediation_items.values():
                if gap_id and self.compliance_gaps[item.gap_id].gap_id != gap_id:
                    continue
                if framework and self.compliance_gaps[item.gap_id].framework != framework:
                    continue
                filtered_items.append(item)

            # Calculate progress metrics
            total_items = len(filtered_items)
            completed_items = len([item for item in filtered_items if item.status == "completed"])
            in_progress_items = len([item for item in filtered_items if item.status == "in_progress"])
            overdue_items = len([
                item for item in filtered_items
                if item.due_date < datetime.utcnow() and item.status != "completed"
            ])

            progress_report["summary"] = {
                "total_items": total_items,
                "completed_items": completed_items,
                "in_progress_items": in_progress_items,
                "pending_items": total_items - completed_items - in_progress_items,
                "overdue_items": overdue_items,
                "overall_progress": (completed_items / total_items * 100) if total_items > 0 else 0
            }

            # Detailed item information
            for item in filtered_items:
                item_data = asdict(item)
                item_data["gap_info"] = asdict(self.compliance_gaps[item.gap_id])
                item_data["is_overdue"] = item.due_date < datetime.utcnow() and item.status != "completed"
                progress_report["remediation_items"].append(item_data)

            # Generate timeline
            progress_report["timeline"] = await self._generate_remediation_timeline(filtered_items)

            # Calculate cost and effort metrics
            total_cost = sum([item.cost_estimate or 0 for item in filtered_items])
            total_effort = sum([item.estimated_hours for item in filtered_items])
            completed_cost = sum([
                item.cost_estimate or 0 for item in filtered_items
                if item.status == "completed"
            ])

            progress_report["metrics"] = {
                "total_estimated_cost": total_cost,
                "completed_cost": completed_cost,
                "remaining_cost": total_cost - completed_cost,
                "total_estimated_hours": total_effort,
                "cost_efficiency": (completed_cost / total_cost * 100) if total_cost > 0 else 0
            }

            return progress_report

        except Exception as e:
            logger.error(f"Failed to track remediation progress: {e}")
            raise

    # ========================================================================
    # AUTOMATED COMPLIANCE MONITORING
    # ========================================================================

    async def run_automated_assessment(
        self,
        control_id: str,
        assessment_config: Optional[Dict[str, Any]] = None
    ) -> ControlAssessment:
        """Run automated compliance assessment for a control"""

        try:
            if control_id not in self.compliance_controls:
                raise ValueError(f"Control {control_id} not found")

            control = self.compliance_controls[control_id]

            if not control.automated_check:
                raise ValueError(f"Control {control_id} does not support automated assessment")

            logger.info(f"Running automated assessment for control {control_id}")

            # Get automated check configuration
            check_config = self.automated_checks.get(control_id, {})
            if assessment_config:
                check_config.update(assessment_config)

            # Perform automated checks
            assessment_result = await self._run_automated_control_check(control, check_config)

            # Create assessment record
            assessment = ControlAssessment(
                assessment_id=str(uuid.uuid4()),
                control_id=control_id,
                assessment_date=datetime.utcnow(),
                assessor="automated_system",
                status=assessment_result["status"],
                score=assessment_result["score"],
                findings=assessment_result["findings"],
                evidence_collected=assessment_result["evidence"],
                gaps_identified=assessment_result["gaps"],
                recommendations=assessment_result["recommendations"],
                next_assessment_due=datetime.utcnow() + timedelta(days=30)  # Default 30 days
            )

            # Store assessment
            self.control_assessments[assessment.assessment_id] = assessment

            # Update metrics
            if assessment.status == ComplianceStatus.COMPLIANT:
                self.compliance_metrics["compliant_controls"] += 1
            else:
                self.compliance_metrics["non_compliant_controls"] += 1

            # Create compliance gap if non-compliant
            if assessment.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT]:
                await self._create_compliance_gap_from_assessment(assessment)

            logger.info(f"Automated assessment completed for {control_id}: {assessment.status.value}")

            return assessment

        except Exception as e:
            logger.error(f"Automated assessment failed for {control_id}: {e}")
            raise

    async def schedule_recurring_assessments(
        self,
        framework: ComplianceFramework,
        frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        """Schedule recurring compliance assessments"""

        try:
            schedule_id = str(uuid.uuid4())

            # Calculate next assessment dates
            if frequency == "monthly":
                next_date = datetime.utcnow() + timedelta(days=30)
                interval_days = 30
            elif frequency == "quarterly":
                next_date = datetime.utcnow() + timedelta(days=90)
                interval_days = 90
            elif frequency == "annual":
                next_date = datetime.utcnow() + timedelta(days=365)
                interval_days = 365
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")

            # Get framework controls
            framework_controls = self.framework_controls.get(framework.value, [])

            schedule_config = {
                "schedule_id": schedule_id,
                "framework": framework.value,
                "frequency": frequency,
                "interval_days": interval_days,
                "next_assessment": next_date.isoformat(),
                "controls_included": framework_controls,
                "auto_generate_reports": True,
                "notification_enabled": True,
                "created_at": datetime.utcnow().isoformat()
            }

            # Schedule assessments for each control
            for control_id in framework_controls:
                if control_id in self.compliance_controls:
                    control = self.compliance_controls[control_id]
                    if control.automated_check:
                        self.assessment_schedules[f"{schedule_id}_{control_id}"] = next_date

            # Start recurring assessment task
            recurring_task = asyncio.create_task(
                self._run_recurring_assessments(schedule_config)
            )
            self.automation_tasks.append(recurring_task)

            logger.info(f"Scheduled recurring {frequency} assessments for {framework.value}")

            return schedule_config

        except Exception as e:
            logger.error(f"Failed to schedule recurring assessments: {e}")
            raise

    # ========================================================================
    # COMPLIANCE SERVICE INTERFACE IMPLEMENTATION
    # ========================================================================

    async def validate_compliance(
        self,
        framework: str,
        scope: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate compliance against specific framework"""
        try:
            framework_enum = ComplianceFramework(framework)
            return await self.validate_compliance(framework_enum, scope)
        except ValueError:
            raise ValueError(f"Unsupported compliance framework: {framework}")

    async def generate_compliance_report(
        self,
        framework: str,
        period_start: str,
        period_end: str
    ) -> Dict[str, Any]:
        """Generate compliance report for specified period"""
        try:
            framework_enum = ComplianceFramework(framework)
            start_date = datetime.fromisoformat(period_start)
            end_date = datetime.fromisoformat(period_end)

            return await self.generate_compliance_report(
                framework_enum, start_date, end_date
            )
        except ValueError:
            raise ValueError(f"Invalid framework or date format")

    async def get_compliance_gaps(
        self,
        framework: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Identify compliance gaps and remediation steps"""
        try:
            framework_enum = None
            if framework:
                framework_enum = ComplianceFramework(framework)

            return await self.get_compliance_gaps(framework_enum)
        except ValueError:
            raise ValueError(f"Unsupported compliance framework: {framework}")

    async def track_remediation_progress(
        self,
        gap_id: str
    ) -> Dict[str, Any]:
        """Track progress of compliance remediation efforts"""
        return await self.track_remediation_progress(gap_id)

    # ========================================================================
    # SECURITY SERVICE INTERFACE IMPLEMENTATION
    # ========================================================================

    async def process_security_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a security event"""
        try:
            event_type = event.get("type", "unknown")

            if event_type == "compliance_violation":
                return await self._process_compliance_violation_event(event)
            elif event_type == "control_failure":
                return await self._process_control_failure_event(event)
            elif event_type == "audit_finding":
                return await self._process_audit_finding_event(event)
            else:
                return {"status": "ignored", "reason": f"Unknown event type: {event_type}"}

        except Exception as e:
            logger.error(f"Failed to process security event: {e}")
            return {"status": "error", "error": str(e)}

    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-specific metrics"""
        try:
            return {
                **self.compliance_metrics,
                "automated_checks_available": len(self.automated_checks),
                "assessment_schedules_active": len(self.assessment_schedules),
                "overdue_remediations": len([
                    item for item in self.remediation_items.values()
                    if item.due_date < datetime.utcnow() and item.status != "completed"
                ]),
                "critical_gaps": len([
                    gap for gap in self.compliance_gaps.values()
                    if gap.risk_level == ControlSeverity.CRITICAL
                ]),
                "frameworks_monitored": len(set([
                    gap.framework.value for gap in self.compliance_gaps.values()
                ]))
            }

        except Exception as e:
            logger.error(f"Failed to get security metrics: {e}")
            return {"error": str(e)}

    async def health_check(self) -> ServiceHealth:
        """Perform health check"""
        try:
            checks = {
                "frameworks_loaded": len(self.framework_controls),
                "controls_defined": len(self.compliance_controls),
                "automated_checks": len(self.automated_checks),
                "active_assessments": len(self.control_assessments),
                "open_gaps": len([g for g in self.compliance_gaps.values() if g.status == "open"]),
                "automation_tasks": len([t for t in self.automation_tasks if not t.done()]),
                "continuous_monitoring": self.continuous_monitoring
            }

            status = ServiceStatus.HEALTHY
            message = "Compliance automation service operational"

            # Check for critical issues
            critical_gaps = len([
                gap for gap in self.compliance_gaps.values()
                if gap.risk_level == ControlSeverity.CRITICAL and gap.status == "open"
            ])

            if critical_gaps > 0:
                status = ServiceStatus.DEGRADED
                message = f"{critical_gaps} critical compliance gaps require attention"

            overdue_assessments = len([
                schedule_date for schedule_date in self.assessment_schedules.values()
                if schedule_date < datetime.utcnow()
            ])

            if overdue_assessments > 10:
                status = ServiceStatus.DEGRADED
                message = f"{overdue_assessments} assessments overdue"

            return ServiceHealth(
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                checks=checks
            )

        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _initialize_framework_controls(self) -> Dict[str, List[str]]:
        """Initialize framework control mappings"""
        return {
            "pci-dss": [
                "pci_1.1", "pci_1.2", "pci_2.1", "pci_2.2", "pci_3.1", "pci_3.2",
                "pci_4.1", "pci_4.2", "pci_6.1", "pci_6.2", "pci_8.1", "pci_8.2",
                "pci_10.1", "pci_10.2", "pci_11.1", "pci_11.2", "pci_12.1", "pci_12.2"
            ],
            "hipaa": [
                "hipaa_164.308", "hipaa_164.310", "hipaa_164.312", "hipaa_164.314",
                "hipaa_164.316", "hipaa_164.318", "hipaa_164.320", "hipaa_164.322"
            ],
            "sox": [
                "sox_302", "sox_404", "sox_409", "sox_802", "sox_906"
            ],
            "iso-27001": [
                "iso_a5", "iso_a6", "iso_a7", "iso_a8", "iso_a9", "iso_a10",
                "iso_a11", "iso_a12", "iso_a13", "iso_a14", "iso_a15", "iso_a16"
            ]
        }

    def _initialize_control_mappings(self) -> Dict[str, List[str]]:
        """Initialize cross-framework control mappings"""
        return {
            "access_control": ["pci_8.1", "hipaa_164.308", "iso_a9"],
            "encryption": ["pci_3.1", "hipaa_164.312", "iso_a10"],
            "network_security": ["pci_1.1", "iso_a13"],
            "vulnerability_management": ["pci_6.1", "iso_a12"],
            "incident_response": ["pci_12.1", "iso_a16"],
            "audit_logging": ["pci_10.1", "sox_404", "iso_a12"]
        }

    # Additional utility methods would be implemented here...


# Global service instance
_compliance_service: Optional[EnterpriseComplianceAutomationService] = None

async def get_compliance_automation_service() -> EnterpriseComplianceAutomationService:
    """Get global compliance automation service instance"""
    global _compliance_service

    if _compliance_service is None:
        _compliance_service = EnterpriseComplianceAutomationService()
        await _compliance_service.initialize()

        # Register with global service registry
        from .base_service import service_registry
        service_registry.register(_compliance_service)

    return _compliance_service
