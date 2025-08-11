#!/usr/bin/env python3
"""
Automated Compliance Reporting Service
SOC2, ISO27001, NIST CSF, and GDPR compliance automation for XORB Platform

This service provides:
- Automated compliance control validation
- Real-time compliance monitoring
- Automated report generation
- Evidence collection and management
- Compliance dashboard and metrics
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOC2_TYPE_II = "soc2_type_ii"
    ISO27001 = "iso27001"
    NIST_CSF = "nist_csf"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"


class ControlStatus(Enum):
    """Compliance control status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNDER_REVIEW = "under_review"


class EvidenceType(Enum):
    """Types of compliance evidence"""
    LOG_DATA = "log_data"
    CONFIGURATION = "configuration"
    POLICY_DOCUMENT = "policy_document"
    AUDIT_TRAIL = "audit_trail"
    SECURITY_SCAN = "security_scan"
    ACCESS_REVIEW = "access_review"
    INCIDENT_REPORT = "incident_report"
    TRAINING_RECORD = "training_record"


@dataclass
class ComplianceControl:
    """Compliance control definition"""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    control_family: str
    implementation_guidance: str
    testing_procedures: List[str]
    evidence_requirements: List[EvidenceType]
    automation_possible: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL


@dataclass
class ControlAssessment:
    """Assessment result for a compliance control"""
    control_id: str
    framework: ComplianceFramework
    status: ControlStatus
    compliance_score: float  # 0.0 - 1.0
    assessment_date: datetime
    assessor: str
    findings: List[str]
    evidence_collected: List[str]
    remediation_required: List[str]
    next_assessment_due: datetime
    automated: bool


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    framework: ComplianceFramework
    reporting_period_start: datetime
    reporting_period_end: datetime
    overall_compliance_score: float
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    control_assessments: List[ControlAssessment]
    executive_summary: str
    remediation_plan: List[str]
    generated_by: str
    generated_at: datetime


class ComplianceControlDatabase:
    """Database of compliance controls and requirements"""
    
    def __init__(self):
        self.controls: Dict[str, ComplianceControl] = {}
        self._initialize_controls()
    
    def _initialize_controls(self):
        """Initialize compliance controls for supported frameworks"""
        # SOC2 Type II Controls
        soc2_controls = [
            ComplianceControl(
                control_id="CC1.1",
                framework=ComplianceFramework.SOC2_TYPE_II,
                title="Control Environment - Management Oversight",
                description="The entity demonstrates a commitment to integrity and ethical values",
                control_family="Common Criteria",
                implementation_guidance="Establish and maintain code of conduct, ethics policies, and management oversight",
                testing_procedures=[
                    "Review and test code of conduct",
                    "Verify management oversight procedures",
                    "Test whistleblower mechanisms"
                ],
                evidence_requirements=[
                    EvidenceType.POLICY_DOCUMENT,
                    EvidenceType.TRAINING_RECORD,
                    EvidenceType.AUDIT_TRAIL
                ],
                automation_possible=True,
                risk_level="HIGH"
            ),
            ComplianceControl(
                control_id="CC6.1",
                framework=ComplianceFramework.SOC2_TYPE_II,
                title="Data Encryption",
                description="The entity implements logical access security software and infrastructure",
                control_family="Logical and Physical Access Controls",
                implementation_guidance="Implement encryption for data at rest and in transit",
                testing_procedures=[
                    "Test encryption implementation",
                    "Verify key management procedures",
                    "Review encryption standards compliance"
                ],
                evidence_requirements=[
                    EvidenceType.CONFIGURATION,
                    EvidenceType.SECURITY_SCAN,
                    EvidenceType.LOG_DATA
                ],
                automation_possible=True,
                risk_level="CRITICAL"
            ),
            ComplianceControl(
                control_id="CC6.6",
                framework=ComplianceFramework.SOC2_TYPE_II,
                title="Data Classification and Handling",
                description="The entity implements controls to protect against unauthorized disclosure",
                control_family="Logical and Physical Access Controls",
                implementation_guidance="Classify data and implement appropriate handling procedures",
                testing_procedures=[
                    "Review data classification scheme",
                    "Test data handling procedures",
                    "Verify access controls by classification"
                ],
                evidence_requirements=[
                    EvidenceType.POLICY_DOCUMENT,
                    EvidenceType.ACCESS_REVIEW,
                    EvidenceType.CONFIGURATION
                ],
                automation_possible=True,
                risk_level="HIGH"
            ),
            ComplianceControl(
                control_id="CC6.7",
                framework=ComplianceFramework.SOC2_TYPE_II,
                title="Data Transmission Security",
                description="The entity restricts the transmission of data to authorized users",
                control_family="Logical and Physical Access Controls",
                implementation_guidance="Implement secure transmission protocols and monitoring",
                testing_procedures=[
                    "Test transmission encryption",
                    "Review transmission logs",
                    "Verify authorized user restrictions"
                ],
                evidence_requirements=[
                    EvidenceType.CONFIGURATION,
                    EvidenceType.LOG_DATA,
                    EvidenceType.SECURITY_SCAN
                ],
                automation_possible=True,
                risk_level="CRITICAL"
            )
        ]
        
        # ISO27001 Controls
        iso27001_controls = [
            ComplianceControl(
                control_id="A.10.1",
                framework=ComplianceFramework.ISO27001,
                title="Cryptographic Controls",
                description="A policy on the use of cryptographic controls for protection of information",
                control_family="Cryptography",
                implementation_guidance="Develop and implement cryptographic policy and procedures",
                testing_procedures=[
                    "Review cryptographic policy",
                    "Test cryptographic implementations",
                    "Verify key management procedures"
                ],
                evidence_requirements=[
                    EvidenceType.POLICY_DOCUMENT,
                    EvidenceType.CONFIGURATION,
                    EvidenceType.SECURITY_SCAN
                ],
                automation_possible=True,
                risk_level="CRITICAL"
            ),
            ComplianceControl(
                control_id="A.13.1",
                framework=ComplianceFramework.ISO27001,
                title="Network Security Management",
                description="Network controls shall be managed and controlled",
                control_family="Communications Security",
                implementation_guidance="Implement network security controls and monitoring",
                testing_procedures=[
                    "Test network segmentation",
                    "Review firewall configurations",
                    "Verify network monitoring"
                ],
                evidence_requirements=[
                    EvidenceType.CONFIGURATION,
                    EvidenceType.LOG_DATA,
                    EvidenceType.SECURITY_SCAN
                ],
                automation_possible=True,
                risk_level="HIGH"
            ),
            ComplianceControl(
                control_id="A.13.2",
                framework=ComplianceFramework.ISO27001,
                title="Information Transfer",
                description="Information transfer shall be subject to formal transfer policies",
                control_family="Communications Security",
                implementation_guidance="Implement secure information transfer procedures",
                testing_procedures=[
                    "Review transfer policies",
                    "Test transfer mechanisms",
                    "Verify authorization procedures"
                ],
                evidence_requirements=[
                    EvidenceType.POLICY_DOCUMENT,
                    EvidenceType.LOG_DATA,
                    EvidenceType.AUDIT_TRAIL
                ],
                automation_possible=True,
                risk_level="MEDIUM"
            ),
            ComplianceControl(
                control_id="A.18.1",
                framework=ComplianceFramework.ISO27001,
                title="Compliance Monitoring",
                description="Compliance with legal and contractual requirements",
                control_family="Compliance",
                implementation_guidance="Implement compliance monitoring and reporting",
                testing_procedures=[
                    "Review compliance procedures",
                    "Test monitoring capabilities",
                    "Verify reporting mechanisms"
                ],
                evidence_requirements=[
                    EvidenceType.POLICY_DOCUMENT,
                    EvidenceType.AUDIT_TRAIL,
                    EvidenceType.LOG_DATA
                ],
                automation_possible=True,
                risk_level="HIGH"
            )
        ]
        
        # NIST CSF Controls
        nist_controls = [
            ComplianceControl(
                control_id="PR.DS-2",
                framework=ComplianceFramework.NIST_CSF,
                title="Data-in-transit Protection",
                description="Data-in-transit is protected",
                control_family="Protect",
                implementation_guidance="Implement encryption and secure protocols for data transmission",
                testing_procedures=[
                    "Test transmission encryption",
                    "Review protocol configurations",
                    "Verify certificate management"
                ],
                evidence_requirements=[
                    EvidenceType.CONFIGURATION,
                    EvidenceType.SECURITY_SCAN,
                    EvidenceType.LOG_DATA
                ],
                automation_possible=True,
                risk_level="CRITICAL"
            ),
            ComplianceControl(
                control_id="PR.AC-7",
                framework=ComplianceFramework.NIST_CSF,
                title="Network Segregation",
                description="Users, devices, and other assets are authenticated",
                control_family="Protect",
                implementation_guidance="Implement network segmentation and access controls",
                testing_procedures=[
                    "Test network segmentation",
                    "Review access controls",
                    "Verify authentication mechanisms"
                ],
                evidence_requirements=[
                    EvidenceType.CONFIGURATION,
                    EvidenceType.ACCESS_REVIEW,
                    EvidenceType.LOG_DATA
                ],
                automation_possible=True,
                risk_level="HIGH"
            ),
            ComplianceControl(
                control_id="DE.CM-1",
                framework=ComplianceFramework.NIST_CSF,
                title="Continuous Monitoring",
                description="The network is monitored to detect potential cybersecurity events",
                control_family="Detect",
                implementation_guidance="Implement continuous monitoring and detection capabilities",
                testing_procedures=[
                    "Test monitoring systems",
                    "Review detection capabilities",
                    "Verify alerting mechanisms"
                ],
                evidence_requirements=[
                    EvidenceType.CONFIGURATION,
                    EvidenceType.LOG_DATA,
                    EvidenceType.AUDIT_TRAIL
                ],
                automation_possible=True,
                risk_level="HIGH"
            )
        ]
        
        # Add all controls to database
        for control in soc2_controls + iso27001_controls + nist_controls:
            self.controls[control.control_id] = control
        
        logger.info(f"Initialized {len(self.controls)} compliance controls")
    
    def get_controls_by_framework(self, framework: ComplianceFramework) -> List[ComplianceControl]:
        """Get all controls for a specific framework"""
        return [control for control in self.controls.values() if control.framework == framework]
    
    def get_control(self, control_id: str) -> Optional[ComplianceControl]:
        """Get specific control by ID"""
        return self.controls.get(control_id)


class ComplianceAssessor:
    """Automated compliance assessment engine"""
    
    def __init__(self):
        self.control_db = ComplianceControlDatabase()
    
    async def assess_control(self, control_id: str, system_data: Dict[str, Any]) -> ControlAssessment:
        """Assess a specific compliance control"""
        control = self.control_db.get_control(control_id)
        if not control:
            raise ValueError(f"Control {control_id} not found")
        
        # Automated assessment logic based on control type
        if control_id == "CC6.1":  # Data Encryption
            return await self._assess_encryption_control(control, system_data)
        elif control_id == "CC6.7":  # Data Transmission
            return await self._assess_transmission_control(control, system_data)
        elif control_id == "A.10.1":  # Cryptographic Controls
            return await self._assess_cryptographic_control(control, system_data)
        elif control_id == "A.13.1":  # Network Security
            return await self._assess_network_security_control(control, system_data)
        elif control_id == "PR.DS-2":  # Data-in-transit Protection
            return await self._assess_data_transit_control(control, system_data)
        else:
            return await self._assess_generic_control(control, system_data)
    
    async def _assess_encryption_control(self, control: ComplianceControl, system_data: Dict[str, Any]) -> ControlAssessment:
        """Assess encryption implementation"""
        findings = []
        evidence = []
        compliance_score = 0.0
        
        # Check database encryption
        if system_data.get("database_encryption_enabled", False):
            compliance_score += 0.3
            evidence.append("Database encryption configuration verified")
        else:
            findings.append("Database encryption not enabled")
        
        # Check transmission encryption
        if system_data.get("tls_enabled", False):
            compliance_score += 0.3
            evidence.append("TLS encryption enabled for data transmission")
        else:
            findings.append("TLS encryption not properly configured")
        
        # Check encryption key management
        if system_data.get("key_management_implemented", False):
            compliance_score += 0.2
            evidence.append("Key management system implemented")
        else:
            findings.append("Key management system not implemented")
        
        # Check encryption standards
        encryption_standard = system_data.get("encryption_standard", "")
        if "AES-256" in encryption_standard or "ChaCha20" in encryption_standard:
            compliance_score += 0.2
            evidence.append(f"Strong encryption standard in use: {encryption_standard}")
        else:
            findings.append("Weak or unknown encryption standard")
        
        # Determine status
        if compliance_score >= 0.9:
            status = ControlStatus.COMPLIANT
        elif compliance_score >= 0.7:
            status = ControlStatus.PARTIALLY_COMPLIANT
        else:
            status = ControlStatus.NON_COMPLIANT
        
        remediation = []
        if compliance_score < 1.0:
            remediation.extend([
                "Enable database encryption if not implemented",
                "Implement proper TLS configuration",
                "Deploy key management system",
                "Upgrade to strong encryption standards"
            ])
        
        return ControlAssessment(
            control_id=control.control_id,
            framework=control.framework,
            status=status,
            compliance_score=compliance_score,
            assessment_date=datetime.utcnow(),
            assessor="automated_system",
            findings=findings,
            evidence_collected=evidence,
            remediation_required=remediation,
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            automated=True
        )
    
    async def _assess_transmission_control(self, control: ComplianceControl, system_data: Dict[str, Any]) -> ControlAssessment:
        """Assess data transmission security"""
        findings = []
        evidence = []
        compliance_score = 0.0
        
        # Check TLS configuration
        tls_version = system_data.get("tls_version", "")
        if "1.3" in tls_version:
            compliance_score += 0.4
            evidence.append("TLS 1.3 implemented")
        elif "1.2" in tls_version:
            compliance_score += 0.3
            evidence.append("TLS 1.2 implemented")
        else:
            findings.append("Outdated or no TLS version")
        
        # Check certificate management
        if system_data.get("certificate_management", False):
            compliance_score += 0.2
            evidence.append("Certificate management system in place")
        else:
            findings.append("No certificate management system")
        
        # Check network segmentation
        if system_data.get("network_segmentation", False):
            compliance_score += 0.2
            evidence.append("Network segmentation implemented")
        else:
            findings.append("Network segmentation not implemented")
        
        # Check transmission monitoring
        if system_data.get("transmission_monitoring", False):
            compliance_score += 0.2
            evidence.append("Transmission monitoring enabled")
        else:
            findings.append("No transmission monitoring")
        
        # Determine status
        if compliance_score >= 0.9:
            status = ControlStatus.COMPLIANT
        elif compliance_score >= 0.7:
            status = ControlStatus.PARTIALLY_COMPLIANT
        else:
            status = ControlStatus.NON_COMPLIANT
        
        remediation = []
        if compliance_score < 1.0:
            remediation.extend([
                "Upgrade to TLS 1.3",
                "Implement certificate management",
                "Deploy network segmentation",
                "Enable transmission monitoring"
            ])
        
        return ControlAssessment(
            control_id=control.control_id,
            framework=control.framework,
            status=status,
            compliance_score=compliance_score,
            assessment_date=datetime.utcnow(),
            assessor="automated_system",
            findings=findings,
            evidence_collected=evidence,
            remediation_required=remediation,
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            automated=True
        )
    
    async def _assess_cryptographic_control(self, control: ComplianceControl, system_data: Dict[str, Any]) -> ControlAssessment:
        """Assess cryptographic controls (ISO27001 A.10.1)"""
        # Similar implementation to encryption control but with ISO27001 specific requirements
        return await self._assess_encryption_control(control, system_data)
    
    async def _assess_network_security_control(self, control: ComplianceControl, system_data: Dict[str, Any]) -> ControlAssessment:
        """Assess network security management"""
        findings = []
        evidence = []
        compliance_score = 0.0
        
        # Check firewall configuration
        if system_data.get("firewall_configured", False):
            compliance_score += 0.25
            evidence.append("Firewall properly configured")
        else:
            findings.append("Firewall not properly configured")
        
        # Check network monitoring
        if system_data.get("network_monitoring", False):
            compliance_score += 0.25
            evidence.append("Network monitoring implemented")
        else:
            findings.append("Network monitoring not implemented")
        
        # Check intrusion detection
        if system_data.get("intrusion_detection", False):
            compliance_score += 0.25
            evidence.append("Intrusion detection system in place")
        else:
            findings.append("No intrusion detection system")
        
        # Check network segmentation
        if system_data.get("network_segmentation", False):
            compliance_score += 0.25
            evidence.append("Network segmentation implemented")
        else:
            findings.append("Network segmentation not implemented")
        
        # Determine status
        status = ControlStatus.COMPLIANT if compliance_score >= 0.8 else ControlStatus.PARTIALLY_COMPLIANT if compliance_score >= 0.6 else ControlStatus.NON_COMPLIANT
        
        return ControlAssessment(
            control_id=control.control_id,
            framework=control.framework,
            status=status,
            compliance_score=compliance_score,
            assessment_date=datetime.utcnow(),
            assessor="automated_system",
            findings=findings,
            evidence_collected=evidence,
            remediation_required=[],
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            automated=True
        )
    
    async def _assess_data_transit_control(self, control: ComplianceControl, system_data: Dict[str, Any]) -> ControlAssessment:
        """Assess data-in-transit protection (NIST)"""
        # Similar to transmission control assessment
        return await self._assess_transmission_control(control, system_data)
    
    async def _assess_generic_control(self, control: ComplianceControl, system_data: Dict[str, Any]) -> ControlAssessment:
        """Generic control assessment for manual review"""
        return ControlAssessment(
            control_id=control.control_id,
            framework=control.framework,
            status=ControlStatus.UNDER_REVIEW,
            compliance_score=0.5,
            assessment_date=datetime.utcnow(),
            assessor="automated_system",
            findings=["Manual assessment required"],
            evidence_collected=["System data collected for manual review"],
            remediation_required=["Complete manual assessment"],
            next_assessment_due=datetime.utcnow() + timedelta(days=30),
            automated=False
        )


class ComplianceReportGenerator:
    """Automated compliance report generation"""
    
    def __init__(self):
        self.assessor = ComplianceAssessor()
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        system_data: Dict[str, Any],
        reporting_period_days: int = 90
    ) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        
        report_id = f"{framework.value}_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=reporting_period_days)
        
        # Get controls for framework
        controls = self.assessor.control_db.get_controls_by_framework(framework)
        
        # Assess all controls
        assessments = []
        for control in controls:
            try:
                assessment = await self.assessor.assess_control(control.control_id, system_data)
                assessments.append(assessment)
            except Exception as e:
                logger.error(f"Failed to assess control {control.control_id}: {str(e)}")
        
        # Calculate compliance metrics
        total_controls = len(assessments)
        compliant_controls = sum(1 for a in assessments if a.status == ControlStatus.COMPLIANT)
        non_compliant_controls = sum(1 for a in assessments if a.status == ControlStatus.NON_COMPLIANT)
        
        overall_score = sum(a.compliance_score for a in assessments) / total_controls if total_controls > 0 else 0.0
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(framework, overall_score, assessments)
        
        # Generate remediation plan
        remediation_plan = self._generate_remediation_plan(assessments)
        
        return ComplianceReport(
            report_id=report_id,
            framework=framework,
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            overall_compliance_score=overall_score,
            total_controls=total_controls,
            compliant_controls=compliant_controls,
            non_compliant_controls=non_compliant_controls,
            control_assessments=assessments,
            executive_summary=executive_summary,
            remediation_plan=remediation_plan,
            generated_by="automated_compliance_system",
            generated_at=datetime.utcnow()
        )
    
    def _generate_executive_summary(
        self,
        framework: ComplianceFramework,
        overall_score: float,
        assessments: List[ControlAssessment]
    ) -> str:
        """Generate executive summary for compliance report"""
        
        compliance_percentage = overall_score * 100
        
        critical_findings = [
            a for a in assessments 
            if a.status == ControlStatus.NON_COMPLIANT and any("CRITICAL" in str(finding) for finding in a.findings)
        ]
        
        summary = f"""
        EXECUTIVE SUMMARY - {framework.value.upper()} COMPLIANCE ASSESSMENT
        
        Overall Compliance Score: {compliance_percentage:.1f}%
        
        The XORB Platform demonstrates {'strong' if compliance_percentage >= 80 else 'adequate' if compliance_percentage >= 60 else 'limited'} compliance with {framework.value.upper()} requirements.
        
        Key Findings:
        - {len([a for a in assessments if a.status == ControlStatus.COMPLIANT])} controls are fully compliant
        - {len([a for a in assessments if a.status == ControlStatus.PARTIALLY_COMPLIANT])} controls are partially compliant
        - {len([a for a in assessments if a.status == ControlStatus.NON_COMPLIANT])} controls require remediation
        
        {'Critical security gaps have been identified and require immediate attention.' if critical_findings else 'No critical security gaps identified.'}
        
        Automated assessment coverage: {sum(1 for a in assessments if a.automated)}/{len(assessments)} controls
        """
        
        return summary.strip()
    
    def _generate_remediation_plan(self, assessments: List[ControlAssessment]) -> List[str]:
        """Generate prioritized remediation plan"""
        
        remediation_items = []
        
        # High priority - non-compliant controls
        non_compliant = [a for a in assessments if a.status == ControlStatus.NON_COMPLIANT]
        for assessment in non_compliant:
            remediation_items.extend([
                f"HIGH PRIORITY - {assessment.control_id}: {item}" 
                for item in assessment.remediation_required
            ])
        
        # Medium priority - partially compliant controls
        partially_compliant = [a for a in assessments if a.status == ControlStatus.PARTIALLY_COMPLIANT]
        for assessment in partially_compliant:
            remediation_items.extend([
                f"MEDIUM PRIORITY - {assessment.control_id}: {item}" 
                for item in assessment.remediation_required
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in remediation_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        
        return unique_items[:20]  # Top 20 remediation items


class ComplianceReportingService:
    """Main compliance reporting service"""
    
    def __init__(self):
        self.report_generator = ComplianceReportGenerator()
        self.reports: Dict[str, ComplianceReport] = {}
    
    async def generate_report(
        self,
        framework: ComplianceFramework,
        system_data: Optional[Dict[str, Any]] = None
    ) -> ComplianceReport:
        """Generate compliance report for specified framework"""
        
        if system_data is None:
            system_data = await self._collect_system_data()
        
        report = await self.report_generator.generate_compliance_report(
            framework, system_data
        )
        
        # Store report
        self.reports[report.report_id] = report
        
        logger.info(
            f"Generated compliance report",
            framework=framework.value,
            report_id=report.report_id,
            compliance_score=report.overall_compliance_score
        )
        
        return report
    
    async def _collect_system_data(self) -> Dict[str, Any]:
        """Collect current system data for compliance assessment"""
        
        # This would integrate with actual system components
        # For now, return realistic mock data based on XORB implementation
        
        return {
            # Encryption and security
            "database_encryption_enabled": True,
            "tls_enabled": True,
            "tls_version": "1.3",
            "encryption_standard": "AES-256-GCM",
            "key_management_implemented": False,  # To be implemented with Vault
            
            # Network security
            "network_segmentation": True,
            "firewall_configured": True,
            "network_monitoring": True,
            "intrusion_detection": False,  # To be enhanced
            
            # Transmission security
            "certificate_management": True,
            "transmission_monitoring": True,
            
            # Access controls
            "mfa_enabled": True,
            "rbac_implemented": True,
            "access_logging": True,
            
            # Monitoring and logging
            "audit_logging": True,
            "security_monitoring": True,
            "incident_response": True,
            "log_retention": True,
            
            # Additional metadata
            "assessment_date": datetime.utcnow().isoformat(),
            "platform_version": "1.0.0",
            "environment": "production"
        }
    
    def get_report(self, report_id: str) -> Optional[ComplianceReport]:
        """Get specific compliance report"""
        return self.reports.get(report_id)
    
    def list_reports(self, framework: Optional[ComplianceFramework] = None) -> List[ComplianceReport]:
        """List compliance reports, optionally filtered by framework"""
        reports = list(self.reports.values())
        
        if framework:
            reports = [r for r in reports if r.framework == framework]
        
        return sorted(reports, key=lambda r: r.generated_at, reverse=True)
    
    async def export_report_json(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Export report as JSON"""
        report = self.get_report(report_id)
        if not report:
            return None
        
        return asdict(report)
    
    async def export_report_html(self, report_id: str) -> Optional[str]:
        """Export report as HTML"""
        report = self.get_report(report_id)
        if not report:
            return None
        
        # HTML template for compliance report
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report.framework.value.upper() }} Compliance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #2c3e50; color: white; padding: 20px; }
                .summary { background: #ecf0f1; padding: 20px; margin: 20px 0; }
                .control { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
                .compliant { border-left: 5px solid #27ae60; }
                .non-compliant { border-left: 5px solid #e74c3c; }
                .partial { border-left: 5px solid #f39c12; }
                .score { font-size: 2em; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report.framework.value.upper() }} Compliance Report</h1>
                <p>Generated: {{ report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
                <p>Report ID: {{ report.report_id }}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <div class="score">Overall Score: {{ (report.overall_compliance_score * 100)|round(1) }}%</div>
                <pre>{{ report.executive_summary }}</pre>
            </div>
            
            <h2>Control Assessments</h2>
            {% for assessment in report.control_assessments %}
            <div class="control {{ 'compliant' if assessment.status.value == 'compliant' else 'non-compliant' if assessment.status.value == 'non_compliant' else 'partial' }}">
                <h3>{{ assessment.control_id }} - {{ assessment.status.value.title() }}</h3>
                <p><strong>Score:</strong> {{ (assessment.compliance_score * 100)|round(1) }}%</p>
                <p><strong>Automated:</strong> {{ 'Yes' if assessment.automated else 'No' }}</p>
                {% if assessment.findings %}
                <p><strong>Findings:</strong></p>
                <ul>
                {% for finding in assessment.findings %}
                    <li>{{ finding }}</li>
                {% endfor %}
                </ul>
                {% endif %}
            </div>
            {% endfor %}
            
            {% if report.remediation_plan %}
            <h2>Remediation Plan</h2>
            <ol>
            {% for item in report.remediation_plan %}
                <li>{{ item }}</li>
            {% endfor %}
            </ol>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(report=report)


# Global service instance
_compliance_service: Optional[ComplianceReportingService] = None


async def get_compliance_service() -> ComplianceReportingService:
    """Get global compliance reporting service"""
    global _compliance_service
    
    if _compliance_service is None:
        _compliance_service = ComplianceReportingService()
    
    return _compliance_service


# Export main classes
__all__ = [
    "ComplianceReportingService",
    "ComplianceFramework",
    "ControlStatus",
    "ComplianceReport",
    "get_compliance_service"
]