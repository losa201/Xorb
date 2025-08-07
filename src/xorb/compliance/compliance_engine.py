"""
Advanced Compliance Reporting Automation Engine
Automated compliance assessments for GDPR, NIS2, SOC2, ISO 27001, and other frameworks
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiohttp
import yaml
from jinja2 import Template
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    GDPR = "gdpr"
    NIS2 = "nis2"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    BSI_GRUNDSCHUTZ = "bsi_grundschutz"
    NIST_CSF = "nist_csf"
    CIS_CONTROLS = "cis_controls"
    PCI_DSS = "pci_dss"


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    IN_PROGRESS = "in_progress"


class RiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ComplianceControl:
    """Individual compliance control"""
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    requirements: List[str]
    assessment_criteria: List[str]
    automated_checks: List[str] = field(default_factory=list)
    manual_checks: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Assessment results
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    score: float = 0.0
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence_collected: List[Dict[str, Any]] = field(default_factory=list)
    last_assessed: Optional[datetime] = None


@dataclass
class ComplianceAssessment:
    """Complete compliance assessment"""
    id: str
    framework: ComplianceFramework
    organization: str
    scope: str
    assessment_date: datetime
    assessor: str
    
    # Controls and results
    controls: List[ComplianceControl] = field(default_factory=list)
    overall_score: float = 0.0
    overall_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    
    # Summary statistics
    total_controls: int = 0
    compliant_controls: int = 0
    non_compliant_controls: int = 0
    partial_controls: int = 0
    
    # Risk assessment
    critical_findings: List[str] = field(default_factory=list)
    high_risk_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Remediation tracking
    remediation_plan: List[Dict[str, Any]] = field(default_factory=list)
    next_assessment_date: Optional[datetime] = None


class ComplianceEngine:
    """Advanced compliance automation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.frameworks = {}
        self.assessment_templates = {}
        self.automated_checks = {}
        self.evidence_collectors = {}
        
    async def initialize(self):
        """Initialize compliance engine"""
        logger.info("Initializing Compliance Engine...")
        
        # Load compliance frameworks
        await self._load_frameworks()
        
        # Initialize automated checks
        await self._initialize_automated_checks()
        
        # Load assessment templates
        await self._load_assessment_templates()
        
        # Initialize evidence collectors
        await self._initialize_evidence_collectors()
        
        logger.info("Compliance Engine initialized successfully")
        
    async def _load_frameworks(self):
        """Load compliance framework definitions"""
        frameworks_dir = Path(self.config.get("frameworks_dir", "./compliance_frameworks"))
        
        # GDPR Framework
        self.frameworks[ComplianceFramework.GDPR] = {
            "name": "General Data Protection Regulation",
            "version": "2018",
            "categories": [
                "Data Protection Principles",
                "Lawful Basis for Processing",
                "Individual Rights",
                "Data Protection by Design",
                "Data Breach Management",
                "International Transfers",
                "Governance and Accountability"
            ],
            "controls": self._get_gdpr_controls()
        }
        
        # NIS2 Framework
        self.frameworks[ComplianceFramework.NIS2] = {
            "name": "Network and Information Systems Directive 2",
            "version": "2022",
            "categories": [
                "Risk Management",
                "Incident Handling",
                "Business Continuity",
                "Supply Chain Security",
                "Network Security",
                "Vulnerability Management",
                "Access Control"
            ],
            "controls": self._get_nis2_controls()
        }
        
        # SOC 2 Framework
        self.frameworks[ComplianceFramework.SOC2] = {
            "name": "Service Organization Control 2",
            "version": "2017",
            "categories": [
                "Security",
                "Availability", 
                "Processing Integrity",
                "Confidentiality",
                "Privacy"
            ],
            "controls": self._get_soc2_controls()
        }
        
        # ISO 27001 Framework
        self.frameworks[ComplianceFramework.ISO27001] = {
            "name": "Information Security Management Systems",
            "version": "2022",
            "categories": [
                "Information Security Policies",
                "Organization of Information Security",
                "Human Resource Security",
                "Asset Management",
                "Access Control",
                "Cryptography",
                "Physical and Environmental Security",
                "Operations Security",
                "Communications Security",
                "System Acquisition",
                "Supplier Relationships",
                "Information Security Incident Management",
                "Business Continuity",
                "Compliance"
            ],
            "controls": self._get_iso27001_controls()
        }
        
    def _get_gdpr_controls(self) -> List[ComplianceControl]:
        """Get GDPR compliance controls"""
        return [
            ComplianceControl(
                id="GDPR-1.1",
                framework=ComplianceFramework.GDPR,
                title="Lawful Basis for Processing",
                description="Ensure all personal data processing has a valid lawful basis",
                category="Data Protection Principles",
                requirements=[
                    "Identify lawful basis for each processing activity",
                    "Document lawful basis in records of processing",
                    "Communicate lawful basis to data subjects"
                ],
                assessment_criteria=[
                    "Processing activities mapped to lawful bases",
                    "Legal basis documented in privacy notices",
                    "Consent mechanisms implemented where required"
                ],
                automated_checks=[
                    "privacy_policy_analysis",
                    "consent_mechanism_check",
                    "data_processing_audit"
                ],
                risk_level=RiskLevel.HIGH
            ),
            ComplianceControl(
                id="GDPR-2.1",
                framework=ComplianceFramework.GDPR,
                title="Data Subject Rights",
                description="Implement processes to handle data subject rights requests",
                category="Individual Rights",
                requirements=[
                    "Right to information and access",
                    "Right to rectification",
                    "Right to erasure (right to be forgotten)",
                    "Right to restrict processing",
                    "Right to data portability",
                    "Right to object"
                ],
                assessment_criteria=[
                    "Request handling procedures documented",
                    "Response times within legal requirements",
                    "Technical mechanisms for data extraction/deletion"
                ],
                automated_checks=[
                    "data_subject_request_workflow",
                    "data_deletion_capability",
                    "data_export_functionality"
                ],
                risk_level=RiskLevel.HIGH
            ),
            ComplianceControl(
                id="GDPR-3.1",
                framework=ComplianceFramework.GDPR,
                title="Data Protection Impact Assessment",
                description="Conduct DPIA for high-risk processing activities",
                category="Data Protection by Design",
                requirements=[
                    "Identify high-risk processing activities",
                    "Conduct systematic DPIA process",
                    "Implement risk mitigation measures",
                    "Consult with supervisory authority if required"
                ],
                assessment_criteria=[
                    "DPIA methodology documented",
                    "High-risk activities identified",
                    "Risk mitigation measures implemented"
                ],
                automated_checks=[
                    "dpia_completion_check",
                    "risk_assessment_validation",
                    "mitigation_measure_verification"
                ],
                risk_level=RiskLevel.MEDIUM
            )
        ]
        
    def _get_nis2_controls(self) -> List[ComplianceControl]:
        """Get NIS2 compliance controls"""
        return [
            ComplianceControl(
                id="NIS2-1.1",
                framework=ComplianceFramework.NIS2,
                title="Cybersecurity Risk Management",
                description="Implement comprehensive cybersecurity risk management",
                category="Risk Management",
                requirements=[
                    "Establish cybersecurity governance",
                    "Implement risk assessment processes",
                    "Define risk management policies",
                    "Regular risk reviews and updates"
                ],
                assessment_criteria=[
                    "Risk management framework documented",
                    "Regular risk assessments conducted",
                    "Risk treatment plans implemented"
                ],
                automated_checks=[
                    "risk_management_policy_check",
                    "risk_assessment_frequency",
                    "risk_register_maintenance"
                ],
                risk_level=RiskLevel.HIGH
            ),
            ComplianceControl(
                id="NIS2-2.1",
                framework=ComplianceFramework.NIS2,
                title="Incident Handling and Response",
                description="Establish incident handling and response capabilities",
                category="Incident Handling",
                requirements=[
                    "Incident response plan development",
                    "Incident detection and analysis",
                    "Incident containment and eradication",
                    "Incident recovery and lessons learned",
                    "Incident reporting to authorities"
                ],
                assessment_criteria=[
                    "Incident response plan documented and tested",
                    "Incident detection capabilities operational",
                    "Response team roles and responsibilities defined"
                ],
                automated_checks=[
                    "incident_response_plan_check",
                    "detection_system_status",
                    "response_team_readiness"
                ],
                risk_level=RiskLevel.CRITICAL
            )
        ]
        
    def _get_soc2_controls(self) -> List[ComplianceControl]:
        """Get SOC 2 compliance controls"""
        return [
            ComplianceControl(
                id="SOC2-CC1.1",
                framework=ComplianceFramework.SOC2,
                title="Control Environment - Integrity and Ethical Values",
                description="Demonstrate commitment to integrity and ethical values",
                category="Security",
                requirements=[
                    "Code of conduct established and communicated",
                    "Ethical values integrated into performance measures",
                    "Disciplinary actions for violations documented"
                ],
                assessment_criteria=[
                    "Code of conduct exists and is current",
                    "Ethics training provided to personnel",
                    "Violations tracked and addressed"
                ],
                automated_checks=[
                    "code_of_conduct_check",
                    "ethics_training_completion",
                    "violation_tracking_system"
                ],
                risk_level=RiskLevel.MEDIUM
            ),
            ComplianceControl(
                id="SOC2-CC6.1",
                framework=ComplianceFramework.SOC2,
                title="Logical and Physical Access Controls",
                description="Implement logical and physical access restrictions",
                category="Security",
                requirements=[
                    "Access control policies and procedures",
                    "User access provisioning and deprovisioning",
                    "Privileged access management",
                    "Physical access controls"
                ],
                assessment_criteria=[
                    "Access control procedures documented",
                    "Regular access reviews conducted",
                    "Segregation of duties implemented"
                ],
                automated_checks=[
                    "access_control_policy_check",
                    "access_review_frequency",
                    "privileged_access_monitoring"
                ],
                risk_level=RiskLevel.HIGH
            )
        ]
        
    def _get_iso27001_controls(self) -> List[ComplianceControl]:
        """Get ISO 27001 compliance controls"""
        return [
            ComplianceControl(
                id="ISO-A.5.1.1",
                framework=ComplianceFramework.ISO27001,
                title="Information Security Policy",
                description="Establish and maintain information security policy",
                category="Information Security Policies",
                requirements=[
                    "Information security policy documented",
                    "Policy approved by management",
                    "Policy communicated to all personnel",
                    "Regular policy reviews and updates"
                ],
                assessment_criteria=[
                    "Current information security policy exists",
                    "Policy approval documentation available",
                    "Evidence of policy communication"
                ],
                automated_checks=[
                    "security_policy_check",
                    "policy_approval_validation",
                    "policy_communication_tracking"
                ],
                risk_level=RiskLevel.MEDIUM
            ),
            ComplianceControl(
                id="ISO-A.8.1.1",
                framework=ComplianceFramework.ISO27001,
                title="Asset Inventory",
                description="Maintain accurate inventory of information assets",
                category="Asset Management",
                requirements=[
                    "Asset inventory maintained",
                    "Asset owners identified",
                    "Asset classification implemented",
                    "Asset handling procedures defined"
                ],
                assessment_criteria=[
                    "Complete asset inventory exists",
                    "Asset ownership documented",
                    "Classification scheme implemented"
                ],
                automated_checks=[
                    "asset_inventory_completeness",
                    "asset_ownership_tracking",
                    "asset_classification_check"
                ],
                risk_level=RiskLevel.HIGH
            )
        ]
        
    async def _initialize_automated_checks(self):
        """Initialize automated compliance checks"""
        self.automated_checks = {
            # GDPR checks
            "privacy_policy_analysis": self._check_privacy_policy,
            "consent_mechanism_check": self._check_consent_mechanisms,
            "data_processing_audit": self._check_data_processing,
            "data_subject_request_workflow": self._check_dsr_workflow,
            "data_deletion_capability": self._check_data_deletion,
            "data_export_functionality": self._check_data_export,
            "dpia_completion_check": self._check_dpia_completion,
            
            # NIS2 checks
            "risk_management_policy_check": self._check_risk_management_policy,
            "risk_assessment_frequency": self._check_risk_assessment_frequency,
            "incident_response_plan_check": self._check_incident_response_plan,
            "detection_system_status": self._check_detection_systems,
            
            # SOC2 checks  
            "code_of_conduct_check": self._check_code_of_conduct,
            "ethics_training_completion": self._check_ethics_training,
            "access_control_policy_check": self._check_access_control_policy,
            "access_review_frequency": self._check_access_reviews,
            
            # ISO 27001 checks
            "security_policy_check": self._check_security_policy,
            "policy_approval_validation": self._check_policy_approval,
            "asset_inventory_completeness": self._check_asset_inventory,
            "asset_ownership_tracking": self._check_asset_ownership
        }
        
    async def _load_assessment_templates(self):
        """Load assessment report templates"""
        templates_dir = Path(self.config.get("templates_dir", "./templates"))
        
        self.assessment_templates = {
            "executive_summary": """
# Compliance Assessment Executive Summary

**Organization:** {{ assessment.organization }}
**Framework:** {{ assessment.framework.value.upper() }}
**Assessment Date:** {{ assessment.assessment_date.strftime('%Y-%m-%d') }}
**Overall Score:** {{ "%.1f"|format(assessment.overall_score) }}%
**Status:** {{ assessment.overall_status.value.title() }}

## Key Findings

{% for finding in assessment.critical_findings %}
- **CRITICAL:** {{ finding }}
{% endfor %}

{% for finding in assessment.high_risk_findings %}
- **HIGH:** {{ finding }}
{% endfor %}

## Compliance Summary

- **Total Controls:** {{ assessment.total_controls }}
- **Compliant:** {{ assessment.compliant_controls }} ({{ "%.1f"|format((assessment.compliant_controls / assessment.total_controls * 100) if assessment.total_controls > 0 else 0) }}%)
- **Non-Compliant:** {{ assessment.non_compliant_controls }} ({{ "%.1f"|format((assessment.non_compliant_controls / assessment.total_controls * 100) if assessment.total_controls > 0 else 0) }}%)
- **Partially Compliant:** {{ assessment.partial_controls }} ({{ "%.1f"|format((assessment.partial_controls / assessment.total_controls * 100) if assessment.total_controls > 0 else 0) }}%)

## Next Steps

{% for recommendation in assessment.recommendations[:5] %}
{{ loop.index }}. {{ recommendation }}
{% endfor %}

**Next Assessment:** {{ assessment.next_assessment_date.strftime('%Y-%m-%d') if assessment.next_assessment_date else 'TBD' }}
            """,
            
            "detailed_report": """
# Detailed Compliance Assessment Report

**Framework:** {{ framework_info.name }}
**Version:** {{ framework_info.version }}
**Assessment Date:** {{ assessment.assessment_date.strftime('%Y-%m-%d') }}

{% for category in framework_info.categories %}
## {{ category }}

{% for control in assessment.controls %}
{% if control.category == category %}
### {{ control.id }}: {{ control.title }}

**Status:** {{ control.status.value.title() }}
**Score:** {{ "%.1f"|format(control.score) }}%
**Risk Level:** {{ control.risk_level.value.title() }}

**Description:** {{ control.description }}

**Requirements:**
{% for req in control.requirements %}
- {{ req }}
{% endfor %}

{% if control.findings %}
**Findings:**
{% for finding in control.findings %}
- {{ finding }}
{% endfor %}
{% endif %}

{% if control.recommendations %}
**Recommendations:**
{% for rec in control.recommendations %}
- {{ rec }}
{% endfor %}
{% endif %}

**Last Assessed:** {{ control.last_assessed.strftime('%Y-%m-%d %H:%M') if control.last_assessed else 'Not assessed' }}

---
{% endif %}
{% endfor %}
{% endfor %}
            """
        }
        
    async def _initialize_evidence_collectors(self):
        """Initialize evidence collection mechanisms"""
        self.evidence_collectors = {
            "system_configuration": self._collect_system_config,
            "access_logs": self._collect_access_logs,
            "policy_documents": self._collect_policy_documents,
            "training_records": self._collect_training_records,
            "incident_records": self._collect_incident_records,
            "vulnerability_scans": self._collect_vulnerability_scans,
            "network_topology": self._collect_network_topology,
            "data_flow_diagrams": self._collect_data_flows
        }
        
    async def conduct_assessment(self, framework: ComplianceFramework, 
                               organization: str, scope: str, 
                               assessor: str) -> ComplianceAssessment:
        """Conduct comprehensive compliance assessment"""
        logger.info(f"Starting {framework.value} assessment for {organization}")
        
        assessment = ComplianceAssessment(
            id=f"assessment_{framework.value}_{organization}_{datetime.now().strftime('%Y%m%d')}",
            framework=framework,
            organization=organization,
            scope=scope,
            assessment_date=datetime.now(),
            assessor=assessor
        )
        
        # Get framework controls
        framework_info = self.frameworks.get(framework)
        if not framework_info:
            raise ValueError(f"Framework {framework.value} not supported")
            
        controls = framework_info["controls"].copy()
        
        # Execute automated checks for each control
        for control in controls:
            logger.info(f"Assessing control {control.id}: {control.title}")
            
            # Run automated checks
            await self._assess_control(control)
            
            # Collect evidence
            await self._collect_control_evidence(control)
            
            # Update control status based on results
            self._calculate_control_score(control)
            
        assessment.controls = controls
        
        # Calculate overall assessment results
        self._calculate_assessment_results(assessment)
        
        # Generate recommendations
        assessment.recommendations = await self._generate_recommendations(assessment)
        
        # Set next assessment date
        assessment.next_assessment_date = self._calculate_next_assessment_date(framework)
        
        logger.info(f"Assessment completed. Overall score: {assessment.overall_score:.1f}%")
        
        return assessment
        
    async def _assess_control(self, control: ComplianceControl):
        """Assess individual compliance control"""
        control.last_assessed = datetime.now()
        control.findings = []
        control.recommendations = []
        
        # Run automated checks
        check_results = []
        for check_name in control.automated_checks:
            if check_name in self.automated_checks:
                try:
                    result = await self.automated_checks[check_name](control)
                    check_results.append(result)
                except Exception as e:
                    logger.error(f"Automated check {check_name} failed: {e}")
                    check_results.append({
                        "passed": False,
                        "message": f"Check failed: {str(e)}",
                        "evidence": []
                    })
                    
        # Analyze results
        passed_checks = sum(1 for result in check_results if result.get("passed", False))
        total_checks = len(check_results)
        
        if total_checks == 0:
            control.status = ComplianceStatus.NOT_ASSESSED
            control.score = 0.0
        elif passed_checks == total_checks:
            control.status = ComplianceStatus.COMPLIANT
            control.score = 100.0
        elif passed_checks == 0:
            control.status = ComplianceStatus.NON_COMPLIANT
            control.score = 0.0
        else:
            control.status = ComplianceStatus.PARTIALLY_COMPLIANT
            control.score = (passed_checks / total_checks) * 100
            
        # Collect findings and recommendations
        for result in check_results:
            if not result.get("passed", False):
                control.findings.append(result.get("message", "Check failed"))
            if "recommendations" in result:
                control.recommendations.extend(result["recommendations"])
                
    async def _collect_control_evidence(self, control: ComplianceControl):
        """Collect evidence for compliance control"""
        control.evidence_collected = []
        
        for evidence_type in control.evidence_requirements:
            if evidence_type in self.evidence_collectors:
                try:
                    evidence = await self.evidence_collectors[evidence_type](control)
                    control.evidence_collected.append({
                        "type": evidence_type,
                        "collected_at": datetime.now().isoformat(),
                        "data": evidence
                    })
                except Exception as e:
                    logger.error(f"Evidence collection for {evidence_type} failed: {e}")
                    
    def _calculate_control_score(self, control: ComplianceControl):
        """Calculate control compliance score"""
        # Score already calculated in _assess_control
        pass
        
    def _calculate_assessment_results(self, assessment: ComplianceAssessment):
        """Calculate overall assessment results"""
        if not assessment.controls:
            return
            
        assessment.total_controls = len(assessment.controls)
        assessment.compliant_controls = sum(
            1 for c in assessment.controls 
            if c.status == ComplianceStatus.COMPLIANT
        )
        assessment.non_compliant_controls = sum(
            1 for c in assessment.controls 
            if c.status == ComplianceStatus.NON_COMPLIANT
        )
        assessment.partial_controls = sum(
            1 for c in assessment.controls 
            if c.status == ComplianceStatus.PARTIALLY_COMPLIANT
        )
        
        # Calculate overall score
        total_score = sum(c.score for c in assessment.controls)
        assessment.overall_score = total_score / assessment.total_controls if assessment.total_controls > 0 else 0
        
        # Determine overall status
        if assessment.overall_score >= 95:
            assessment.overall_status = ComplianceStatus.COMPLIANT
        elif assessment.overall_score >= 70:
            assessment.overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            assessment.overall_status = ComplianceStatus.NON_COMPLIANT
            
        # Collect critical and high-risk findings
        for control in assessment.controls:
            if control.status != ComplianceStatus.COMPLIANT:
                for finding in control.findings:
                    if control.risk_level == RiskLevel.CRITICAL:
                        assessment.critical_findings.append(f"{control.id}: {finding}")
                    elif control.risk_level == RiskLevel.HIGH:
                        assessment.high_risk_findings.append(f"{control.id}: {finding}")
                        
    async def _generate_recommendations(self, assessment: ComplianceAssessment) -> List[str]:
        """Generate assessment recommendations"""
        recommendations = []
        
        # Framework-specific recommendations
        if assessment.framework == ComplianceFramework.GDPR:
            if assessment.overall_score < 80:
                recommendations.append("Implement comprehensive data protection impact assessments")
                recommendations.append("Establish clear data subject rights procedures")
                recommendations.append("Review and update privacy policies and notices")
                
        elif assessment.framework == ComplianceFramework.NIS2:
            if assessment.overall_score < 85:
                recommendations.append("Enhance cybersecurity risk management processes")
                recommendations.append("Improve incident detection and response capabilities")
                recommendations.append("Strengthen supply chain security measures")
                
        # General recommendations based on control failures
        failed_controls = [c for c in assessment.controls if c.status == ComplianceStatus.NON_COMPLIANT]
        if len(failed_controls) > 3:
            recommendations.append("Conduct comprehensive compliance gap analysis")
            recommendations.append("Develop detailed remediation roadmap with timelines")
            recommendations.append("Implement compliance monitoring and reporting processes")
            
        return recommendations[:10]  # Limit to top 10 recommendations
        
    def _calculate_next_assessment_date(self, framework: ComplianceFramework) -> datetime:
        """Calculate next assessment date based on framework requirements"""
        assessment_cycles = {
            ComplianceFramework.GDPR: 365,  # Annual
            ComplianceFramework.NIS2: 365,  # Annual
            ComplianceFramework.SOC2: 365,  # Annual
            ComplianceFramework.ISO27001: 365  # Annual
        }
        
        days = assessment_cycles.get(framework, 365)
        return datetime.now() + timedelta(days=days)
        
    async def generate_report(self, assessment: ComplianceAssessment, 
                            report_type: str = "detailed") -> str:
        """Generate compliance assessment report"""
        template_content = self.assessment_templates.get(report_type, self.assessment_templates["detailed_report"])
        template = Template(template_content)
        
        framework_info = self.frameworks.get(assessment.framework, {})
        
        return template.render(
            assessment=assessment,
            framework_info=framework_info
        )
        
    async def export_assessment_data(self, assessment: ComplianceAssessment, 
                                   format: str = "json") -> Any:
        """Export assessment data in various formats"""
        if format == "json":
            return json.dumps(asdict(assessment), default=str, indent=2)
        elif format == "csv":
            # Convert controls to DataFrame
            controls_data = []
            for control in assessment.controls:
                controls_data.append({
                    "ID": control.id,
                    "Title": control.title,
                    "Category": control.category,
                    "Status": control.status.value,
                    "Score": control.score,
                    "Risk Level": control.risk_level.value,
                    "Findings Count": len(control.findings),
                    "Last Assessed": control.last_assessed.isoformat() if control.last_assessed else ""
                })
            
            df = pd.DataFrame(controls_data)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    # Automated check implementations
    async def _check_privacy_policy(self, control: ComplianceControl) -> Dict[str, Any]:
        """Check privacy policy compliance"""
        return {
            "passed": True,  # Placeholder
            "message": "Privacy policy analysis passed",
            "evidence": ["privacy_policy_document.pdf"],
            "recommendations": []
        }
        
    async def _check_consent_mechanisms(self, control: ComplianceControl) -> Dict[str, Any]:
        """Check consent mechanism implementation"""
        return {
            "passed": False,
            "message": "Consent mechanisms not fully implemented",
            "evidence": [],
            "recommendations": ["Implement granular consent options", "Add consent withdrawal mechanisms"]
        }
        
    async def _check_data_processing(self, control: ComplianceControl) -> Dict[str, Any]:
        """Audit data processing activities"""
        return {
            "passed": True,
            "message": "Data processing activities documented",
            "evidence": ["data_processing_register.xlsx"],
            "recommendations": []
        }
        
    # Additional check methods would be implemented here...
    async def _check_dsr_workflow(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "DSR workflow operational", "evidence": []}
        
    async def _check_data_deletion(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": False, "message": "Data deletion capabilities incomplete", "recommendations": ["Implement automated data deletion"]}
        
    async def _check_data_export(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Data export functionality available", "evidence": []}
        
    async def _check_dpia_completion(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": False, "message": "DPIA not completed for high-risk processing", "recommendations": ["Complete DPIA for all high-risk activities"]}
        
    async def _check_risk_management_policy(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Risk management policy documented", "evidence": ["risk_management_policy.pdf"]}
        
    async def _check_risk_assessment_frequency(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": False, "message": "Risk assessments not conducted regularly", "recommendations": ["Establish quarterly risk assessment schedule"]}
        
    async def _check_incident_response_plan(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Incident response plan documented and tested", "evidence": ["ir_plan.pdf", "tabletop_exercise_2024.pdf"]}
        
    async def _check_detection_systems(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Detection systems operational", "evidence": ["siem_dashboard.png"]}
        
    async def _check_code_of_conduct(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Code of conduct established", "evidence": ["code_of_conduct.pdf"]}
        
    async def _check_ethics_training(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": False, "message": "Ethics training completion rate below 90%", "recommendations": ["Increase training completion rates"]}
        
    async def _check_access_control_policy(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Access control policy documented", "evidence": ["access_control_policy.pdf"]}
        
    async def _check_access_reviews(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": False, "message": "Access reviews not conducted quarterly", "recommendations": ["Implement quarterly access reviews"]}
        
    async def _check_security_policy(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Information security policy current", "evidence": ["security_policy_v2024.pdf"]}
        
    async def _check_policy_approval(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Policy approval documentation available", "evidence": ["policy_approval_2024.pdf"]}
        
    async def _check_asset_inventory(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": False, "message": "Asset inventory incomplete", "recommendations": ["Complete asset discovery and inventory update"]}
        
    async def _check_asset_ownership(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"passed": True, "message": "Asset ownership documented", "evidence": ["asset_ownership_matrix.xlsx"]}
        
    # Evidence collection implementations
    async def _collect_system_config(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"config_files": [], "settings": {}}
        
    async def _collect_access_logs(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"log_entries": [], "summary": {}}
        
    async def _collect_policy_documents(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"documents": []}
        
    async def _collect_training_records(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"training_completion": {}}
        
    async def _collect_incident_records(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"incidents": []}
        
    async def _collect_vulnerability_scans(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"scan_results": []}
        
    async def _collect_network_topology(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"topology_diagram": ""}
        
    async def _collect_data_flows(self, control: ComplianceControl) -> Dict[str, Any]:
        return {"data_flows": []}


# Factory function
def create_compliance_engine(config: Dict[str, Any]) -> ComplianceEngine:
    """Create and configure compliance engine"""
    default_config = {
        "frameworks_dir": "./compliance_frameworks",
        "templates_dir": "./templates",
        "evidence_dir": "./evidence",
        "reports_dir": "./reports"
    }
    
    final_config = {**default_config, **config}
    return ComplianceEngine(final_config)