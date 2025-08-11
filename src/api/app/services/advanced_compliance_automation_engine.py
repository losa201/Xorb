"""
Advanced Compliance Automation Engine
Sophisticated compliance management and automation for multiple regulatory frameworks
"""

import asyncio
import json
import logging
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available for compliance reporting")

from .interfaces import ComplianceService
from .base_service import XORBService, ServiceType
from ..domain.entities import User, Organization


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    GDPR = "gdpr"
    NIST_CSF = "nist_csf"
    SOC2 = "soc2"
    FISMA = "fisma"
    COBIT = "cobit"
    COSO = "coso"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    REMEDIATION_REQUIRED = "remediation_required"


class ControlSeverity(Enum):
    """Control importance/severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class ComplianceControl:
    """Individual compliance control"""
    control_id: str
    framework: ComplianceFramework
    name: str
    description: str
    category: str
    severity: ControlSeverity
    requirements: List[str]
    test_procedures: List[str]
    evidence_requirements: List[str]
    automated_checks: List[str] = field(default_factory=list)
    manual_verification: bool = False
    frequency: str = "annual"  # daily, weekly, monthly, quarterly, annual
    responsible_party: str = "security_team"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control_id,
            "framework": self.framework.value,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "severity": self.severity.value,
            "requirements": self.requirements,
            "test_procedures": self.test_procedures,
            "evidence_requirements": self.evidence_requirements,
            "automated_checks": self.automated_checks,
            "manual_verification": self.manual_verification,
            "frequency": self.frequency,
            "responsible_party": self.responsible_party
        }


@dataclass
class ComplianceEvidence:
    """Evidence for compliance control"""
    evidence_id: str
    control_id: str
    evidence_type: str  # document, screenshot, log, configuration, test_result
    description: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=datetime.utcnow)
    collected_by: str = "automated"
    validity_period: int = 365  # days
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "control_id": self.control_id,
            "evidence_type": self.evidence_type,
            "description": self.description,
            "file_path": self.file_path,
            "content": self.content,
            "metadata": self.metadata,
            "collected_at": self.collected_at.isoformat(),
            "collected_by": self.collected_by,
            "validity_period": self.validity_period
        }


@dataclass
class ComplianceAssessment:
    """Assessment result for a compliance control"""
    assessment_id: str
    control_id: str
    status: ComplianceStatus
    score: float  # 0.0 - 1.0
    findings: List[str]
    evidence: List[ComplianceEvidence]
    gaps: List[str]
    remediation_actions: List[str]
    assessor: str
    assessment_date: datetime = field(default_factory=datetime.utcnow)
    next_assessment_due: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "control_id": self.control_id,
            "status": self.status.value,
            "score": self.score,
            "findings": self.findings,
            "evidence": [e.to_dict() for e in self.evidence],
            "gaps": self.gaps,
            "remediation_actions": self.remediation_actions,
            "assessor": self.assessor,
            "assessment_date": self.assessment_date.isoformat(),
            "next_assessment_due": self.next_assessment_due.isoformat() if self.next_assessment_due else None
        }


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    framework: ComplianceFramework
    organization: str
    report_type: str  # assessment, gap_analysis, remediation_plan, certification
    overall_score: float
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    partially_compliant_controls: int
    assessments: List[ComplianceAssessment]
    executive_summary: str
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "framework": self.framework.value,
            "organization": self.organization,
            "report_type": self.report_type,
            "overall_score": self.overall_score,
            "total_controls": self.total_controls,
            "compliant_controls": self.compliant_controls,
            "non_compliant_controls": self.non_compliant_controls,
            "partially_compliant_controls": self.partially_compliant_controls,
            "assessments": [a.to_dict() for a in self.assessments],
            "executive_summary": self.executive_summary,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None
        }


class ComplianceFrameworkDatabase:
    """Database of compliance frameworks and their controls"""
    
    def __init__(self):
        self.frameworks = self._initialize_frameworks()
    
    def _initialize_frameworks(self) -> Dict[ComplianceFramework, List[ComplianceControl]]:
        """Initialize compliance frameworks with their controls"""
        frameworks = {}
        
        # PCI-DSS Controls
        frameworks[ComplianceFramework.PCI_DSS] = [
            ComplianceControl(
                control_id="PCI-1.1",
                framework=ComplianceFramework.PCI_DSS,
                name="Firewall Configuration Standards",
                description="Establish and implement firewall and router configuration standards",
                category="Network Security",
                severity=ControlSeverity.CRITICAL,
                requirements=[
                    "Document firewall and router configuration standards",
                    "Implement configuration standards on all firewalls and routers",
                    "Review configuration standards annually"
                ],
                test_procedures=[
                    "Review firewall configuration documentation",
                    "Compare actual configurations to standards",
                    "Verify annual review process"
                ],
                evidence_requirements=[
                    "Firewall configuration documentation",
                    "Router configuration files",
                    "Annual review records"
                ],
                automated_checks=["firewall_config_scan", "router_config_audit"],
                frequency="quarterly"
            ),
            ComplianceControl(
                control_id="PCI-2.1",
                framework=ComplianceFramework.PCI_DSS,
                name="Vendor Default Settings",
                description="Always change vendor-supplied defaults and remove unnecessary default accounts",
                category="System Configuration",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Change all vendor-supplied defaults",
                    "Remove unnecessary default accounts",
                    "Document configuration changes"
                ],
                test_procedures=[
                    "Scan for default accounts",
                    "Review system configurations",
                    "Verify password changes"
                ],
                evidence_requirements=[
                    "Default account inventory",
                    "Configuration change logs",
                    "Password policy documentation"
                ],
                automated_checks=["default_account_scan", "password_policy_check"]
            ),
            ComplianceControl(
                control_id="PCI-3.1",
                framework=ComplianceFramework.PCI_DSS,
                name="Data Retention Policy",
                description="Keep cardholder data storage to minimum required",
                category="Data Protection",
                severity=ControlSeverity.CRITICAL,
                requirements=[
                    "Implement data retention policy",
                    "Identify all cardholder data locations",
                    "Securely delete data when no longer needed"
                ],
                test_procedures=[
                    "Review data retention policy",
                    "Inventory cardholder data locations",
                    "Verify secure deletion procedures"
                ],
                evidence_requirements=[
                    "Data retention policy document",
                    "Data inventory records",
                    "Secure deletion logs"
                ],
                automated_checks=["data_discovery_scan", "retention_policy_check"]
            ),
            ComplianceControl(
                control_id="PCI-8.1",
                framework=ComplianceFramework.PCI_DSS,
                name="User Identification",
                description="Define and implement policies for proper user identification management",
                category="Access Control",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Assign unique ID to each user",
                    "Implement user authentication controls",
                    "Manage user accounts lifecycle"
                ],
                test_procedures=[
                    "Review user account policies",
                    "Test user authentication mechanisms",
                    "Verify account management procedures"
                ],
                evidence_requirements=[
                    "User account policies",
                    "Authentication system logs",
                    "Account lifecycle documentation"
                ],
                automated_checks=["user_account_audit", "authentication_test"]
            )
        ]
        
        # HIPAA Controls
        frameworks[ComplianceFramework.HIPAA] = [
            ComplianceControl(
                control_id="HIPAA-164.308(a)(1)",
                framework=ComplianceFramework.HIPAA,
                name="Security Officer",
                description="Assign security responsibilities to ensure compliance with HIPAA",
                category="Administrative Safeguards",
                severity=ControlSeverity.CRITICAL,
                requirements=[
                    "Designate a security officer",
                    "Assign security responsibilities",
                    "Document security procedures"
                ],
                test_procedures=[
                    "Review security officer assignment",
                    "Verify security responsibilities documentation",
                    "Check security procedure implementation"
                ],
                evidence_requirements=[
                    "Security officer appointment letter",
                    "Security responsibilities matrix",
                    "Security procedures documentation"
                ],
                manual_verification=True,
                frequency="annual"
            ),
            ComplianceControl(
                control_id="HIPAA-164.312(a)(1)",
                framework=ComplianceFramework.HIPAA,
                name="Access Control",
                description="Implement technical safeguards to allow access only to authorized persons",
                category="Technical Safeguards",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Implement access control systems",
                    "Assign unique user identification",
                    "Implement automatic logoff"
                ],
                test_procedures=[
                    "Test access control mechanisms",
                    "Review user access rights",
                    "Verify automatic logoff functionality"
                ],
                evidence_requirements=[
                    "Access control system documentation",
                    "User access reports",
                    "Logoff configuration settings"
                ],
                automated_checks=["access_control_test", "user_rights_audit", "logoff_test"]
            ),
            ComplianceControl(
                control_id="HIPAA-164.312(e)(1)",
                framework=ComplianceFramework.HIPAA,
                name="Transmission Security",
                description="Implement technical safeguards to guard against unauthorized access to ePHI transmitted over networks",
                category="Technical Safeguards",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Implement transmission security controls",
                    "Use encryption for ePHI transmission",
                    "Monitor network transmissions"
                ],
                test_procedures=[
                    "Test transmission encryption",
                    "Review network security controls",
                    "Verify transmission monitoring"
                ],
                evidence_requirements=[
                    "Encryption implementation documentation",
                    "Network security configuration",
                    "Transmission monitoring logs"
                ],
                automated_checks=["encryption_test", "network_security_scan", "transmission_monitor"]
            )
        ]
        
        # SOX Controls
        frameworks[ComplianceFramework.SOX] = [
            ComplianceControl(
                control_id="SOX-302",
                framework=ComplianceFramework.SOX,
                name="Financial Reporting Controls",
                description="Establish and maintain disclosure controls and procedures",
                category="Financial Reporting",
                severity=ControlSeverity.CRITICAL,
                requirements=[
                    "Design effective disclosure controls",
                    "Evaluate effectiveness quarterly",
                    "Maintain supporting documentation"
                ],
                test_procedures=[
                    "Review disclosure control design",
                    "Test control effectiveness",
                    "Verify quarterly evaluations"
                ],
                evidence_requirements=[
                    "Control design documentation",
                    "Effectiveness testing reports",
                    "Quarterly evaluation records"
                ],
                manual_verification=True,
                frequency="quarterly"
            ),
            ComplianceControl(
                control_id="SOX-404",
                framework=ComplianceFramework.SOX,
                name="Internal Control Assessment",
                description="Assess and report on internal control over financial reporting",
                category="Internal Controls",
                severity=ControlSeverity.CRITICAL,
                requirements=[
                    "Document internal control framework",
                    "Test control effectiveness annually",
                    "Report control deficiencies"
                ],
                test_procedures=[
                    "Review internal control documentation",
                    "Perform control testing",
                    "Evaluate control deficiencies"
                ],
                evidence_requirements=[
                    "Internal control documentation",
                    "Control testing workpapers",
                    "Deficiency reports"
                ],
                manual_verification=True,
                frequency="annual"
            )
        ]
        
        # ISO 27001 Controls
        frameworks[ComplianceFramework.ISO_27001] = [
            ComplianceControl(
                control_id="ISO-A.5.1.1",
                framework=ComplianceFramework.ISO_27001,
                name="Information Security Policies",
                description="A set of policies for information security shall be defined, approved by management",
                category="Security Policies",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Define information security policies",
                    "Obtain management approval",
                    "Communicate to all personnel"
                ],
                test_procedures=[
                    "Review security policy documentation",
                    "Verify management approval",
                    "Check communication records"
                ],
                evidence_requirements=[
                    "Security policy documents",
                    "Management approval records",
                    "Communication tracking"
                ],
                automated_checks=["policy_review_audit"],
                frequency="annual"
            ),
            ComplianceControl(
                control_id="ISO-A.9.1.1",
                framework=ComplianceFramework.ISO_27001,
                name="Access Control Policy",
                description="An access control policy shall be established, documented and reviewed",
                category="Access Control",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Establish access control policy",
                    "Document access procedures",
                    "Review policy regularly"
                ],
                test_procedures=[
                    "Review access control policy",
                    "Test access procedures",
                    "Verify regular reviews"
                ],
                evidence_requirements=[
                    "Access control policy",
                    "Access procedure documentation",
                    "Policy review records"
                ],
                automated_checks=["access_policy_audit", "access_procedure_test"]
            )
        ]
        
        # GDPR Controls
        frameworks[ComplianceFramework.GDPR] = [
            ComplianceControl(
                control_id="GDPR-Art.5",
                framework=ComplianceFramework.GDPR,
                name="Principles for Processing Personal Data",
                description="Personal data shall be processed lawfully, fairly and transparently",
                category="Data Processing",
                severity=ControlSeverity.CRITICAL,
                requirements=[
                    "Ensure lawful basis for processing",
                    "Implement transparency measures",
                    "Minimize data processing"
                ],
                test_procedures=[
                    "Review lawful basis documentation",
                    "Test transparency mechanisms",
                    "Verify data minimization"
                ],
                evidence_requirements=[
                    "Lawful basis documentation",
                    "Privacy notices",
                    "Data minimization evidence"
                ],
                automated_checks=["data_processing_audit", "privacy_notice_check"]
            ),
            ComplianceControl(
                control_id="GDPR-Art.32",
                framework=ComplianceFramework.GDPR,
                name="Security of Processing",
                description="Implement appropriate technical and organizational measures",
                category="Data Security",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Implement encryption",
                    "Ensure system confidentiality",
                    "Test security measures regularly"
                ],
                test_procedures=[
                    "Test encryption implementation",
                    "Verify confidentiality controls",
                    "Review security testing"
                ],
                evidence_requirements=[
                    "Encryption documentation",
                    "Confidentiality controls evidence",
                    "Security testing reports"
                ],
                automated_checks=["encryption_audit", "confidentiality_test", "security_scan"]
            )
        ]
        
        # NIST CSF Controls
        frameworks[ComplianceFramework.NIST_CSF] = [
            ComplianceControl(
                control_id="NIST-ID.AM-1",
                framework=ComplianceFramework.NIST_CSF,
                name="Asset Management",
                description="Physical devices and systems within the organization are inventoried",
                category="Identify",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Maintain asset inventory",
                    "Document asset owners",
                    "Update inventory regularly"
                ],
                test_procedures=[
                    "Review asset inventory",
                    "Verify asset ownership",
                    "Check inventory updates"
                ],
                evidence_requirements=[
                    "Asset inventory database",
                    "Asset ownership documentation",
                    "Inventory update logs"
                ],
                automated_checks=["asset_discovery_scan", "inventory_update_check"]
            ),
            ComplianceControl(
                control_id="NIST-PR.AC-1",
                framework=ComplianceFramework.NIST_CSF,
                name="Identity Management",
                description="Identities and credentials are issued, managed, verified, revoked",
                category="Protect",
                severity=ControlSeverity.HIGH,
                requirements=[
                    "Implement identity management system",
                    "Manage credential lifecycle",
                    "Verify identity authenticity"
                ],
                test_procedures=[
                    "Test identity management system",
                    "Review credential procedures",
                    "Verify identity verification"
                ],
                evidence_requirements=[
                    "Identity management documentation",
                    "Credential lifecycle procedures",
                    "Identity verification logs"
                ],
                automated_checks=["identity_system_test", "credential_audit", "verification_check"]
            )
        ]
        
        return frameworks
    
    def get_framework_controls(self, framework: ComplianceFramework) -> List[ComplianceControl]:
        """Get all controls for a specific framework"""
        return self.frameworks.get(framework, [])
    
    def get_control(self, control_id: str) -> Optional[ComplianceControl]:
        """Get specific control by ID"""
        for controls in self.frameworks.values():
            for control in controls:
                if control.control_id == control_id:
                    return control
        return None
    
    def get_controls_by_category(self, framework: ComplianceFramework, category: str) -> List[ComplianceControl]:
        """Get controls by category within a framework"""
        controls = self.frameworks.get(framework, [])
        return [control for control in controls if control.category.lower() == category.lower()]


class ComplianceAutomationEngine:
    """Core engine for compliance automation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.framework_db = ComplianceFrameworkDatabase()
        self.evidence_store: Dict[str, List[ComplianceEvidence]] = {}
        self.assessment_history: Dict[str, List[ComplianceAssessment]] = {}
    
    async def automated_control_assessment(
        self,
        control: ComplianceControl,
        system_data: Dict[str, Any]
    ) -> ComplianceAssessment:
        """Perform automated assessment of a compliance control"""
        try:
            assessment_id = str(uuid.uuid4())
            findings = []
            evidence = []
            gaps = []
            remediation_actions = []
            
            # Perform automated checks if available
            if control.automated_checks:
                for check in control.automated_checks:
                    check_result = await self._execute_automated_check(check, system_data)
                    
                    if check_result["passed"]:
                        findings.append(f"Automated check '{check}' passed: {check_result['details']}")
                        # Create evidence
                        evidence.append(ComplianceEvidence(
                            evidence_id=str(uuid.uuid4()),
                            control_id=control.control_id,
                            evidence_type="test_result",
                            description=f"Automated check result for {check}",
                            content=json.dumps(check_result),
                            metadata={"check_type": check, "automated": True}
                        ))
                    else:
                        gaps.append(f"Automated check '{check}' failed: {check_result['details']}")
                        remediation_actions.append(check_result.get("remediation", f"Fix issues identified in {check}"))
            
            # Calculate compliance score
            if control.automated_checks:
                passed_checks = len([f for f in findings if "passed" in f])
                total_checks = len(control.automated_checks)
                score = passed_checks / total_checks if total_checks > 0 else 0.0
            else:
                # For manual controls, use a basic scoring method
                score = 0.5  # Partial compliance assumed for manual verification
                findings.append("Manual verification required - automated assessment not available")
            
            # Determine status
            if score >= 0.9:
                status = ComplianceStatus.COMPLIANT
            elif score >= 0.7:
                status = ComplianceStatus.PARTIALLY_COMPLIANT
            elif score > 0:
                status = ComplianceStatus.REMEDIATION_REQUIRED
            else:
                status = ComplianceStatus.NON_COMPLIANT
            
            # Calculate next assessment due date
            frequency_days = {
                "daily": 1,
                "weekly": 7,
                "monthly": 30,
                "quarterly": 90,
                "annual": 365
            }
            days_to_add = frequency_days.get(control.frequency, 365)
            next_assessment_due = datetime.utcnow() + timedelta(days=days_to_add)
            
            assessment = ComplianceAssessment(
                assessment_id=assessment_id,
                control_id=control.control_id,
                status=status,
                score=score,
                findings=findings,
                evidence=evidence,
                gaps=gaps,
                remediation_actions=remediation_actions,
                assessor="automated_system",
                next_assessment_due=next_assessment_due
            )
            
            # Store evidence and assessment
            if control.control_id not in self.evidence_store:
                self.evidence_store[control.control_id] = []
            self.evidence_store[control.control_id].extend(evidence)
            
            if control.control_id not in self.assessment_history:
                self.assessment_history[control.control_id] = []
            self.assessment_history[control.control_id].append(assessment)
            
            self.logger.info(f"Completed automated assessment for control {control.control_id}: {status.value} (score: {score:.2f})")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in automated control assessment: {str(e)}")
            # Return failed assessment
            return ComplianceAssessment(
                assessment_id=str(uuid.uuid4()),
                control_id=control.control_id,
                status=ComplianceStatus.NOT_ASSESSED,
                score=0.0,
                findings=[f"Assessment failed: {str(e)}"],
                evidence=[],
                gaps=["Assessment could not be completed"],
                remediation_actions=["Resolve assessment errors and retry"],
                assessor="automated_system"
            )
    
    async def _execute_automated_check(
        self,
        check_name: str,
        system_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an automated compliance check"""
        try:
            # Simulate automated check execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            check_results = {
                "firewall_config_scan": self._check_firewall_config(system_data),
                "router_config_audit": self._check_router_config(system_data),
                "default_account_scan": self._check_default_accounts(system_data),
                "password_policy_check": self._check_password_policy(system_data),
                "data_discovery_scan": self._check_data_discovery(system_data),
                "retention_policy_check": self._check_retention_policy(system_data),
                "user_account_audit": self._check_user_accounts(system_data),
                "authentication_test": self._check_authentication(system_data),
                "access_control_test": self._check_access_control(system_data),
                "user_rights_audit": self._check_user_rights(system_data),
                "logoff_test": self._check_automatic_logoff(system_data),
                "encryption_test": self._check_encryption(system_data),
                "network_security_scan": self._check_network_security(system_data),
                "transmission_monitor": self._check_transmission_security(system_data),
                "policy_review_audit": self._check_policy_review(system_data),
                "access_policy_audit": self._check_access_policy(system_data),
                "access_procedure_test": self._check_access_procedures(system_data),
                "data_processing_audit": self._check_data_processing(system_data),
                "privacy_notice_check": self._check_privacy_notices(system_data),
                "encryption_audit": self._check_encryption_implementation(system_data),
                "confidentiality_test": self._check_confidentiality_controls(system_data),
                "security_scan": self._check_security_measures(system_data),
                "asset_discovery_scan": self._check_asset_inventory(system_data),
                "inventory_update_check": self._check_inventory_updates(system_data),
                "identity_system_test": self._check_identity_management(system_data),
                "credential_audit": self._check_credential_management(system_data),
                "verification_check": self._check_identity_verification(system_data)
            }
            
            return check_results.get(check_name, {
                "passed": False,
                "details": f"Check '{check_name}' not implemented",
                "remediation": f"Implement automated check for {check_name}"
            })
            
        except Exception as e:
            self.logger.error(f"Error executing automated check {check_name}: {str(e)}")
            return {
                "passed": False,
                "details": f"Check execution failed: {str(e)}",
                "remediation": f"Fix automated check implementation for {check_name}"
            }
    
    # Automated check implementations
    
    def _check_firewall_config(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check firewall configuration compliance"""
        firewalls = system_data.get("firewalls", [])
        
        if not firewalls:
            return {
                "passed": False,
                "details": "No firewall configurations found",
                "remediation": "Deploy and configure firewalls according to security standards"
            }
        
        # Check for basic firewall rules
        compliant_firewalls = 0
        for firewall in firewalls:
            rules = firewall.get("rules", [])
            if len(rules) > 0 and any("deny all" in str(rule).lower() for rule in rules):
                compliant_firewalls += 1
        
        compliance_rate = compliant_firewalls / len(firewalls)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"{compliant_firewalls}/{len(firewalls)} firewalls properly configured",
            "remediation": "Configure all firewalls with proper deny-all default rules"
        }
    
    def _check_router_config(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check router configuration compliance"""
        routers = system_data.get("routers", [])
        
        if not routers:
            return {
                "passed": True,  # No routers to check
                "details": "No routers found in scope",
                "remediation": "N/A"
            }
        
        secure_routers = 0
        for router in routers:
            config = router.get("configuration", {})
            if config.get("ssh_enabled") and not config.get("telnet_enabled"):
                secure_routers += 1
        
        compliance_rate = secure_routers / len(routers)
        
        return {
            "passed": compliance_rate >= 0.9,
            "details": f"{secure_routers}/{len(routers)} routers securely configured",
            "remediation": "Disable Telnet and enable SSH on all routers"
        }
    
    def _check_default_accounts(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for default accounts"""
        user_accounts = system_data.get("user_accounts", [])
        
        default_accounts = ["admin", "administrator", "root", "guest", "test", "demo"]
        found_defaults = []
        
        for account in user_accounts:
            username = account.get("username", "").lower()
            if username in default_accounts and account.get("status") == "active":
                found_defaults.append(username)
        
        return {
            "passed": len(found_defaults) == 0,
            "details": f"Found {len(found_defaults)} active default accounts: {', '.join(found_defaults)}" if found_defaults else "No active default accounts found",
            "remediation": "Disable or rename all default accounts" if found_defaults else "N/A"
        }
    
    def _check_password_policy(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check password policy compliance"""
        policy = system_data.get("password_policy", {})
        
        required_settings = {
            "min_length": 8,
            "require_upper": True,
            "require_lower": True,
            "require_numbers": True,
            "require_special": True,
            "max_age_days": 90
        }
        
        compliant_settings = 0
        total_settings = len(required_settings)
        
        for setting, required_value in required_settings.items():
            actual_value = policy.get(setting)
            if isinstance(required_value, bool):
                if actual_value == required_value:
                    compliant_settings += 1
            elif isinstance(required_value, int):
                if isinstance(actual_value, int) and actual_value >= required_value:
                    compliant_settings += 1
        
        compliance_rate = compliant_settings / total_settings
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Password policy compliance: {compliant_settings}/{total_settings} requirements met",
            "remediation": "Update password policy to meet all security requirements"
        }
    
    def _check_data_discovery(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data discovery and classification"""
        data_stores = system_data.get("data_stores", [])
        
        classified_stores = 0
        for store in data_stores:
            if store.get("classification") and store.get("data_inventory"):
                classified_stores += 1
        
        if not data_stores:
            return {
                "passed": False,
                "details": "No data stores identified",
                "remediation": "Perform comprehensive data discovery and classification"
            }
        
        compliance_rate = classified_stores / len(data_stores)
        
        return {
            "passed": compliance_rate >= 0.9,
            "details": f"{classified_stores}/{len(data_stores)} data stores properly classified",
            "remediation": "Complete data classification for all data stores"
        }
    
    def _check_retention_policy(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data retention policy implementation"""
        retention_policy = system_data.get("data_retention_policy", {})
        
        if not retention_policy:
            return {
                "passed": False,
                "details": "No data retention policy found",
                "remediation": "Implement comprehensive data retention policy"
            }
        
        required_elements = ["retention_periods", "deletion_procedures", "policy_review_date"]
        present_elements = [elem for elem in required_elements if retention_policy.get(elem)]
        
        compliance_rate = len(present_elements) / len(required_elements)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Data retention policy has {len(present_elements)}/{len(required_elements)} required elements",
            "remediation": "Complete all required elements in data retention policy"
        }
    
    def _check_user_accounts(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check user account management"""
        user_accounts = system_data.get("user_accounts", [])
        
        if not user_accounts:
            return {
                "passed": False,
                "details": "No user accounts found",
                "remediation": "Implement user account inventory and management system"
            }
        
        compliant_accounts = 0
        for account in user_accounts:
            if (account.get("unique_id") and 
                account.get("last_review_date") and 
                account.get("access_level")):
                compliant_accounts += 1
        
        compliance_rate = compliant_accounts / len(user_accounts)
        
        return {
            "passed": compliance_rate >= 0.9,
            "details": f"{compliant_accounts}/{len(user_accounts)} user accounts properly managed",
            "remediation": "Ensure all user accounts have unique IDs, regular reviews, and defined access levels"
        }
    
    def _check_authentication(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check authentication mechanisms"""
        auth_systems = system_data.get("authentication_systems", [])
        
        if not auth_systems:
            return {
                "passed": False,
                "details": "No authentication systems found",
                "remediation": "Implement robust authentication systems"
            }
        
        strong_auth_systems = 0
        for system in auth_systems:
            if (system.get("multi_factor_enabled") and 
                system.get("password_complexity") and 
                system.get("account_lockout")):
                strong_auth_systems += 1
        
        compliance_rate = strong_auth_systems / len(auth_systems)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"{strong_auth_systems}/{len(auth_systems)} authentication systems properly configured",
            "remediation": "Enable MFA, password complexity, and account lockout on all authentication systems"
        }
    
    def _check_access_control(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check access control implementation"""
        access_controls = system_data.get("access_controls", {})
        
        required_controls = ["role_based_access", "principle_of_least_privilege", "access_review_process"]
        implemented_controls = [control for control in required_controls if access_controls.get(control)]
        
        compliance_rate = len(implemented_controls) / len(required_controls)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Access controls: {len(implemented_controls)}/{len(required_controls)} implemented",
            "remediation": "Implement all required access control mechanisms"
        }
    
    def _check_user_rights(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check user rights and permissions"""
        user_rights = system_data.get("user_rights_audit", {})
        
        if not user_rights:
            return {
                "passed": False,
                "details": "No user rights audit data available",
                "remediation": "Perform comprehensive user rights audit"
            }
        
        excessive_rights = user_rights.get("excessive_rights_count", 0)
        total_users = user_rights.get("total_users", 1)
        
        excessive_rate = excessive_rights / total_users
        
        return {
            "passed": excessive_rate <= 0.1,
            "details": f"{excessive_rights}/{total_users} users have excessive rights",
            "remediation": "Review and reduce excessive user rights"
        }
    
    def _check_automatic_logoff(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check automatic logoff configuration"""
        systems = system_data.get("systems", [])
        
        compliant_systems = 0
        for system in systems:
            logoff_config = system.get("automatic_logoff", {})
            if logoff_config.get("enabled") and logoff_config.get("timeout_minutes", 0) <= 30:
                compliant_systems += 1
        
        if not systems:
            return {
                "passed": False,
                "details": "No systems found",
                "remediation": "Configure automatic logoff on all systems"
            }
        
        compliance_rate = compliant_systems / len(systems)
        
        return {
            "passed": compliance_rate >= 0.9,
            "details": f"{compliant_systems}/{len(systems)} systems have proper automatic logoff",
            "remediation": "Configure automatic logoff (≤30 minutes) on all systems"
        }
    
    def _check_encryption(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check encryption implementation"""
        encryption_config = system_data.get("encryption", {})
        
        required_encryption = ["data_at_rest", "data_in_transit", "key_management"]
        implemented_encryption = [enc for enc in required_encryption if encryption_config.get(enc)]
        
        compliance_rate = len(implemented_encryption) / len(required_encryption)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Encryption: {len(implemented_encryption)}/{len(required_encryption)} types implemented",
            "remediation": "Implement encryption for data at rest, in transit, and proper key management"
        }
    
    def _check_network_security(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check network security controls"""
        network_security = system_data.get("network_security", {})
        
        required_controls = ["intrusion_detection", "network_segmentation", "secure_protocols"]
        implemented_controls = [control for control in required_controls if network_security.get(control)]
        
        compliance_rate = len(implemented_controls) / len(required_controls)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Network security: {len(implemented_controls)}/{len(required_controls)} controls implemented",
            "remediation": "Implement intrusion detection, network segmentation, and secure protocols"
        }
    
    def _check_transmission_security(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check transmission security"""
        transmission_security = system_data.get("transmission_security", {})
        
        if transmission_security.get("encryption_enabled") and transmission_security.get("secure_protocols"):
            return {
                "passed": True,
                "details": "Transmission security properly implemented",
                "remediation": "N/A"
            }
        else:
            return {
                "passed": False,
                "details": "Transmission security not properly implemented",
                "remediation": "Enable encryption and secure protocols for all data transmission"
            }
    
    # Additional check methods for other frameworks...
    
    def _check_policy_review(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check policy review processes"""
        policies = system_data.get("policies", [])
        
        if not policies:
            return {
                "passed": False,
                "details": "No policies found",
                "remediation": "Establish and document security policies"
            }
        
        current_policies = 0
        for policy in policies:
            review_date = policy.get("last_review_date")
            if review_date and self._is_recent_date(review_date, days=365):
                current_policies += 1
        
        compliance_rate = current_policies / len(policies)
        
        return {
            "passed": compliance_rate >= 0.9,
            "details": f"{current_policies}/{len(policies)} policies recently reviewed",
            "remediation": "Review all policies at least annually"
        }
    
    def _check_access_policy(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check access control policy"""
        access_policy = system_data.get("access_control_policy", {})
        
        required_elements = ["policy_document", "approval_date", "review_schedule"]
        present_elements = [elem for elem in required_elements if access_policy.get(elem)]
        
        compliance_rate = len(present_elements) / len(required_elements)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Access policy has {len(present_elements)}/{len(required_elements)} required elements",
            "remediation": "Complete access control policy documentation and approval"
        }
    
    def _check_access_procedures(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check access control procedures"""
        procedures = system_data.get("access_procedures", {})
        
        required_procedures = ["user_provisioning", "access_review", "access_revocation"]
        documented_procedures = [proc for proc in required_procedures if procedures.get(proc)]
        
        compliance_rate = len(documented_procedures) / len(required_procedures)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Access procedures: {len(documented_procedures)}/{len(required_procedures)} documented",
            "remediation": "Document all required access control procedures"
        }
    
    def _check_data_processing(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data processing compliance (GDPR)"""
        data_processing = system_data.get("data_processing", {})
        
        required_elements = ["lawful_basis", "data_minimization", "purpose_limitation"]
        implemented_elements = [elem for elem in required_elements if data_processing.get(elem)]
        
        compliance_rate = len(implemented_elements) / len(required_elements)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Data processing: {len(implemented_elements)}/{len(required_elements)} principles implemented",
            "remediation": "Implement all GDPR data processing principles"
        }
    
    def _check_privacy_notices(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check privacy notice compliance"""
        privacy_notices = system_data.get("privacy_notices", [])
        
        if not privacy_notices:
            return {
                "passed": False,
                "details": "No privacy notices found",
                "remediation": "Create and publish privacy notices"
            }
        
        compliant_notices = 0
        for notice in privacy_notices:
            if (notice.get("purpose_of_processing") and 
                notice.get("legal_basis") and 
                notice.get("data_retention")):
                compliant_notices += 1
        
        compliance_rate = compliant_notices / len(privacy_notices)
        
        return {
            "passed": compliance_rate >= 0.9,
            "details": f"{compliant_notices}/{len(privacy_notices)} privacy notices compliant",
            "remediation": "Update privacy notices to include all required elements"
        }
    
    def _check_encryption_implementation(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check encryption implementation details"""
        encryption = system_data.get("encryption_implementation", {})
        
        strong_encryption = (
            encryption.get("algorithm_strength") == "AES-256" and
            encryption.get("key_length", 0) >= 256 and
            encryption.get("key_rotation", False)
        )
        
        return {
            "passed": strong_encryption,
            "details": "Strong encryption implemented" if strong_encryption else "Weak or missing encryption",
            "remediation": "Implement AES-256 encryption with proper key management" if not strong_encryption else "N/A"
        }
    
    def _check_confidentiality_controls(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check confidentiality controls"""
        confidentiality = system_data.get("confidentiality_controls", {})
        
        required_controls = ["access_controls", "encryption", "data_classification"]
        implemented_controls = [control for control in required_controls if confidentiality.get(control)]
        
        compliance_rate = len(implemented_controls) / len(required_controls)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Confidentiality controls: {len(implemented_controls)}/{len(required_controls)} implemented",
            "remediation": "Implement all confidentiality controls"
        }
    
    def _check_security_measures(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check general security measures"""
        security_measures = system_data.get("security_measures", {})
        
        critical_measures = ["antivirus", "firewall", "intrusion_detection", "patch_management"]
        implemented_measures = [measure for measure in critical_measures if security_measures.get(measure)]
        
        compliance_rate = len(implemented_measures) / len(critical_measures)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Security measures: {len(implemented_measures)}/{len(critical_measures)} implemented",
            "remediation": "Implement all critical security measures"
        }
    
    def _check_asset_inventory(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check asset inventory management"""
        asset_inventory = system_data.get("asset_inventory", {})
        
        if not asset_inventory:
            return {
                "passed": False,
                "details": "No asset inventory found",
                "remediation": "Implement comprehensive asset inventory system"
            }
        
        required_data = ["asset_count", "last_update", "owner_assignment"]
        present_data = [data for data in required_data if asset_inventory.get(data)]
        
        compliance_rate = len(present_data) / len(required_data)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Asset inventory: {len(present_data)}/{len(required_data)} data elements present",
            "remediation": "Complete asset inventory with all required data elements"
        }
    
    def _check_inventory_updates(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check inventory update processes"""
        update_process = system_data.get("inventory_updates", {})
        
        if update_process.get("automated") and update_process.get("frequency") == "monthly":
            return {
                "passed": True,
                "details": "Automated monthly inventory updates configured",
                "remediation": "N/A"
            }
        else:
            return {
                "passed": False,
                "details": "Inventory updates not properly configured",
                "remediation": "Implement automated monthly inventory updates"
            }
    
    def _check_identity_management(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check identity management system"""
        identity_system = system_data.get("identity_management", {})
        
        required_features = ["identity_provisioning", "lifecycle_management", "identity_verification"]
        implemented_features = [feature for feature in required_features if identity_system.get(feature)]
        
        compliance_rate = len(implemented_features) / len(required_features)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Identity management: {len(implemented_features)}/{len(required_features)} features implemented",
            "remediation": "Implement all identity management features"
        }
    
    def _check_credential_management(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check credential management"""
        credential_mgmt = system_data.get("credential_management", {})
        
        required_controls = ["credential_issuance", "lifecycle_management", "revocation_process"]
        implemented_controls = [control for control in required_controls if credential_mgmt.get(control)]
        
        compliance_rate = len(implemented_controls) / len(required_controls)
        
        return {
            "passed": compliance_rate >= 0.8,
            "details": f"Credential management: {len(implemented_controls)}/{len(required_controls)} controls implemented",
            "remediation": "Implement all credential management controls"
        }
    
    def _check_identity_verification(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check identity verification processes"""
        verification = system_data.get("identity_verification", {})
        
        if verification.get("multi_factor") and verification.get("verification_process"):
            return {
                "passed": True,
                "details": "Identity verification properly implemented",
                "remediation": "N/A"
            }
        else:
            return {
                "passed": False,
                "details": "Identity verification not properly implemented",
                "remediation": "Implement multi-factor identity verification"
            }
    
    def _is_recent_date(self, date_str: str, days: int) -> bool:
        """Check if date is within specified number of days"""
        try:
            if isinstance(date_str, str):
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                date = date_str
            
            return (datetime.utcnow() - date).days <= days
        except:
            return False


class AdvancedComplianceAutomationEngine(ComplianceService, XORBService):
    """
    Advanced Compliance Automation Engine
    Comprehensive compliance management with automated assessments and reporting
    """
    
    def __init__(self):
        super().__init__(
            service_id="advanced_compliance_automation",
            service_type=ServiceType.COMPLIANCE
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize automation engine
        self.automation_engine = ComplianceAutomationEngine()
        
        # Compliance monitoring
        self.active_assessments: Dict[str, Dict[str, Any]] = {}
        self.compliance_reports: Dict[str, ComplianceReport] = {}
        
        self.logger.info("✅ Advanced Compliance Automation Engine initialized")
    
    async def validate_compliance(
        self,
        framework: str,
        scan_results: Dict[str, Any],
        organization: Organization
    ) -> Dict[str, Any]:
        """Validate compliance against specific framework"""
        try:
            # Parse framework
            try:
                compliance_framework = ComplianceFramework(framework.lower())
            except ValueError:
                return {"error": f"Unsupported compliance framework: {framework}"}
            
            validation_id = str(uuid.uuid4())
            
            self.logger.info(f"Starting compliance validation for {framework} - validation ID: {validation_id}")
            
            # Get framework controls
            controls = self.automation_engine.framework_db.get_framework_controls(compliance_framework)
            
            if not controls:
                return {"error": f"No controls found for framework {framework}"}
            
            # Prepare system data from scan results
            system_data = self._prepare_system_data(scan_results, organization)
            
            # Perform assessments for each control
            assessments = []
            for control in controls:
                assessment = await self.automation_engine.automated_control_assessment(control, system_data)
                assessments.append(assessment)
            
            # Calculate overall compliance score
            total_score = sum(assessment.score for assessment in assessments)
            overall_score = total_score / len(assessments) if assessments else 0.0
            
            # Count compliance status
            compliant_count = len([a for a in assessments if a.status == ComplianceStatus.COMPLIANT])
            non_compliant_count = len([a for a in assessments if a.status == ComplianceStatus.NON_COMPLIANT])
            partial_count = len([a for a in assessments if a.status == ComplianceStatus.PARTIALLY_COMPLIANT])
            
            # Determine overall compliance status
            if overall_score >= 0.9:
                overall_status = "compliant"
            elif overall_score >= 0.7:
                overall_status = "partially_compliant"
            else:
                overall_status = "non_compliant"
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(assessments, compliance_framework)
            
            result = {
                "validation_id": validation_id,
                "framework": framework,
                "organization": organization.name,
                "overall_status": overall_status,
                "overall_score": round(overall_score, 3),
                "total_controls": len(controls),
                "compliant_controls": compliant_count,
                "non_compliant_controls": non_compliant_count,
                "partially_compliant_controls": partial_count,
                "control_assessments": [assessment.to_dict() for assessment in assessments],
                "recommendations": recommendations,
                "validation_date": datetime.utcnow().isoformat(),
                "next_assessment_due": (datetime.utcnow() + timedelta(days=365)).isoformat()
            }
            
            # Store assessment
            self.active_assessments[validation_id] = result
            
            self.logger.info(f"Completed compliance validation for {framework}: {overall_status} ({overall_score:.1%})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating compliance: {str(e)}")
            return {"error": str(e)}
    
    async def generate_compliance_report(
        self,
        framework: str,
        time_period: str,
        organization: Organization
    ) -> Dict[str, Any]:
        """Generate compliance report for specified period"""
        try:
            # Parse framework
            try:
                compliance_framework = ComplianceFramework(framework.lower())
            except ValueError:
                return {"error": f"Unsupported compliance framework: {framework}"}
            
            report_id = str(uuid.uuid4())
            
            self.logger.info(f"Generating compliance report for {framework} - report ID: {report_id}")
            
            # Get recent assessments for the framework
            recent_assessments = self._get_recent_assessments(compliance_framework, time_period)
            
            if not recent_assessments:
                return {"error": f"No recent assessments found for {framework}"}
            
            # Calculate report metrics
            total_controls = len(recent_assessments)
            compliant_controls = len([a for a in recent_assessments if a.status == ComplianceStatus.COMPLIANT])
            non_compliant_controls = len([a for a in recent_assessments if a.status == ComplianceStatus.NON_COMPLIANT])
            partial_controls = len([a for a in recent_assessments if a.status == ComplianceStatus.PARTIALLY_COMPLIANT])
            
            overall_score = sum(a.score for a in recent_assessments) / total_controls if total_controls > 0 else 0.0
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                compliance_framework, overall_score, compliant_controls, total_controls
            )
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(recent_assessments, compliance_framework)
            
            # Create compliance report
            report = ComplianceReport(
                report_id=report_id,
                framework=compliance_framework,
                organization=organization.name,
                report_type="assessment",
                overall_score=overall_score,
                total_controls=total_controls,
                compliant_controls=compliant_controls,
                non_compliant_controls=non_compliant_controls,
                partially_compliant_controls=partial_controls,
                assessments=recent_assessments,
                executive_summary=executive_summary,
                recommendations=recommendations,
                valid_until=datetime.utcnow() + timedelta(days=365)
            )
            
            # Store report
            self.compliance_reports[report_id] = report
            
            self.logger.info(f"Generated compliance report {report_id} for {framework}")
            
            return report.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {str(e)}")
            return {"error": str(e)}
    
    async def get_compliance_gaps(
        self,
        framework: str,
        current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify compliance gaps and remediation steps"""
        try:
            # Parse framework
            try:
                compliance_framework = ComplianceFramework(framework.lower())
            except ValueError:
                return [{"error": f"Unsupported compliance framework: {framework}"}]
            
            # Get framework controls
            controls = self.automation_engine.framework_db.get_framework_controls(compliance_framework)
            
            gaps = []
            
            for control in controls:
                # Perform quick assessment
                assessment = await self.automation_engine.automated_control_assessment(control, current_state)
                
                if assessment.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.REMEDIATION_REQUIRED]:
                    gap = {
                        "control_id": control.control_id,
                        "control_name": control.name,
                        "category": control.category,
                        "severity": control.severity.value,
                        "current_status": assessment.status.value,
                        "compliance_score": assessment.score,
                        "gaps_identified": assessment.gaps,
                        "remediation_actions": assessment.remediation_actions,
                        "priority": self._calculate_gap_priority(control, assessment),
                        "estimated_effort": self._estimate_remediation_effort(control, assessment),
                        "business_impact": self._assess_business_impact(control)
                    }
                    gaps.append(gap)
            
            # Sort gaps by priority
            gaps.sort(key=lambda x: (x["priority"], x["severity"]), reverse=True)
            
            self.logger.info(f"Identified {len(gaps)} compliance gaps for {framework}")
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error identifying compliance gaps: {str(e)}")
            return [{"error": str(e)}]
    
    async def track_remediation_progress(
        self,
        compliance_issues: List[str],
        organization: Organization
    ) -> Dict[str, Any]:
        """Track progress of compliance remediation efforts"""
        try:
            tracking_id = str(uuid.uuid4())
            
            progress_data = {
                "tracking_id": tracking_id,
                "organization": organization.name,
                "total_issues": len(compliance_issues),
                "issues_progress": [],
                "overall_progress": 0.0,
                "completion_estimate": None,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            total_progress = 0.0
            
            for issue_id in compliance_issues:
                # Get issue details (this would typically query a remediation tracking system)
                issue_progress = self._get_remediation_progress(issue_id)
                
                progress_data["issues_progress"].append({
                    "issue_id": issue_id,
                    "status": issue_progress.get("status", "not_started"),
                    "progress_percent": issue_progress.get("progress_percent", 0),
                    "assigned_to": issue_progress.get("assigned_to", "unassigned"),
                    "due_date": issue_progress.get("due_date"),
                    "last_update": issue_progress.get("last_update", datetime.utcnow().isoformat())
                })
                
                total_progress += issue_progress.get("progress_percent", 0)
            
            # Calculate overall progress
            progress_data["overall_progress"] = total_progress / len(compliance_issues) if compliance_issues else 0.0
            
            # Estimate completion date
            if progress_data["overall_progress"] > 0:
                # Simple linear projection (could be enhanced with more sophisticated estimation)
                days_elapsed = 30  # Placeholder
                days_to_completion = (100 - progress_data["overall_progress"]) / progress_data["overall_progress"] * days_elapsed
                completion_estimate = datetime.utcnow() + timedelta(days=days_to_completion)
                progress_data["completion_estimate"] = completion_estimate.isoformat()
            
            return progress_data
            
        except Exception as e:
            self.logger.error(f"Error tracking remediation progress: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_system_data(self, scan_results: Dict[str, Any], organization: Organization) -> Dict[str, Any]:
        """Prepare system data for compliance assessment"""
        try:
            # Extract relevant data from scan results
            system_data = {
                "organization": organization.name,
                "scan_timestamp": datetime.utcnow().isoformat(),
                "firewalls": scan_results.get("network_security", {}).get("firewalls", []),
                "routers": scan_results.get("network_security", {}).get("routers", []),
                "user_accounts": scan_results.get("access_control", {}).get("user_accounts", []),
                "password_policy": scan_results.get("access_control", {}).get("password_policy", {}),
                "data_stores": scan_results.get("data_protection", {}).get("data_stores", []),
                "data_retention_policy": scan_results.get("data_protection", {}).get("retention_policy", {}),
                "authentication_systems": scan_results.get("access_control", {}).get("authentication_systems", []),
                "access_controls": scan_results.get("access_control", {}),
                "user_rights_audit": scan_results.get("access_control", {}).get("user_rights_audit", {}),
                "systems": scan_results.get("systems", []),
                "encryption": scan_results.get("encryption", {}),
                "network_security": scan_results.get("network_security", {}),
                "transmission_security": scan_results.get("transmission_security", {}),
                "policies": scan_results.get("policies", []),
                "access_control_policy": scan_results.get("access_control_policy", {}),
                "access_procedures": scan_results.get("access_procedures", {}),
                "data_processing": scan_results.get("data_processing", {}),
                "privacy_notices": scan_results.get("privacy_notices", []),
                "encryption_implementation": scan_results.get("encryption_implementation", {}),
                "confidentiality_controls": scan_results.get("confidentiality_controls", {}),
                "security_measures": scan_results.get("security_measures", {}),
                "asset_inventory": scan_results.get("asset_inventory", {}),
                "inventory_updates": scan_results.get("inventory_updates", {}),
                "identity_management": scan_results.get("identity_management", {}),
                "credential_management": scan_results.get("credential_management", {}),
                "identity_verification": scan_results.get("identity_verification", {})
            }
            
            return system_data
            
        except Exception as e:
            self.logger.error(f"Error preparing system data: {str(e)}")
            return {}
    
    def _generate_compliance_recommendations(
        self,
        assessments: List[ComplianceAssessment],
        framework: ComplianceFramework
    ) -> List[str]:
        """Generate compliance recommendations based on assessments"""
        recommendations = []
        
        try:
            # High-priority recommendations for non-compliant controls
            critical_failures = [a for a in assessments if 
                               a.status == ComplianceStatus.NON_COMPLIANT and 
                               any("critical" in str(gap).lower() for gap in a.gaps)]
            
            if critical_failures:
                recommendations.append(f"URGENT: Address {len(critical_failures)} critical compliance failures immediately")
                for assessment in critical_failures[:3]:  # Top 3 critical issues
                    recommendations.extend(assessment.remediation_actions[:1])  # Top remediation for each
            
            # Medium-priority recommendations
            partial_compliance = [a for a in assessments if a.status == ComplianceStatus.PARTIALLY_COMPLIANT]
            if partial_compliance:
                recommendations.append(f"MODERATE: Improve {len(partial_compliance)} partially compliant controls")
            
            # Framework-specific recommendations
            framework_recommendations = {
                ComplianceFramework.PCI_DSS: [
                    "Implement network segmentation to isolate cardholder data environment",
                    "Deploy file integrity monitoring for critical system files",
                    "Establish regular penetration testing and vulnerability scanning programs"
                ],
                ComplianceFramework.HIPAA: [
                    "Implement comprehensive ePHI encryption for data at rest and in transit",
                    "Establish workforce training program for HIPAA privacy and security",
                    "Deploy audit logging and monitoring for all ePHI access"
                ],
                ComplianceFramework.SOX: [
                    "Strengthen IT general controls for financial reporting systems",
                    "Implement change management controls for financial applications",
                    "Establish formal risk assessment process for IT controls"
                ],
                ComplianceFramework.GDPR: [
                    "Implement privacy by design principles in all data processing",
                    "Establish data subject rights fulfillment procedures",
                    "Deploy data breach detection and notification capabilities"
                ],
                ComplianceFramework.ISO_27001: [
                    "Establish comprehensive information security management system (ISMS)",
                    "Implement risk management process for information security",
                    "Deploy security awareness and training program"
                ]
            }
            
            framework_specific = framework_recommendations.get(framework, [])
            recommendations.extend(framework_specific)
            
            # General recommendations
            recommendations.extend([
                "Implement continuous compliance monitoring and reporting",
                "Establish regular compliance training for relevant personnel",
                "Deploy automated compliance testing and validation tools",
                "Create incident response procedures for compliance violations"
            ])
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Manual compliance review required due to analysis error")
        
        return recommendations
    
    def _get_recent_assessments(
        self,
        framework: ComplianceFramework,
        time_period: str
    ) -> List[ComplianceAssessment]:
        """Get recent assessments for a framework"""
        # This would typically query a database of assessments
        # For now, return assessments from automation engine
        assessments = []
        
        try:
            for control_id, control_assessments in self.automation_engine.assessment_history.items():
                # Get most recent assessment for each control
                if control_assessments:
                    recent_assessment = max(control_assessments, key=lambda x: x.assessment_date)
                    assessments.append(recent_assessment)
            
            return assessments
            
        except Exception as e:
            self.logger.error(f"Error getting recent assessments: {str(e)}")
            return []
    
    def _generate_executive_summary(
        self,
        framework: ComplianceFramework,
        overall_score: float,
        compliant_controls: int,
        total_controls: int
    ) -> str:
        """Generate executive summary for compliance report"""
        try:
            compliance_percentage = (compliant_controls / total_controls * 100) if total_controls > 0 else 0
            
            summary = f"Compliance assessment for {framework.value.upper()} framework completed. "
            summary += f"Overall compliance score: {overall_score:.1%} "
            summary += f"({compliant_controls}/{total_controls} controls fully compliant). "
            
            if overall_score >= 0.9:
                summary += "Organization demonstrates strong compliance posture with minimal gaps. "
                summary += "Continue regular monitoring and maintenance of existing controls."
            elif overall_score >= 0.7:
                summary += "Organization shows good compliance foundation with some areas for improvement. "
                summary += "Focus on addressing partially compliant controls to achieve full compliance."
            elif overall_score >= 0.5:
                summary += "Organization has basic compliance measures in place but requires significant improvement. "
                summary += "Prioritize addressing non-compliant controls and strengthening existing measures."
            else:
                summary += "Organization faces significant compliance challenges requiring immediate attention. "
                summary += "Implement comprehensive remediation plan to address critical compliance gaps."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return "Executive summary could not be generated due to analysis error."
    
    def _calculate_gap_priority(self, control: ComplianceControl, assessment: ComplianceAssessment) -> int:
        """Calculate priority score for compliance gap (1-10, higher is more urgent)"""
        try:
            base_priority = {
                ControlSeverity.CRITICAL: 10,
                ControlSeverity.HIGH: 8,
                ControlSeverity.MEDIUM: 5,
                ControlSeverity.LOW: 3,
                ControlSeverity.INFORMATIONAL: 1
            }.get(control.severity, 5)
            
            # Adjust based on compliance score (lower score = higher priority)
            score_adjustment = int((1.0 - assessment.score) * 3)
            
            # Adjust based on status
            status_adjustment = {
                ComplianceStatus.NON_COMPLIANT: 3,
                ComplianceStatus.REMEDIATION_REQUIRED: 2,
                ComplianceStatus.PARTIALLY_COMPLIANT: 1,
                ComplianceStatus.NOT_ASSESSED: 1
            }.get(assessment.status, 0)
            
            final_priority = min(10, base_priority + score_adjustment + status_adjustment)
            return final_priority
            
        except Exception:
            return 5  # Default medium priority
    
    def _estimate_remediation_effort(self, control: ComplianceControl, assessment: ComplianceAssessment) -> str:
        """Estimate effort required for remediation"""
        try:
            gap_count = len(assessment.gaps)
            remediation_count = len(assessment.remediation_actions)
            
            total_work = gap_count + remediation_count
            
            if total_work >= 10:
                return "high"
            elif total_work >= 5:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _assess_business_impact(self, control: ComplianceControl) -> str:
        """Assess business impact of compliance control"""
        impact_map = {
            ControlSeverity.CRITICAL: "high",
            ControlSeverity.HIGH: "high",
            ControlSeverity.MEDIUM: "medium",
            ControlSeverity.LOW: "low",
            ControlSeverity.INFORMATIONAL: "low"
        }
        
        return impact_map.get(control.severity, "medium")
    
    def _get_remediation_progress(self, issue_id: str) -> Dict[str, Any]:
        """Get remediation progress for a specific issue"""
        # This would typically query a remediation tracking system
        # For now, return simulated progress data
        import random
        
        statuses = ["not_started", "in_progress", "testing", "completed"]
        
        return {
            "status": random.choice(statuses),
            "progress_percent": random.randint(0, 100),
            "assigned_to": "security_team",
            "due_date": (datetime.utcnow() + timedelta(days=random.randint(7, 90))).isoformat(),
            "last_update": datetime.utcnow().isoformat()
        }