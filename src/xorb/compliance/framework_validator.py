"""
Enterprise Compliance Framework Validator
Real-world compliance validation for PCI-DSS, HIPAA, SOX, ISO-27001, and other frameworks
"""

import asyncio
import json
import logging
import csv
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import uuid
import hashlib
import re

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    PCI_DSS = "PCI-DSS"
    HIPAA = "HIPAA"
    SOX = "SOX"
    ISO_27001 = "ISO-27001"
    GDPR = "GDPR"
    NIST_CSF = "NIST-CSF"
    SOC2 = "SOC2"
    FISMA = "FISMA"
    CCPA = "CCPA"
    FedRAMP = "FedRAMP"

class ControlStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    NOT_TESTED = "not_tested"
    REMEDIATION_REQUIRED = "remediation_required"

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceControl:
    """Individual compliance control definition"""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    category: str
    subcategory: str
    implementation_guidance: List[str]
    testing_procedures: List[str]
    evidence_requirements: List[str]
    automated_tests: List[str]
    manual_validation_required: bool
    severity: SeverityLevel
    references: List[str]

@dataclass
class ControlAssessment:
    """Assessment result for a compliance control"""
    assessment_id: str
    control_id: str
    framework: ComplianceFramework
    status: ControlStatus
    compliance_score: float  # 0.0 to 1.0
    findings: List[str]
    evidence_collected: List[str]
    gaps_identified: List[str]
    remediation_steps: List[str]
    assessor: str
    assessment_date: datetime
    next_assessment_due: datetime
    notes: str

@dataclass
class ComplianceReport:
    """Comprehensive compliance assessment report"""
    report_id: str
    framework: ComplianceFramework
    organization: str
    assessment_scope: str
    assessment_period: Tuple[datetime, datetime]
    overall_compliance_score: float
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    partially_compliant_controls: int
    critical_findings: List[str]
    high_findings: List[str]
    remediation_priorities: List[str]
    assessments: List[ControlAssessment]
    generated_at: datetime
    valid_until: datetime

class ComplianceFrameworkValidator:
    """Production compliance framework validation system"""

    def __init__(self, data_path: str = "./data/compliance"):
        self.data_path = Path(data_path)
        self.db_path = self.data_path / "compliance.db"
        self.controls: Dict[str, ComplianceControl] = {}
        self.assessments: List[ControlAssessment] = []
        self.frameworks: Dict[ComplianceFramework, Dict[str, Any]] = {}

        # Initialize compliance framework definitions
        self._initialize_framework_controls()

    async def initialize(self) -> bool:
        """Initialize the compliance validation system"""
        try:
            logger.info("Initializing Enterprise Compliance Framework Validator...")

            # Create data directory
            self.data_path.mkdir(parents=True, exist_ok=True)

            # Initialize database
            await self._initialize_database()

            # Load compliance framework definitions
            await self._load_framework_definitions()

            # Load historical assessments
            await self._load_assessment_history()

            logger.info(f"Compliance validator initialized with {len(self.controls)} controls across {len(self.frameworks)} frameworks")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize compliance validator: {e}")
            return False

    async def _initialize_database(self):
        """Initialize SQLite database for compliance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_controls (
                control_id TEXT PRIMARY KEY,
                framework TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                requirement TEXT,
                category TEXT,
                subcategory TEXT,
                implementation_guidance TEXT,
                testing_procedures TEXT,
                evidence_requirements TEXT,
                automated_tests TEXT,
                manual_validation_required BOOLEAN,
                severity TEXT,
                references TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS control_assessments (
                assessment_id TEXT PRIMARY KEY,
                control_id TEXT NOT NULL,
                framework TEXT NOT NULL,
                status TEXT NOT NULL,
                compliance_score REAL,
                findings TEXT,
                evidence_collected TEXT,
                gaps_identified TEXT,
                remediation_steps TEXT,
                assessor TEXT,
                assessment_date TIMESTAMP,
                next_assessment_due TIMESTAMP,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_reports (
                report_id TEXT PRIMARY KEY,
                framework TEXT NOT NULL,
                organization TEXT,
                assessment_scope TEXT,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                overall_compliance_score REAL,
                total_controls INTEGER,
                compliant_controls INTEGER,
                non_compliant_controls INTEGER,
                partially_compliant_controls INTEGER,
                critical_findings TEXT,
                high_findings TEXT,
                remediation_priorities TEXT,
                generated_at TIMESTAMP,
                valid_until TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_controls_framework ON compliance_controls(framework)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assessments_control ON control_assessments(control_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assessments_date ON control_assessments(assessment_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_framework ON compliance_reports(framework)")

        conn.commit()
        conn.close()

        logger.info("Compliance database initialized")

    def _initialize_framework_controls(self):
        """Initialize comprehensive compliance framework controls"""

        # PCI-DSS Controls
        pci_controls = [
            ComplianceControl(
                control_id="PCI.1.1",
                framework=ComplianceFramework.PCI_DSS,
                title="Install and maintain firewall and router configuration standards",
                description="Establish firewall and router configuration standards that include a formal process for approving and testing all network connections",
                requirement="Document and implement firewall and router configuration standards",
                category="Network Security",
                subcategory="Firewall Management",
                implementation_guidance=[
                    "Document firewall and router standards",
                    "Implement change control processes",
                    "Regular review and updates",
                    "Testing of all network connections"
                ],
                testing_procedures=[
                    "Review firewall documentation",
                    "Test firewall rules",
                    "Verify change control process",
                    "Validate network segmentation"
                ],
                evidence_requirements=[
                    "Firewall configuration standards document",
                    "Change control procedures",
                    "Testing documentation",
                    "Network diagrams"
                ],
                automated_tests=[
                    "firewall_config_scan",
                    "network_segmentation_test",
                    "open_ports_scan"
                ],
                manual_validation_required=True,
                severity=SeverityLevel.HIGH,
                references=["PCI DSS v4.0"]
            ),
            ComplianceControl(
                control_id="PCI.3.4",
                framework=ComplianceFramework.PCI_DSS,
                title="Render cardholder data unreadable anywhere it is stored",
                description="Protect stored cardholder data through strong cryptography and security management",
                requirement="Encrypt cardholder data in databases, files, and removable media",
                category="Data Protection",
                subcategory="Encryption",
                implementation_guidance=[
                    "Implement strong encryption for stored data",
                    "Use approved encryption algorithms",
                    "Manage encryption keys securely",
                    "Regular validation of encryption"
                ],
                testing_procedures=[
                    "Verify encryption implementation",
                    "Test key management procedures",
                    "Validate data protection methods",
                    "Check for unencrypted cardholder data"
                ],
                evidence_requirements=[
                    "Encryption policy documentation",
                    "Key management procedures",
                    "Data discovery scan results",
                    "Encryption verification reports"
                ],
                automated_tests=[
                    "data_encryption_scan",
                    "key_management_check",
                    "unencrypted_data_discovery"
                ],
                manual_validation_required=True,
                severity=SeverityLevel.CRITICAL,
                references=["PCI DSS v4.0"]
            ),
            ComplianceControl(
                control_id="PCI.11.2",
                framework=ComplianceFramework.PCI_DSS,
                title="Run internal and external network vulnerability scans",
                description="Perform quarterly internal vulnerability scans and annual penetration testing",
                requirement="Conduct regular vulnerability assessments",
                category="Security Testing",
                subcategory="Vulnerability Management",
                implementation_guidance=[
                    "Quarterly internal vulnerability scans",
                    "Annual external penetration testing",
                    "Address high and critical vulnerabilities",
                    "Use qualified personnel or vendors"
                ],
                testing_procedures=[
                    "Review vulnerability scan reports",
                    "Verify scan frequency and coverage",
                    "Check remediation timelines",
                    "Validate penetration testing results"
                ],
                evidence_requirements=[
                    "Vulnerability scan reports",
                    "Penetration testing reports",
                    "Remediation tracking",
                    "Scan tool configurations"
                ],
                automated_tests=[
                    "vulnerability_scan",
                    "penetration_test_execution",
                    "remediation_verification"
                ],
                manual_validation_required=False,
                severity=SeverityLevel.HIGH,
                references=["PCI DSS v4.0"]
            )
        ]

        # HIPAA Controls
        hipaa_controls = [
            ComplianceControl(
                control_id="HIPAA.164.308.a.1.i",
                framework=ComplianceFramework.HIPAA,
                title="Security Officer",
                description="Assign security responsibilities to a designated security officer",
                requirement="Designate a security officer responsible for developing and implementing security policies",
                category="Administrative Safeguards",
                subcategory="Security Management Process",
                implementation_guidance=[
                    "Designate security officer",
                    "Define security responsibilities",
                    "Implement security policies",
                    "Regular security reviews"
                ],
                testing_procedures=[
                    "Verify security officer designation",
                    "Review security policies",
                    "Check responsibility assignments",
                    "Validate security program effectiveness"
                ],
                evidence_requirements=[
                    "Security officer designation documentation",
                    "Security policies and procedures",
                    "Responsibility matrix",
                    "Security program documentation"
                ],
                automated_tests=[
                    "policy_documentation_check",
                    "security_program_assessment"
                ],
                manual_validation_required=True,
                severity=SeverityLevel.HIGH,
                references=["45 CFR 164.308(a)(2)"]
            ),
            ComplianceControl(
                control_id="HIPAA.164.312.a.1",
                framework=ComplianceFramework.HIPAA,
                title="Access Control",
                description="Implement technical policies and procedures for electronic information systems",
                requirement="Assign unique user identification for each person with access to ePHI",
                category="Technical Safeguards",
                subcategory="Access Control",
                implementation_guidance=[
                    "Unique user identification",
                    "Role-based access control",
                    "Minimum necessary access",
                    "Regular access reviews"
                ],
                testing_procedures=[
                    "Review user access provisioning",
                    "Test access control mechanisms",
                    "Verify role assignments",
                    "Check access logging"
                ],
                evidence_requirements=[
                    "User access documentation",
                    "Role definitions",
                    "Access review reports",
                    "Audit logs"
                ],
                automated_tests=[
                    "user_access_audit",
                    "role_based_access_check",
                    "access_review_validation"
                ],
                manual_validation_required=True,
                severity=SeverityLevel.HIGH,
                references=["45 CFR 164.312(a)(1)"]
            )
        ]

        # SOX Controls
        sox_controls = [
            ComplianceControl(
                control_id="SOX.302",
                framework=ComplianceFramework.SOX,
                title="Corporate Responsibility for Financial Reports",
                description="Principal executive and financial officers must certify financial reports",
                requirement="CEO and CFO certification of financial reports accuracy",
                category="Financial Reporting",
                subcategory="Executive Certification",
                implementation_guidance=[
                    "Establish certification process",
                    "Document control procedures",
                    "Regular control testing",
                    "Deficiency remediation"
                ],
                testing_procedures=[
                    "Review certification documentation",
                    "Test control effectiveness",
                    "Verify remediation procedures",
                    "Check disclosure controls"
                ],
                evidence_requirements=[
                    "Certification statements",
                    "Control documentation",
                    "Testing results",
                    "Deficiency reports"
                ],
                automated_tests=[
                    "control_documentation_review",
                    "certification_process_check"
                ],
                manual_validation_required=True,
                severity=SeverityLevel.CRITICAL,
                references=["SOX Section 302"]
            ),
            ComplianceControl(
                control_id="SOX.404",
                framework=ComplianceFramework.SOX,
                title="Management Assessment of Internal Controls",
                description="Annual assessment of internal control over financial reporting",
                requirement="Management must assess and report on internal control effectiveness",
                category="Internal Controls",
                subcategory="Management Assessment",
                implementation_guidance=[
                    "Annual internal control assessment",
                    "Document control framework",
                    "Test control effectiveness",
                    "Report control deficiencies"
                ],
                testing_procedures=[
                    "Review assessment documentation",
                    "Test control design and operation",
                    "Verify deficiency reporting",
                    "Check remediation efforts"
                ],
                evidence_requirements=[
                    "Assessment reports",
                    "Control testing documentation",
                    "Deficiency analysis",
                    "Remediation plans"
                ],
                automated_tests=[
                    "control_effectiveness_test",
                    "assessment_documentation_check"
                ],
                manual_validation_required=True,
                severity=SeverityLevel.CRITICAL,
                references=["SOX Section 404"]
            )
        ]

        # ISO 27001 Controls
        iso_controls = [
            ComplianceControl(
                control_id="ISO.A.12.6.1",
                framework=ComplianceFramework.ISO_27001,
                title="Management of technical vulnerabilities",
                description="Information about technical vulnerabilities should be obtained and managed",
                requirement="Establish procedures for managing technical vulnerabilities",
                category="Operations Security",
                subcategory="Vulnerability Management",
                implementation_guidance=[
                    "Define vulnerability management process",
                    "Regular vulnerability assessments",
                    "Prioritize vulnerability remediation",
                    "Track remediation progress"
                ],
                testing_procedures=[
                    "Review vulnerability management procedures",
                    "Test vulnerability scanning processes",
                    "Verify remediation timelines",
                    "Check vulnerability tracking"
                ],
                evidence_requirements=[
                    "Vulnerability management policy",
                    "Vulnerability scan reports",
                    "Remediation tracking",
                    "Process documentation"
                ],
                automated_tests=[
                    "vulnerability_management_assessment",
                    "remediation_tracking_check"
                ],
                manual_validation_required=True,
                severity=SeverityLevel.HIGH,
                references=["ISO/IEC 27001:2022"]
            )
        ]

        # GDPR Controls
        gdpr_controls = [
            ComplianceControl(
                control_id="GDPR.Art.32",
                framework=ComplianceFramework.GDPR,
                title="Security of processing",
                description="Implement appropriate technical and organizational measures",
                requirement="Ensure appropriate security of personal data processing",
                category="Security of Processing",
                subcategory="Technical Measures",
                implementation_guidance=[
                    "Implement data encryption",
                    "Ensure data confidentiality",
                    "Maintain data integrity",
                    "Ensure system availability"
                ],
                testing_procedures=[
                    "Review security measures",
                    "Test encryption implementation",
                    "Verify access controls",
                    "Check data integrity measures"
                ],
                evidence_requirements=[
                    "Security policy documentation",
                    "Encryption verification",
                    "Access control testing",
                    "Integrity checks"
                ],
                automated_tests=[
                    "data_encryption_verification",
                    "access_control_audit",
                    "integrity_check"
                ],
                manual_validation_required=True,
                severity=SeverityLevel.HIGH,
                references=["GDPR Article 32"]
            )
        ]

        # Add all controls to the main dictionary
        all_controls = pci_controls + hipaa_controls + sox_controls + iso_controls + gdpr_controls
        for control in all_controls:
            self.controls[control.control_id] = control

        logger.info(f"Initialized {len(all_controls)} compliance controls")

    async def _load_framework_definitions(self):
        """Load comprehensive framework definitions"""
        self.frameworks = {
            ComplianceFramework.PCI_DSS: {
                "name": "Payment Card Industry Data Security Standard",
                "version": "4.0",
                "description": "Security standard for organizations that handle payment card data",
                "scope": "Payment card data processing, storage, and transmission",
                "assessment_frequency": "Annual",
                "categories": [
                    "Build and Maintain a Secure Network",
                    "Protect Cardholder Data",
                    "Maintain a Vulnerability Management Program",
                    "Implement Strong Access Control Measures",
                    "Regularly Monitor and Test Networks",
                    "Maintain an Information Security Policy"
                ]
            },
            ComplianceFramework.HIPAA: {
                "name": "Health Insurance Portability and Accountability Act",
                "version": "Final Rule 2013",
                "description": "Privacy and security requirements for protected health information",
                "scope": "Healthcare organizations and business associates",
                "assessment_frequency": "Ongoing",
                "categories": [
                    "Administrative Safeguards",
                    "Physical Safeguards",
                    "Technical Safeguards"
                ]
            },
            ComplianceFramework.SOX: {
                "name": "Sarbanes-Oxley Act",
                "version": "2002",
                "description": "Financial reporting and corporate governance requirements",
                "scope": "Public companies and their auditors",
                "assessment_frequency": "Annual",
                "categories": [
                    "Corporate Responsibility",
                    "Auditor Independence",
                    "Corporate Responsibility",
                    "Enhanced Financial Disclosures"
                ]
            },
            ComplianceFramework.ISO_27001: {
                "name": "Information Security Management",
                "version": "2022",
                "description": "International standard for information security management systems",
                "scope": "Information security management",
                "assessment_frequency": "Annual with ongoing monitoring",
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
                    "System Acquisition, Development and Maintenance",
                    "Supplier Relationships",
                    "Information Security Incident Management",
                    "Information Security Aspects of Business Continuity Management",
                    "Compliance"
                ]
            },
            ComplianceFramework.GDPR: {
                "name": "General Data Protection Regulation",
                "version": "2018",
                "description": "Data protection and privacy regulation",
                "scope": "Organizations processing EU personal data",
                "assessment_frequency": "Ongoing",
                "categories": [
                    "Lawfulness of Processing",
                    "Data Subject Rights",
                    "Data Protection by Design",
                    "Security of Processing",
                    "Data Breach Notification",
                    "Data Protection Impact Assessment"
                ]
            }
        }

    async def _load_assessment_history(self):
        """Load historical assessment data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT assessment_id, control_id, framework, status, compliance_score,
                       findings, evidence_collected, gaps_identified, remediation_steps,
                       assessor, assessment_date, next_assessment_due, notes
                FROM control_assessments
                ORDER BY assessment_date DESC
            """)

            rows = cursor.fetchall()
            for row in rows:
                assessment = ControlAssessment(
                    assessment_id=row[0],
                    control_id=row[1],
                    framework=ComplianceFramework(row[2]),
                    status=ControlStatus(row[3]),
                    compliance_score=row[4] or 0.0,
                    findings=json.loads(row[5]) if row[5] else [],
                    evidence_collected=json.loads(row[6]) if row[6] else [],
                    gaps_identified=json.loads(row[7]) if row[7] else [],
                    remediation_steps=json.loads(row[8]) if row[8] else [],
                    assessor=row[9] or "",
                    assessment_date=datetime.fromisoformat(row[10]) if row[10] else datetime.now(),
                    next_assessment_due=datetime.fromisoformat(row[11]) if row[11] else datetime.now() + timedelta(days=365),
                    notes=row[12] or ""
                )
                self.assessments.append(assessment)

            conn.close()
            logger.info(f"Loaded {len(self.assessments)} historical assessments")

        except Exception as e:
            logger.error(f"Error loading assessment history: {e}")

    async def assess_compliance_control(self, control_id: str, assessor: str,
                                      evidence: List[str] = None,
                                      automated_test_results: Dict[str, Any] = None) -> ControlAssessment:
        """Assess a specific compliance control"""
        try:
            control = self.controls.get(control_id)
            if not control:
                raise ValueError(f"Control {control_id} not found")

            assessment_id = str(uuid.uuid4())
            findings = []
            gaps_identified = []
            remediation_steps = []
            evidence_collected = evidence or []

            # Perform automated tests if available
            automated_score = 0.0
            if automated_test_results:
                automated_score = await self._evaluate_automated_tests(control, automated_test_results)

                # Add automated findings
                for test_name, result in automated_test_results.items():
                    if result.get("status") == "fail":
                        findings.append(f"Automated test '{test_name}' failed: {result.get('message', '')}")
                        gaps_identified.append(f"Failed automated test: {test_name}")

            # Manual assessment scoring
            manual_score = await self._perform_manual_assessment(control, evidence_collected)

            # Calculate overall compliance score
            if control.manual_validation_required:
                compliance_score = (automated_score * 0.4) + (manual_score * 0.6)
            else:
                compliance_score = automated_score

            # Determine status
            if compliance_score >= 0.9:
                status = ControlStatus.COMPLIANT
            elif compliance_score >= 0.7:
                status = ControlStatus.PARTIALLY_COMPLIANT
            elif compliance_score >= 0.5:
                status = ControlStatus.REMEDIATION_REQUIRED
            else:
                status = ControlStatus.NON_COMPLIANT

            # Generate remediation steps for non-compliant controls
            if status in [ControlStatus.NON_COMPLIANT, ControlStatus.REMEDIATION_REQUIRED]:
                remediation_steps = await self._generate_remediation_steps(control, gaps_identified)

            # Create assessment
            assessment = ControlAssessment(
                assessment_id=assessment_id,
                control_id=control_id,
                framework=control.framework,
                status=status,
                compliance_score=compliance_score,
                findings=findings,
                evidence_collected=evidence_collected,
                gaps_identified=gaps_identified,
                remediation_steps=remediation_steps,
                assessor=assessor,
                assessment_date=datetime.now(),
                next_assessment_due=datetime.now() + timedelta(days=365),
                notes=""
            )

            # Store assessment
            await self._store_assessment(assessment)
            self.assessments.append(assessment)

            logger.info(f"Assessed control {control_id}: {status.value} (score: {compliance_score:.2f})")
            return assessment

        except Exception as e:
            logger.error(f"Control assessment failed: {e}")
            raise

    async def _evaluate_automated_tests(self, control: ComplianceControl,
                                       test_results: Dict[str, Any]) -> float:
        """Evaluate automated test results for a control"""
        if not control.automated_tests:
            return 1.0  # No automated tests means full score for this component

        total_tests = len(control.automated_tests)
        passed_tests = 0

        for test_name in control.automated_tests:
            if test_name in test_results:
                result = test_results[test_name]
                if result.get("status") == "pass":
                    passed_tests += 1
            else:
                # Test not run - partial credit
                passed_tests += 0.5

        return passed_tests / total_tests if total_tests > 0 else 1.0

    async def _perform_manual_assessment(self, control: ComplianceControl,
                                       evidence: List[str]) -> float:
        """Perform manual assessment based on evidence"""
        if not control.manual_validation_required:
            return 1.0

        required_evidence = len(control.evidence_requirements)
        provided_evidence = len(evidence)

        # Basic scoring based on evidence provision
        evidence_score = min(1.0, provided_evidence / required_evidence) if required_evidence > 0 else 1.0

        # Additional scoring based on evidence quality (simplified)
        quality_score = 1.0
        if evidence:
            # Check for key documents and artifacts
            key_documents = ["policy", "procedure", "documentation", "report", "log", "certificate"]
            evidence_text = " ".join(evidence).lower()

            found_documents = sum(1 for doc in key_documents if doc in evidence_text)
            quality_score = min(1.0, found_documents / len(key_documents))

        return (evidence_score * 0.7) + (quality_score * 0.3)

    async def _generate_remediation_steps(self, control: ComplianceControl,
                                        gaps: List[str]) -> List[str]:
        """Generate specific remediation steps for control gaps"""
        remediation_steps = []

        # Add control-specific remediation guidance
        remediation_steps.extend(control.implementation_guidance)

        # Add gap-specific remediation
        for gap in gaps:
            if "documentation" in gap.lower():
                remediation_steps.append("Create or update required documentation")
            elif "test" in gap.lower():
                remediation_steps.append("Implement automated testing procedures")
            elif "policy" in gap.lower():
                remediation_steps.append("Develop or revise security policies")
            elif "training" in gap.lower():
                remediation_steps.append("Provide staff training and awareness")
            elif "access" in gap.lower():
                remediation_steps.append("Review and update access controls")
            elif "encryption" in gap.lower():
                remediation_steps.append("Implement or strengthen encryption measures")

        # Add framework-specific remediation
        if control.framework == ComplianceFramework.PCI_DSS:
            remediation_steps.append("Engage QSA for formal assessment")
            remediation_steps.append("Review PCI DSS requirements guide")
        elif control.framework == ComplianceFramework.HIPAA:
            remediation_steps.append("Consult HIPAA compliance attorney")
            remediation_steps.append("Review HHS guidance documents")
        elif control.framework == ComplianceFramework.SOX:
            remediation_steps.append("Engage external auditor")
            remediation_steps.append("Review PCAOB guidance")

        return list(set(remediation_steps))  # Remove duplicates

    async def _store_assessment(self, assessment: ControlAssessment):
        """Store assessment in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO control_assessments
                (assessment_id, control_id, framework, status, compliance_score,
                 findings, evidence_collected, gaps_identified, remediation_steps,
                 assessor, assessment_date, next_assessment_due, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                assessment.assessment_id,
                assessment.control_id,
                assessment.framework.value,
                assessment.status.value,
                assessment.compliance_score,
                json.dumps(assessment.findings),
                json.dumps(assessment.evidence_collected),
                json.dumps(assessment.gaps_identified),
                json.dumps(assessment.remediation_steps),
                assessment.assessor,
                assessment.assessment_date.isoformat(),
                assessment.next_assessment_due.isoformat(),
                assessment.notes
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing assessment: {e}")

    async def assess_framework_compliance(self, framework: ComplianceFramework,
                                        assessor: str,
                                        organization: str = "Organization",
                                        scope: str = "Full Assessment") -> ComplianceReport:
        """Assess compliance across an entire framework"""
        try:
            logger.info(f"Starting comprehensive {framework.value} compliance assessment")

            # Get all controls for the framework
            framework_controls = [c for c in self.controls.values() if c.framework == framework]

            if not framework_controls:
                raise ValueError(f"No controls found for framework {framework.value}")

            # Assess each control
            control_assessments = []
            for control in framework_controls:
                try:
                    # Get existing assessment or create new one
                    existing_assessment = self._get_latest_assessment(control.control_id)

                    if existing_assessment and self._is_assessment_current(existing_assessment):
                        assessment = existing_assessment
                    else:
                        # Perform new assessment
                        assessment = await self.assess_compliance_control(
                            control.control_id,
                            assessor,
                            evidence=[],  # In production, evidence would be collected
                            automated_test_results={}  # In production, automated tests would run
                        )

                    control_assessments.append(assessment)

                except Exception as e:
                    logger.error(f"Error assessing control {control.control_id}: {e}")

            # Generate compliance report
            report = await self._generate_compliance_report(
                framework, control_assessments, organization, scope
            )

            # Store report
            await self._store_compliance_report(report)

            logger.info(f"{framework.value} compliance assessment completed: {report.overall_compliance_score:.1%} compliant")
            return report

        except Exception as e:
            logger.error(f"Framework compliance assessment failed: {e}")
            raise

    def _get_latest_assessment(self, control_id: str) -> Optional[ControlAssessment]:
        """Get the latest assessment for a control"""
        control_assessments = [a for a in self.assessments if a.control_id == control_id]
        if control_assessments:
            return max(control_assessments, key=lambda a: a.assessment_date)
        return None

    def _is_assessment_current(self, assessment: ControlAssessment) -> bool:
        """Check if an assessment is still current"""
        return datetime.now() < assessment.next_assessment_due

    async def _generate_compliance_report(self, framework: ComplianceFramework,
                                        assessments: List[ControlAssessment],
                                        organization: str,
                                        scope: str) -> ComplianceReport:
        """Generate comprehensive compliance report"""

        report_id = str(uuid.uuid4())
        total_controls = len(assessments)

        # Count controls by status
        compliant_controls = len([a for a in assessments if a.status == ControlStatus.COMPLIANT])
        non_compliant_controls = len([a for a in assessments if a.status == ControlStatus.NON_COMPLIANT])
        partially_compliant_controls = len([a for a in assessments if a.status == ControlStatus.PARTIALLY_COMPLIANT])

        # Calculate overall compliance score
        total_score = sum(a.compliance_score for a in assessments)
        overall_compliance_score = total_score / total_controls if total_controls > 0 else 0.0

        # Identify critical and high findings
        critical_findings = []
        high_findings = []

        for assessment in assessments:
            control = self.controls.get(assessment.control_id)
            if control and assessment.status in [ControlStatus.NON_COMPLIANT, ControlStatus.REMEDIATION_REQUIRED]:
                if control.severity == SeverityLevel.CRITICAL:
                    critical_findings.extend(assessment.findings)
                elif control.severity == SeverityLevel.HIGH:
                    high_findings.extend(assessment.findings)

        # Generate remediation priorities
        remediation_priorities = await self._prioritize_remediation(assessments)

        # Create report
        report = ComplianceReport(
            report_id=report_id,
            framework=framework,
            organization=organization,
            assessment_scope=scope,
            assessment_period=(
                min(a.assessment_date for a in assessments),
                max(a.assessment_date for a in assessments)
            ),
            overall_compliance_score=overall_compliance_score,
            total_controls=total_controls,
            compliant_controls=compliant_controls,
            non_compliant_controls=non_compliant_controls,
            partially_compliant_controls=partially_compliant_controls,
            critical_findings=critical_findings[:10],  # Limit to top 10
            high_findings=high_findings[:10],  # Limit to top 10
            remediation_priorities=remediation_priorities,
            assessments=assessments,
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(days=365)
        )

        return report

    async def _prioritize_remediation(self, assessments: List[ControlAssessment]) -> List[str]:
        """Prioritize remediation efforts based on risk and compliance impact"""
        priorities = []

        # Group by severity and compliance status
        critical_non_compliant = []
        high_non_compliant = []
        medium_non_compliant = []

        for assessment in assessments:
            if assessment.status in [ControlStatus.NON_COMPLIANT, ControlStatus.REMEDIATION_REQUIRED]:
                control = self.controls.get(assessment.control_id)
                if control:
                    if control.severity == SeverityLevel.CRITICAL:
                        critical_non_compliant.append(assessment)
                    elif control.severity == SeverityLevel.HIGH:
                        high_non_compliant.append(assessment)
                    else:
                        medium_non_compliant.append(assessment)

        # Generate prioritized remediation plan
        if critical_non_compliant:
            priorities.append(f"IMMEDIATE: Address {len(critical_non_compliant)} critical control deficiencies")
            for assessment in critical_non_compliant[:3]:  # Top 3
                control = self.controls.get(assessment.control_id)
                if control:
                    priorities.append(f"- {control.control_id}: {control.title}")

        if high_non_compliant:
            priorities.append(f"HIGH PRIORITY: Address {len(high_non_compliant)} high-priority control deficiencies")
            for assessment in high_non_compliant[:5]:  # Top 5
                control = self.controls.get(assessment.control_id)
                if control:
                    priorities.append(f"- {control.control_id}: {control.title}")

        if medium_non_compliant:
            priorities.append(f"MEDIUM PRIORITY: Address {len(medium_non_compliant)} medium-priority control deficiencies")

        return priorities

    async def _store_compliance_report(self, report: ComplianceReport):
        """Store compliance report in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO compliance_reports
                (report_id, framework, organization, assessment_scope,
                 start_date, end_date, overall_compliance_score,
                 total_controls, compliant_controls, non_compliant_controls,
                 partially_compliant_controls, critical_findings, high_findings,
                 remediation_priorities, generated_at, valid_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.framework.value,
                report.organization,
                report.assessment_scope,
                report.assessment_period[0].isoformat(),
                report.assessment_period[1].isoformat(),
                report.overall_compliance_score,
                report.total_controls,
                report.compliant_controls,
                report.non_compliant_controls,
                report.partially_compliant_controls,
                json.dumps(report.critical_findings),
                json.dumps(report.high_findings),
                json.dumps(report.remediation_priorities),
                report.generated_at.isoformat(),
                report.valid_until.isoformat()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing compliance report: {e}")

    async def generate_compliance_dashboard(self, frameworks: List[ComplianceFramework] = None) -> Dict[str, Any]:
        """Generate compliance dashboard with key metrics"""
        try:
            if not frameworks:
                frameworks = list(self.frameworks.keys())

            dashboard_data = {
                "compliance_summary": {},
                "framework_scores": {},
                "critical_gaps": [],
                "upcoming_assessments": [],
                "trending_metrics": {},
                "generated_at": datetime.now().isoformat()
            }

            for framework in frameworks:
                # Get latest assessments for framework
                framework_assessments = [
                    a for a in self.assessments
                    if a.framework == framework
                ]

                if not framework_assessments:
                    continue

                # Calculate framework metrics
                total_assessments = len(framework_assessments)
                compliant = len([a for a in framework_assessments if a.status == ControlStatus.COMPLIANT])
                non_compliant = len([a for a in framework_assessments if a.status == ControlStatus.NON_COMPLIANT])

                compliance_rate = compliant / total_assessments if total_assessments > 0 else 0
                avg_score = sum(a.compliance_score for a in framework_assessments) / total_assessments if total_assessments > 0 else 0

                dashboard_data["framework_scores"][framework.value] = {
                    "compliance_rate": compliance_rate,
                    "average_score": avg_score,
                    "total_controls": total_assessments,
                    "compliant_controls": compliant,
                    "non_compliant_controls": non_compliant
                }

                # Identify critical gaps
                critical_gaps = [
                    a for a in framework_assessments
                    if a.status == ControlStatus.NON_COMPLIANT and
                    self.controls.get(a.control_id, {}).severity == SeverityLevel.CRITICAL
                ]

                for gap in critical_gaps:
                    control = self.controls.get(gap.control_id)
                    if control:
                        dashboard_data["critical_gaps"].append({
                            "framework": framework.value,
                            "control_id": gap.control_id,
                            "control_title": control.title,
                            "severity": control.severity.value,
                            "assessment_date": gap.assessment_date.isoformat()
                        })

                # Upcoming assessments
                upcoming = [
                    a for a in framework_assessments
                    if a.next_assessment_due <= datetime.now() + timedelta(days=30)
                ]

                for assessment in upcoming:
                    control = self.controls.get(assessment.control_id)
                    if control:
                        dashboard_data["upcoming_assessments"].append({
                            "framework": framework.value,
                            "control_id": assessment.control_id,
                            "control_title": control.title,
                            "due_date": assessment.next_assessment_due.isoformat(),
                            "days_until_due": (assessment.next_assessment_due - datetime.now()).days
                        })

            # Overall compliance summary
            all_assessments = [a for a in self.assessments]
            if all_assessments:
                total_compliant = len([a for a in all_assessments if a.status == ControlStatus.COMPLIANT])
                total_assessments = len(all_assessments)

                dashboard_data["compliance_summary"] = {
                    "overall_compliance_rate": total_compliant / total_assessments if total_assessments > 0 else 0,
                    "total_frameworks": len(frameworks),
                    "total_controls_assessed": total_assessments,
                    "critical_gaps_count": len(dashboard_data["critical_gaps"]),
                    "upcoming_assessments_count": len(dashboard_data["upcoming_assessments"])
                }

            return dashboard_data

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return {"error": str(e)}

    async def export_compliance_data(self, format_type: str = "json",
                                   framework: ComplianceFramework = None) -> str:
        """Export compliance data in various formats"""
        try:
            # Filter assessments by framework if specified
            assessments_to_export = self.assessments
            if framework:
                assessments_to_export = [a for a in self.assessments if a.framework == framework]

            if format_type.lower() == "json":
                export_data = {
                    "framework": framework.value if framework else "all",
                    "export_date": datetime.now().isoformat(),
                    "assessments": [asdict(a) for a in assessments_to_export],
                    "controls": [asdict(c) for c in self.controls.values() if not framework or c.framework == framework]
                }
                return json.dumps(export_data, indent=2, default=str)

            elif format_type.lower() == "csv":
                # Generate CSV export
                csv_data = []
                csv_data.append([
                    "Control ID", "Framework", "Title", "Status", "Compliance Score",
                    "Assessment Date", "Assessor", "Findings Count", "Gaps Count"
                ])

                for assessment in assessments_to_export:
                    control = self.controls.get(assessment.control_id)
                    csv_data.append([
                        assessment.control_id,
                        assessment.framework.value,
                        control.title if control else "Unknown",
                        assessment.status.value,
                        f"{assessment.compliance_score:.2f}",
                        assessment.assessment_date.strftime("%Y-%m-%d"),
                        assessment.assessor,
                        len(assessment.findings),
                        len(assessment.gaps_identified)
                    ])

                # Convert to CSV string
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerows(csv_data)
                return output.getvalue()

            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return f"Export error: {e}"

# Global instance management
_compliance_validator: Optional[ComplianceFrameworkValidator] = None

async def get_compliance_validator() -> ComplianceFrameworkValidator:
    """Get global compliance validator instance"""
    global _compliance_validator

    if _compliance_validator is None:
        _compliance_validator = ComplianceFrameworkValidator()
        await _compliance_validator.initialize()

    return _compliance_validator
