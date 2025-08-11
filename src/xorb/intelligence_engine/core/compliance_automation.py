#!/usr/bin/env python3
"""
Enterprise Compliance Automation Engine
Advanced compliance validation, monitoring, and reporting system
"""

import asyncio
import logging
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
import uuid
import re
import os
import sys

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'api', 'app'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'intelligence'))

try:
    from advanced_llm_orchestrator import (
        get_llm_orchestrator, AIDecisionRequest, DecisionDomain, 
        DecisionComplexity, validate_compliance_controls
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from services.base_service import ComplianceService, ServiceHealth, ServiceStatus
except ImportError:
    ComplianceService = None
    ServiceHealth = None
    ServiceStatus = None

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    SOC2_TYPE2 = "soc2_type2"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    ISO_27001 = "iso_27001"
    NIST_CSF = "nist_csf"
    SOX = "sox"
    FISMA = "fisma"
    FedRAMP = "fedramp"
    CIS_CONTROLS = "cis_controls"
    CCPA = "ccpa"
    PIPEDA = "pipeda"

class ControlStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_TESTED = "not_tested"
    EXCEPTION_APPROVED = "exception_approved"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"

class ControlCategory(Enum):
    ACCESS_CONTROL = "access_control"
    AUDIT_LOGGING = "audit_logging"
    ENCRYPTION = "encryption"
    NETWORK_SECURITY = "network_security"
    INCIDENT_RESPONSE = "incident_response"
    RISK_MANAGEMENT = "risk_management"
    VULNERABILITY_MANAGEMENT = "vulnerability_management"
    CHANGE_MANAGEMENT = "change_management"
    BUSINESS_CONTINUITY = "business_continuity"
    PHYSICAL_SECURITY = "physical_security"
    DATA_PROTECTION = "data_protection"
    TRAINING_AWARENESS = "training_awareness"

class AssessmentType(Enum):
    AUTOMATED = "automated"
    MANUAL = "manual"
    HYBRID = "hybrid"
    CONTINUOUS = "continuous"

@dataclass
class ComplianceControl:
    """Individual compliance control definition"""
    control_id: str
    framework: ComplianceFramework
    category: ControlCategory
    title: str
    description: str
    requirements: List[str]
    implementation_guidance: List[str]
    testing_procedures: List[str]
    evidence_requirements: List[str]
    automation_possible: bool
    risk_level: str
    frequency: str  # daily, weekly, monthly, quarterly, annually
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class ControlAssessment:
    """Assessment result for a specific control"""
    assessment_id: str
    control_id: str
    framework: ComplianceFramework
    status: ControlStatus
    assessment_type: AssessmentType
    assessed_at: datetime
    assessed_by: str
    evidence: List[Dict[str, Any]]
    findings: List[str]
    exceptions: List[str]
    remediation_actions: List[str]
    next_assessment_due: Optional[datetime]
    confidence_score: float
    automation_coverage: float
    
    def __post_init__(self):
        if self.next_assessment_due is None:
            # Default to quarterly assessment
            self.next_assessment_due = self.assessed_at + timedelta(days=90)

@dataclass
class ComplianceReport:
    """Comprehensive compliance assessment report"""
    report_id: str
    framework: ComplianceFramework
    reporting_period: Tuple[datetime, datetime]
    overall_compliance_score: float
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    control_assessments: List[ControlAssessment]
    key_findings: List[str]
    recommendations: List[str]
    executive_summary: str
    generated_at: datetime
    generated_by: str
    status: str = "draft"

class ComplianceControlValidator:
    """Validates compliance controls through automated and manual testing"""
    
    def __init__(self):
        self.validators = {}
        self.evidence_collectors = {}
        self._register_validators()
    
    def _register_validators(self):
        """Register control validators for different categories"""
        self.validators[ControlCategory.ACCESS_CONTROL] = self._validate_access_control
        self.validators[ControlCategory.AUDIT_LOGGING] = self._validate_audit_logging
        self.validators[ControlCategory.ENCRYPTION] = self._validate_encryption
        self.validators[ControlCategory.NETWORK_SECURITY] = self._validate_network_security
        self.validators[ControlCategory.INCIDENT_RESPONSE] = self._validate_incident_response
        self.validators[ControlCategory.VULNERABILITY_MANAGEMENT] = self._validate_vulnerability_management
        self.validators[ControlCategory.CHANGE_MANAGEMENT] = self._validate_change_management
        self.validators[ControlCategory.DATA_PROTECTION] = self._validate_data_protection
        
        # Evidence collectors
        self.evidence_collectors[ControlCategory.ACCESS_CONTROL] = self._collect_access_control_evidence
        self.evidence_collectors[ControlCategory.AUDIT_LOGGING] = self._collect_audit_logging_evidence
        self.evidence_collectors[ControlCategory.ENCRYPTION] = self._collect_encryption_evidence
        self.evidence_collectors[ControlCategory.NETWORK_SECURITY] = self._collect_network_security_evidence
    
    async def validate_control(self, control: ComplianceControl, context: Dict[str, Any]) -> ControlAssessment:
        """Validate a single compliance control"""
        try:
            assessment_id = f"assess_{control.control_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Validating control {control.control_id}: {control.title}")
            
            # Collect evidence
            evidence = await self._collect_evidence(control, context)
            
            # Run validation
            validator = self.validators.get(control.category)
            if validator:
                validation_result = await validator(control, evidence, context)
            else:
                validation_result = await self._default_validation(control, evidence, context)
            
            # Determine status
            status = self._determine_control_status(validation_result)
            
            # Generate findings and recommendations
            findings = validation_result.get("findings", [])
            remediation_actions = validation_result.get("remediation_actions", [])
            
            # Calculate confidence score
            confidence_score = validation_result.get("confidence", 0.7)
            automation_coverage = validation_result.get("automation_coverage", 0.5)
            
            assessment = ControlAssessment(
                assessment_id=assessment_id,
                control_id=control.control_id,
                framework=control.framework,
                status=status,
                assessment_type=AssessmentType.AUTOMATED if control.automation_possible else AssessmentType.MANUAL,
                assessed_at=datetime.utcnow(),
                assessed_by="compliance_automation_engine",
                evidence=evidence,
                findings=findings,
                exceptions=[],
                remediation_actions=remediation_actions,
                confidence_score=confidence_score,
                automation_coverage=automation_coverage
            )
            
            logger.info(f"Control {control.control_id} assessed as {status.value}")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to validate control {control.control_id}: {e}")
            return ControlAssessment(
                assessment_id=f"assess_error_{control.control_id}",
                control_id=control.control_id,
                framework=control.framework,
                status=ControlStatus.NOT_TESTED,
                assessment_type=AssessmentType.MANUAL,
                assessed_at=datetime.utcnow(),
                assessed_by="compliance_automation_engine",
                evidence=[],
                findings=[f"Assessment failed: {str(e)}"],
                exceptions=[],
                remediation_actions=["Manual assessment required"],
                confidence_score=0.0,
                automation_coverage=0.0
            )
    
    async def _collect_evidence(self, control: ComplianceControl, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect evidence for control validation"""
        evidence = []
        
        try:
            collector = self.evidence_collectors.get(control.category)
            if collector:
                evidence = await collector(control, context)
            else:
                # Default evidence collection
                evidence = await self._collect_default_evidence(control, context)
            
            # Add metadata to evidence
            for item in evidence:
                item["collected_at"] = datetime.utcnow().isoformat()
                item["control_id"] = control.control_id
                item["evidence_type"] = item.get("type", "unknown")
            
            return evidence
            
        except Exception as e:
            logger.error(f"Evidence collection failed for {control.control_id}: {e}")
            return [{
                "type": "error",
                "description": f"Evidence collection failed: {str(e)}",
                "collected_at": datetime.utcnow().isoformat()
            }]
    
    def _determine_control_status(self, validation_result: Dict[str, Any]) -> ControlStatus:
        """Determine control status based on validation results"""
        compliance_score = validation_result.get("compliance_score", 0.0)
        
        if compliance_score >= 0.95:
            return ControlStatus.COMPLIANT
        elif compliance_score >= 0.8:
            return ControlStatus.PARTIALLY_COMPLIANT
        elif compliance_score >= 0.5:
            return ControlStatus.REMEDIATION_IN_PROGRESS
        else:
            return ControlStatus.NON_COMPLIANT
    
    # Control-specific validators
    
    async def _validate_access_control(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate access control requirements"""
        findings = []
        remediation_actions = []
        compliance_score = 0.8  # Default score
        
        # Check for multi-factor authentication
        mfa_evidence = [e for e in evidence if e.get("type") == "mfa_configuration"]
        if mfa_evidence:
            mfa_enabled = any(e.get("enabled", False) for e in mfa_evidence)
            if mfa_enabled:
                compliance_score += 0.1
            else:
                findings.append("Multi-factor authentication not enabled for all users")
                remediation_actions.append("Enable MFA for all user accounts")
        
        # Check access reviews
        access_review_evidence = [e for e in evidence if e.get("type") == "access_review"]
        if access_review_evidence:
            recent_reviews = [e for e in access_review_evidence 
                            if datetime.fromisoformat(e.get("last_review", "2020-01-01T00:00:00")) > datetime.utcnow() - timedelta(days=90)]
            if recent_reviews:
                compliance_score += 0.05
            else:
                findings.append("Access reviews not conducted within required timeframe")
                remediation_actions.append("Conduct quarterly access reviews")
        
        # Check privileged access management
        pam_evidence = [e for e in evidence if e.get("type") == "privileged_access"]
        if pam_evidence:
            pam_controls = any(e.get("controlled", False) for e in pam_evidence)
            if pam_controls:
                compliance_score += 0.05
            else:
                findings.append("Privileged access not properly controlled")
                remediation_actions.append("Implement privileged access management solution")
        
        return {
            "compliance_score": min(1.0, compliance_score),
            "findings": findings,
            "remediation_actions": remediation_actions,
            "confidence": 0.85,
            "automation_coverage": 0.7
        }
    
    async def _validate_audit_logging(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate audit logging requirements"""
        findings = []
        remediation_actions = []
        compliance_score = 0.7
        
        # Check log retention
        retention_evidence = [e for e in evidence if e.get("type") == "log_retention"]
        if retention_evidence:
            retention_days = max(e.get("retention_days", 0) for e in retention_evidence)
            if retention_days >= 365:  # 1 year retention
                compliance_score += 0.1
            elif retention_days >= 90:  # 3 months retention
                compliance_score += 0.05
            else:
                findings.append("Log retention period insufficient")
                remediation_actions.append("Configure log retention for minimum 1 year")
        
        # Check log integrity
        integrity_evidence = [e for e in evidence if e.get("type") == "log_integrity"]
        if integrity_evidence:
            integrity_protected = any(e.get("protected", False) for e in integrity_evidence)
            if integrity_protected:
                compliance_score += 0.1
            else:
                findings.append("Log integrity protection not implemented")
                remediation_actions.append("Implement log signing and tamper detection")
        
        # Check monitoring coverage
        coverage_evidence = [e for e in evidence if e.get("type") == "monitoring_coverage"]
        if coverage_evidence:
            coverage_percentage = max(e.get("coverage", 0) for e in coverage_evidence)
            if coverage_percentage >= 90:
                compliance_score += 0.1
            elif coverage_percentage >= 70:
                compliance_score += 0.05
            else:
                findings.append("Insufficient monitoring coverage")
                remediation_actions.append("Expand monitoring to cover all critical systems")
        
        return {
            "compliance_score": min(1.0, compliance_score),
            "findings": findings,
            "remediation_actions": remediation_actions,
            "confidence": 0.8,
            "automation_coverage": 0.9
        }
    
    async def _validate_encryption(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate encryption requirements"""
        findings = []
        remediation_actions = []
        compliance_score = 0.6
        
        # Check data at rest encryption
        rest_encryption_evidence = [e for e in evidence if e.get("type") == "encryption_at_rest"]
        if rest_encryption_evidence:
            strong_encryption = any(e.get("algorithm") in ["AES-256", "ChaCha20"] for e in rest_encryption_evidence)
            if strong_encryption:
                compliance_score += 0.2
            else:
                findings.append("Weak encryption algorithms in use")
                remediation_actions.append("Upgrade to AES-256 or equivalent strong encryption")
        
        # Check data in transit encryption
        transit_encryption_evidence = [e for e in evidence if e.get("type") == "encryption_in_transit"]
        if transit_encryption_evidence:
            tls_version = max(e.get("tls_version", "1.0") for e in transit_encryption_evidence)
            if tls_version >= "1.3":
                compliance_score += 0.15
            elif tls_version >= "1.2":
                compliance_score += 0.1
            else:
                findings.append("Outdated TLS version in use")
                remediation_actions.append("Upgrade to TLS 1.3 for all communications")
        
        # Check key management
        key_mgmt_evidence = [e for e in evidence if e.get("type") == "key_management"]
        if key_mgmt_evidence:
            hsm_used = any(e.get("hsm_protected", False) for e in key_mgmt_evidence)
            if hsm_used:
                compliance_score += 0.05
            else:
                findings.append("Encryption keys not HSM protected")
                remediation_actions.append("Implement HSM for key protection")
        
        return {
            "compliance_score": min(1.0, compliance_score),
            "findings": findings,
            "remediation_actions": remediation_actions,
            "confidence": 0.75,
            "automation_coverage": 0.6
        }
    
    async def _validate_network_security(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate network security requirements"""
        findings = []
        remediation_actions = []
        compliance_score = 0.7
        
        # Check firewall configuration
        firewall_evidence = [e for e in evidence if e.get("type") == "firewall_rules"]
        if firewall_evidence:
            deny_by_default = any(e.get("default_deny", False) for e in firewall_evidence)
            if deny_by_default:
                compliance_score += 0.1
            else:
                findings.append("Firewall not configured with default deny policy")
                remediation_actions.append("Configure firewall with default deny-all policy")
        
        # Check network segmentation
        segmentation_evidence = [e for e in evidence if e.get("type") == "network_segmentation"]
        if segmentation_evidence:
            segments = max(e.get("segments", 0) for e in segmentation_evidence)
            if segments >= 3:  # DMZ, Internal, Management
                compliance_score += 0.1
            else:
                findings.append("Insufficient network segmentation")
                remediation_actions.append("Implement proper network segmentation")
        
        # Check intrusion detection
        ids_evidence = [e for e in evidence if e.get("type") == "intrusion_detection"]
        if ids_evidence:
            ids_active = any(e.get("active", False) for e in ids_evidence)
            if ids_active:
                compliance_score += 0.1
            else:
                findings.append("Intrusion detection system not active")
                remediation_actions.append("Deploy and configure IDS/IPS")
        
        return {
            "compliance_score": min(1.0, compliance_score),
            "findings": findings,
            "remediation_actions": remediation_actions,
            "confidence": 0.8,
            "automation_coverage": 0.75
        }
    
    async def _validate_incident_response(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incident response requirements"""
        findings = []
        remediation_actions = []
        compliance_score = 0.6
        
        # Check incident response plan
        plan_evidence = [e for e in evidence if e.get("type") == "incident_response_plan"]
        if plan_evidence:
            plan_updated = any(
                datetime.fromisoformat(e.get("last_updated", "2020-01-01T00:00:00")) > datetime.utcnow() - timedelta(days=365)
                for e in plan_evidence
            )
            if plan_updated:
                compliance_score += 0.2
            else:
                findings.append("Incident response plan not updated within the last year")
                remediation_actions.append("Update incident response plan annually")
        
        # Check response team training
        training_evidence = [e for e in evidence if e.get("type") == "response_team_training"]
        if training_evidence:
            recent_training = any(
                datetime.fromisoformat(e.get("last_training", "2020-01-01T00:00:00")) > datetime.utcnow() - timedelta(days=180)
                for e in training_evidence
            )
            if recent_training:
                compliance_score += 0.1
            else:
                findings.append("Response team training not conducted recently")
                remediation_actions.append("Conduct bi-annual response team training")
        
        # Check incident tracking
        tracking_evidence = [e for e in evidence if e.get("type") == "incident_tracking"]
        if tracking_evidence:
            tracking_system = any(e.get("system_in_place", False) for e in tracking_evidence)
            if tracking_system:
                compliance_score += 0.1
            else:
                findings.append("No incident tracking system in place")
                remediation_actions.append("Implement incident tracking and management system")
        
        return {
            "compliance_score": min(1.0, compliance_score),
            "findings": findings,
            "remediation_actions": remediation_actions,
            "confidence": 0.7,
            "automation_coverage": 0.4
        }
    
    async def _validate_vulnerability_management(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate vulnerability management requirements"""
        findings = []
        remediation_actions = []
        compliance_score = 0.7
        
        # Check scanning frequency
        scan_evidence = [e for e in evidence if e.get("type") == "vulnerability_scans"]
        if scan_evidence:
            last_scan = max(
                datetime.fromisoformat(e.get("last_scan", "2020-01-01T00:00:00"))
                for e in scan_evidence
            )
            days_since_scan = (datetime.utcnow() - last_scan).days
            
            if days_since_scan <= 7:  # Weekly scans
                compliance_score += 0.15
            elif days_since_scan <= 30:  # Monthly scans
                compliance_score += 0.1
            else:
                findings.append("Vulnerability scans not conducted regularly")
                remediation_actions.append("Implement weekly vulnerability scanning")
        
        # Check patch management
        patch_evidence = [e for e in evidence if e.get("type") == "patch_management"]
        if patch_evidence:
            critical_patch_time = min(e.get("critical_patch_days", 999) for e in patch_evidence)
            if critical_patch_time <= 30:
                compliance_score += 0.1
            else:
                findings.append("Critical patches not applied within required timeframe")
                remediation_actions.append("Establish 30-day critical patch SLA")
        
        # Check remediation tracking
        remediation_evidence = [e for e in evidence if e.get("type") == "remediation_tracking"]
        if remediation_evidence:
            tracking_coverage = max(e.get("coverage", 0) for e in remediation_evidence)
            if tracking_coverage >= 90:
                compliance_score += 0.05
            else:
                findings.append("Incomplete vulnerability remediation tracking")
                remediation_actions.append("Implement comprehensive vulnerability tracking")
        
        return {
            "compliance_score": min(1.0, compliance_score),
            "findings": findings,
            "remediation_actions": remediation_actions,
            "confidence": 0.85,
            "automation_coverage": 0.8
        }
    
    async def _validate_change_management(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate change management requirements"""
        findings = []
        remediation_actions = []
        compliance_score = 0.6
        
        # Check change approval process
        approval_evidence = [e for e in evidence if e.get("type") == "change_approval"]
        if approval_evidence:
            approval_required = any(e.get("approval_required", False) for e in approval_evidence)
            if approval_required:
                compliance_score += 0.2
            else:
                findings.append("Change approval process not enforced")
                remediation_actions.append("Implement mandatory change approval workflow")
        
        # Check change documentation
        doc_evidence = [e for e in evidence if e.get("type") == "change_documentation"]
        if doc_evidence:
            documentation_complete = any(e.get("complete", False) for e in doc_evidence)
            if documentation_complete:
                compliance_score += 0.1
            else:
                findings.append("Change documentation incomplete")
                remediation_actions.append("Standardize change documentation requirements")
        
        # Check rollback procedures
        rollback_evidence = [e for e in evidence if e.get("type") == "rollback_procedures"]
        if rollback_evidence:
            rollback_tested = any(e.get("tested", False) for e in rollback_evidence)
            if rollback_tested:
                compliance_score += 0.1
            else:
                findings.append("Rollback procedures not tested")
                remediation_actions.append("Test rollback procedures for all critical changes")
        
        return {
            "compliance_score": min(1.0, compliance_score),
            "findings": findings,
            "remediation_actions": remediation_actions,
            "confidence": 0.75,
            "automation_coverage": 0.5
        }
    
    async def _validate_data_protection(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data protection requirements"""
        findings = []
        remediation_actions = []
        compliance_score = 0.7
        
        # Check data classification
        classification_evidence = [e for e in evidence if e.get("type") == "data_classification"]
        if classification_evidence:
            classification_complete = any(e.get("complete", False) for e in classification_evidence)
            if classification_complete:
                compliance_score += 0.1
            else:
                findings.append("Data classification not complete")
                remediation_actions.append("Complete data classification for all datasets")
        
        # Check data retention policies
        retention_evidence = [e for e in evidence if e.get("type") == "data_retention"]
        if retention_evidence:
            policies_defined = any(e.get("policies_defined", False) for e in retention_evidence)
            if policies_defined:
                compliance_score += 0.1
            else:
                findings.append("Data retention policies not defined")
                remediation_actions.append("Define and implement data retention policies")
        
        # Check data disposal
        disposal_evidence = [e for e in evidence if e.get("type") == "data_disposal"]
        if disposal_evidence:
            secure_disposal = any(e.get("secure", False) for e in disposal_evidence)
            if secure_disposal:
                compliance_score += 0.1
            else:
                findings.append("Secure data disposal procedures not implemented")
                remediation_actions.append("Implement secure data disposal procedures")
        
        return {
            "compliance_score": min(1.0, compliance_score),
            "findings": findings,
            "remediation_actions": remediation_actions,
            "confidence": 0.8,
            "automation_coverage": 0.6
        }
    
    async def _default_validation(self, control: ComplianceControl, evidence: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Default validation for controls without specific validators"""
        return {
            "compliance_score": 0.5,  # Neutral score requiring manual review
            "findings": ["Manual review required - no automated validation available"],
            "remediation_actions": ["Implement manual assessment procedures"],
            "confidence": 0.3,
            "automation_coverage": 0.0
        }
    
    # Evidence collection methods
    
    async def _collect_access_control_evidence(self, control: ComplianceControl, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect access control evidence"""
        evidence = []
        
        # Simulate MFA configuration check
        evidence.append({
            "type": "mfa_configuration",
            "enabled": True,  # Simulated result
            "coverage": 85,  # Percentage of users with MFA
            "source": "identity_provider"
        })
        
        # Simulate access review evidence
        evidence.append({
            "type": "access_review",
            "last_review": (datetime.utcnow() - timedelta(days=45)).isoformat(),
            "review_type": "quarterly",
            "findings_count": 3,
            "source": "access_management_system"
        })
        
        # Simulate privileged access evidence
        evidence.append({
            "type": "privileged_access",
            "controlled": True,
            "pam_solution": "CyberArk",
            "session_recording": True,
            "source": "privileged_access_manager"
        })
        
        return evidence
    
    async def _collect_audit_logging_evidence(self, control: ComplianceControl, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect audit logging evidence"""
        evidence = []
        
        # Log retention evidence
        evidence.append({
            "type": "log_retention",
            "retention_days": 2555,  # 7 years
            "storage_type": "immutable",
            "compression": True,
            "source": "log_management_system"
        })
        
        # Log integrity evidence
        evidence.append({
            "type": "log_integrity",
            "protected": True,
            "signature_algorithm": "SHA-256",
            "tamper_detection": True,
            "source": "siem"
        })
        
        # Monitoring coverage evidence
        evidence.append({
            "type": "monitoring_coverage",
            "coverage": 92,  # Percentage
            "monitored_systems": 156,
            "total_systems": 170,
            "source": "monitoring_platform"
        })
        
        return evidence
    
    async def _collect_encryption_evidence(self, control: ComplianceControl, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect encryption evidence"""
        evidence = []
        
        # Encryption at rest
        evidence.append({
            "type": "encryption_at_rest",
            "algorithm": "AES-256",
            "key_size": 256,
            "fips_140_2": True,
            "source": "database_encryption"
        })
        
        # Encryption in transit
        evidence.append({
            "type": "encryption_in_transit",
            "tls_version": "1.3",
            "cipher_suites": ["TLS_AES_256_GCM_SHA384"],
            "certificate_validation": True,
            "source": "network_scanner"
        })
        
        # Key management
        evidence.append({
            "type": "key_management",
            "hsm_protected": True,
            "key_rotation": True,
            "rotation_frequency": "quarterly",
            "source": "key_management_service"
        })
        
        return evidence
    
    async def _collect_network_security_evidence(self, control: ComplianceControl, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect network security evidence"""
        evidence = []
        
        # Firewall rules
        evidence.append({
            "type": "firewall_rules",
            "default_deny": True,
            "rule_count": 234,
            "last_review": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "source": "firewall_manager"
        })
        
        # Network segmentation
        evidence.append({
            "type": "network_segmentation",
            "segments": 5,  # DMZ, Internal, Management, Guest, Quarantine
            "vlan_isolation": True,
            "micro_segmentation": True,
            "source": "network_analyzer"
        })
        
        # Intrusion detection
        evidence.append({
            "type": "intrusion_detection",
            "active": True,
            "signature_updates": "daily",
            "alert_tuning": "quarterly",
            "source": "ids_system"
        })
        
        return evidence
    
    async def _collect_default_evidence(self, control: ComplianceControl, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Default evidence collection for controls without specific collectors"""
        return [{
            "type": "manual_review_required",
            "description": f"Manual evidence collection required for {control.category.value}",
            "control_id": control.control_id,
            "requires_human_validation": True
        }]

class EnterpriseComplianceAutomation(ComplianceService if ComplianceService else object):
    """Enterprise-grade compliance automation and monitoring system"""
    
    def __init__(self, **kwargs):
        if ComplianceService:
            super().__init__(**kwargs)
        
        self.frameworks: Dict[ComplianceFramework, Dict[str, ComplianceControl]] = {}
        self.assessments: Dict[str, ControlAssessment] = {}
        self.reports: Dict[str, ComplianceReport] = {}
        self.validator = ComplianceControlValidator()
        self.assessment_schedule: Dict[str, datetime] = {}
        self.continuous_monitoring = False
        
        # AI integration
        self.ai_available = LLM_AVAILABLE
        
        if hasattr(super(), 'logger'):
            self.logger = super().logger
        else:
            self.logger = logging.getLogger("EnterpriseComplianceAutomation")
    
    async def initialize(self) -> bool:
        """Initialize the compliance automation engine"""
        try:
            self.logger.info("Initializing Enterprise Compliance Automation Engine...")
            
            # Load compliance frameworks
            await self._load_compliance_frameworks()
            
            # Schedule assessments
            await self._schedule_assessments()
            
            # Start continuous monitoring if enabled
            if self.continuous_monitoring:
                asyncio.create_task(self._continuous_monitoring_loop())
            
            self.logger.info(f"Compliance Automation initialized with {len(self.frameworks)} frameworks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Compliance Automation: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the compliance automation engine"""
        try:
            self.logger.info("Shutting down Compliance Automation Engine...")
            self.continuous_monitoring = False
            self.logger.info("Compliance Automation shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown Compliance Automation: {e}")
            return False
    
    async def _load_compliance_frameworks(self):
        """Load compliance framework definitions"""
        
        # SOC 2 Type II Framework
        soc2_controls = {
            "CC6.1": ComplianceControl(
                control_id="CC6.1",
                framework=ComplianceFramework.SOC2_TYPE2,
                category=ControlCategory.ACCESS_CONTROL,
                title="Logical and Physical Access Controls",
                description="The entity implements logical and physical access controls to protect against threats from external and internal sources",
                requirements=[
                    "Multi-factor authentication for privileged users",
                    "Regular access reviews",
                    "Principle of least privilege",
                    "Physical access controls to data centers"
                ],
                implementation_guidance=[
                    "Implement MFA for all administrative accounts",
                    "Conduct quarterly access reviews",
                    "Use role-based access control (RBAC)",
                    "Implement biometric controls for sensitive areas"
                ],
                testing_procedures=[
                    "Review MFA configuration",
                    "Test access review process",
                    "Validate RBAC implementation",
                    "Inspect physical security controls"
                ],
                evidence_requirements=[
                    "MFA configuration screenshots",
                    "Access review reports",
                    "RBAC matrix",
                    "Physical security assessment"
                ],
                automation_possible=True,
                risk_level="high",
                frequency="quarterly"
            ),
            "CC6.7": ComplianceControl(
                control_id="CC6.7",
                framework=ComplianceFramework.SOC2_TYPE2,
                category=ControlCategory.AUDIT_LOGGING,
                title="System Activities Monitoring",
                description="The entity restricts the transmission, movement, and removal of information to authorized internal and external users",
                requirements=[
                    "Comprehensive audit logging",
                    "Log integrity protection",
                    "Real-time monitoring",
                    "Incident detection and response"
                ],
                implementation_guidance=[
                    "Enable audit logging on all systems",
                    "Implement log signing and tamper protection",
                    "Deploy SIEM for real-time monitoring",
                    "Establish incident response procedures"
                ],
                testing_procedures=[
                    "Review audit log configurations",
                    "Test log integrity controls",
                    "Validate SIEM alerting",
                    "Test incident response procedures"
                ],
                evidence_requirements=[
                    "Audit log configurations",
                    "Log integrity verification",
                    "SIEM alert reports",
                    "Incident response documentation"
                ],
                automation_possible=True,
                risk_level="high",
                frequency="monthly"
            ),
            "CC6.8": ComplianceControl(
                control_id="CC6.8",
                framework=ComplianceFramework.SOC2_TYPE2,
                category=ControlCategory.ENCRYPTION,
                title="Data Transmission and Disposal",
                description="The entity implements controls to protect data during transmission and disposal",
                requirements=[
                    "Encryption of data in transit",
                    "Encryption of data at rest",
                    "Secure key management",
                    "Secure data disposal"
                ],
                implementation_guidance=[
                    "Use TLS 1.3 for all communications",
                    "Implement AES-256 for data at rest",
                    "Use HSM for key management",
                    "Implement secure data wiping procedures"
                ],
                testing_procedures=[
                    "Test TLS configuration",
                    "Verify encryption at rest",
                    "Review key management procedures",
                    "Test data disposal procedures"
                ],
                evidence_requirements=[
                    "TLS configuration reports",
                    "Encryption verification",
                    "Key management documentation",
                    "Data disposal certificates"
                ],
                automation_possible=True,
                risk_level="high",
                frequency="quarterly"
            )
        }
        
        # PCI DSS Framework
        pci_controls = {
            "REQ1": ComplianceControl(
                control_id="REQ1",
                framework=ComplianceFramework.PCI_DSS,
                category=ControlCategory.NETWORK_SECURITY,
                title="Install and maintain network security controls",
                description="Network security controls are in place to protect the cardholder data environment",
                requirements=[
                    "Firewall configuration standards",
                    "Network segmentation",
                    "DMZ implementation",
                    "Wireless security"
                ],
                implementation_guidance=[
                    "Implement firewall rules with default deny",
                    "Segment cardholder data environment",
                    "Deploy DMZ for public services",
                    "Secure wireless networks with WPA3"
                ],
                testing_procedures=[
                    "Review firewall configurations",
                    "Test network segmentation",
                    "Validate DMZ implementation",
                    "Test wireless security"
                ],
                evidence_requirements=[
                    "Firewall rule documentation",
                    "Network diagrams",
                    "Penetration test reports",
                    "Wireless security configuration"
                ],
                automation_possible=True,
                risk_level="critical",
                frequency="quarterly"
            ),
            "REQ3": ComplianceControl(
                control_id="REQ3",
                framework=ComplianceFramework.PCI_DSS,
                category=ControlCategory.ENCRYPTION,
                title="Protect stored cardholder data",
                description="Cardholder data is protected wherever it is stored",
                requirements=[
                    "Data encryption at rest",
                    "Encryption key management",
                    "Data masking",
                    "Secure data retention"
                ],
                implementation_guidance=[
                    "Encrypt all cardholder data with AES-256",
                    "Use HSM for key management",
                    "Mask PAN in applications",
                    "Implement data retention policies"
                ],
                testing_procedures=[
                    "Verify encryption implementation",
                    "Test key management procedures",
                    "Review data masking",
                    "Validate retention policies"
                ],
                evidence_requirements=[
                    "Encryption configuration",
                    "Key management documentation",
                    "Data masking verification",
                    "Retention policy documentation"
                ],
                automation_possible=True,
                risk_level="critical",
                frequency="quarterly",
                dependencies=["REQ1"]
            )
        }
        
        # GDPR Framework
        gdpr_controls = {
            "ART25": ComplianceControl(
                control_id="ART25",
                framework=ComplianceFramework.GDPR,
                category=ControlCategory.DATA_PROTECTION,
                title="Data protection by design and by default",
                description="Data protection principles are implemented by design and by default",
                requirements=[
                    "Privacy by design",
                    "Data minimization",
                    "Purpose limitation",
                    "Storage limitation"
                ],
                implementation_guidance=[
                    "Implement privacy controls in system design",
                    "Collect only necessary data",
                    "Define clear data purposes",
                    "Implement data retention schedules"
                ],
                testing_procedures=[
                    "Review privacy impact assessments",
                    "Validate data minimization",
                    "Test purpose limitation controls",
                    "Review retention schedules"
                ],
                evidence_requirements=[
                    "Privacy impact assessments",
                    "Data flow diagrams",
                    "Purpose documentation",
                    "Retention schedules"
                ],
                automation_possible=False,
                risk_level="high",
                frequency="annually"
            ),
            "ART32": ComplianceControl(
                control_id="ART32",
                framework=ComplianceFramework.GDPR,
                category=ControlCategory.ENCRYPTION,
                title="Security of processing",
                description="Appropriate technical and organizational measures ensure security of processing",
                requirements=[
                    "Encryption of personal data",
                    "Ongoing confidentiality",
                    "Integrity protection",
                    "Availability assurance"
                ],
                implementation_guidance=[
                    "Encrypt all personal data",
                    "Implement access controls",
                    "Deploy integrity monitoring",
                    "Ensure high availability"
                ],
                testing_procedures=[
                    "Test encryption implementation",
                    "Review access controls",
                    "Validate integrity monitoring",
                    "Test availability controls"
                ],
                evidence_requirements=[
                    "Encryption verification",
                    "Access control documentation",
                    "Integrity monitoring reports",
                    "Availability metrics"
                ],
                automation_possible=True,
                risk_level="high",
                frequency="quarterly"
            )
        }
        
        # Store frameworks
        self.frameworks[ComplianceFramework.SOC2_TYPE2] = soc2_controls
        self.frameworks[ComplianceFramework.PCI_DSS] = pci_controls
        self.frameworks[ComplianceFramework.GDPR] = gdpr_controls
        
        self.logger.info(f"Loaded {len(self.frameworks)} compliance frameworks with {sum(len(controls) for controls in self.frameworks.values())} total controls")
    
    async def _schedule_assessments(self):
        """Schedule periodic assessments for all controls"""
        for framework, controls in self.frameworks.items():
            for control_id, control in controls.items():
                # Calculate next assessment date based on frequency
                if control.frequency == "daily":
                    next_assessment = datetime.utcnow() + timedelta(days=1)
                elif control.frequency == "weekly":
                    next_assessment = datetime.utcnow() + timedelta(weeks=1)
                elif control.frequency == "monthly":
                    next_assessment = datetime.utcnow() + timedelta(days=30)
                elif control.frequency == "quarterly":
                    next_assessment = datetime.utcnow() + timedelta(days=90)
                elif control.frequency == "annually":
                    next_assessment = datetime.utcnow() + timedelta(days=365)
                else:
                    next_assessment = datetime.utcnow() + timedelta(days=90)  # Default quarterly
                
                self.assessment_schedule[f"{framework.value}:{control_id}"] = next_assessment
        
        self.logger.info(f"Scheduled {len(self.assessment_schedule)} control assessments")
    
    async def assess_compliance(self, framework: ComplianceFramework, 
                              control_ids: Optional[List[str]] = None) -> Dict[str, ControlAssessment]:
        """Assess compliance for specific framework and controls"""
        try:
            if framework not in self.frameworks:
                raise ValueError(f"Framework {framework.value} not supported")
            
            controls = self.frameworks[framework]
            if control_ids:
                controls = {cid: controls[cid] for cid in control_ids if cid in controls}
            
            assessments = {}
            
            # Create assessment context
            context = {
                "framework": framework,
                "assessment_date": datetime.utcnow(),
                "automated": True
            }
            
            # Assess each control
            for control_id, control in controls.items():
                try:
                    assessment = await self.validator.validate_control(control, context)
                    assessments[control_id] = assessment
                    self.assessments[assessment.assessment_id] = assessment
                    
                    # Update assessment schedule
                    schedule_key = f"{framework.value}:{control_id}"
                    if control.frequency == "daily":
                        self.assessment_schedule[schedule_key] = datetime.utcnow() + timedelta(days=1)
                    elif control.frequency == "weekly":
                        self.assessment_schedule[schedule_key] = datetime.utcnow() + timedelta(weeks=1)
                    elif control.frequency == "monthly":
                        self.assessment_schedule[schedule_key] = datetime.utcnow() + timedelta(days=30)
                    elif control.frequency == "quarterly":
                        self.assessment_schedule[schedule_key] = datetime.utcnow() + timedelta(days=90)
                    else:
                        self.assessment_schedule[schedule_key] = datetime.utcnow() + timedelta(days=365)
                    
                except Exception as e:
                    self.logger.error(f"Failed to assess control {control_id}: {e}")
            
            self.logger.info(f"Completed assessment of {len(assessments)} controls for {framework.value}")
            return assessments
            
        except Exception as e:
            self.logger.error(f"Compliance assessment failed: {e}")
            raise
    
    async def generate_compliance_report(self, framework: ComplianceFramework, 
                                       reporting_period: Optional[Tuple[datetime, datetime]] = None) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        try:
            if framework not in self.frameworks:
                raise ValueError(f"Framework {framework.value} not supported")
            
            # Default to last quarter if no period specified
            if not reporting_period:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=90)
                reporting_period = (start_date, end_date)
            
            # Get recent assessments for the framework
            framework_assessments = [
                assessment for assessment in self.assessments.values()
                if assessment.framework == framework and
                reporting_period[0] <= assessment.assessed_at <= reporting_period[1]
            ]
            
            # If no recent assessments, perform new assessment
            if not framework_assessments:
                self.logger.info(f"No recent assessments found, performing new assessment for {framework.value}")
                assessment_results = await self.assess_compliance(framework)
                framework_assessments = list(assessment_results.values())
            
            # Calculate compliance metrics
            total_controls = len(framework_assessments)
            compliant_controls = len([a for a in framework_assessments if a.status == ControlStatus.COMPLIANT])
            non_compliant_controls = len([a for a in framework_assessments if a.status == ControlStatus.NON_COMPLIANT])
            
            overall_compliance_score = compliant_controls / total_controls if total_controls > 0 else 0.0
            
            # Generate key findings
            key_findings = []
            high_risk_findings = [a for a in framework_assessments if a.status == ControlStatus.NON_COMPLIANT]
            
            if high_risk_findings:
                key_findings.append(f"{len(high_risk_findings)} critical control failures require immediate attention")
            
            low_automation = [a for a in framework_assessments if a.automation_coverage < 0.5]
            if low_automation:
                key_findings.append(f"{len(low_automation)} controls have low automation coverage")
            
            # Generate recommendations using AI if available
            recommendations = await self._generate_ai_recommendations(framework_assessments)
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                framework, overall_compliance_score, key_findings, recommendations
            )
            
            report = ComplianceReport(
                report_id=f"report_{framework.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                framework=framework,
                reporting_period=reporting_period,
                overall_compliance_score=overall_compliance_score,
                total_controls=total_controls,
                compliant_controls=compliant_controls,
                non_compliant_controls=non_compliant_controls,
                control_assessments=framework_assessments,
                key_findings=key_findings,
                recommendations=recommendations,
                executive_summary=executive_summary,
                generated_at=datetime.utcnow(),
                generated_by="compliance_automation_engine"
            )
            
            self.reports[report.report_id] = report
            
            self.logger.info(f"Generated compliance report {report.report_id} for {framework.value}")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            raise
    
    async def _generate_ai_recommendations(self, assessments: List[ControlAssessment]) -> List[str]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        if self.ai_available:
            try:
                # Prepare data for AI analysis
                compliance_data = {
                    "total_assessments": len(assessments),
                    "compliance_rate": len([a for a in assessments if a.status == ControlStatus.COMPLIANT]) / len(assessments),
                    "automation_coverage": sum(a.automation_coverage for a in assessments) / len(assessments),
                    "key_failures": [
                        {
                            "control_id": a.control_id,
                            "status": a.status.value,
                            "findings": a.findings[:3],  # Top 3 findings
                            "category": a.framework.value
                        }
                        for a in assessments if a.status == ControlStatus.NON_COMPLIANT
                    ]
                }
                
                ai_response = await validate_compliance_controls("multi_framework", compliance_data)
                
                if ai_response and ai_response.recommendations:
                    recommendations.extend(ai_response.recommendations)
                
            except Exception as e:
                self.logger.warning(f"AI recommendation generation failed: {e}")
        
        # Fallback to rule-based recommendations
        if not recommendations:
            recommendations = self._generate_rule_based_recommendations(assessments)
        
        return recommendations
    
    def _generate_rule_based_recommendations(self, assessments: List[ControlAssessment]) -> List[str]:
        """Generate rule-based recommendations"""
        recommendations = []
        
        # Analyze by category
        category_failures = {}
        for assessment in assessments:
            if assessment.status == ControlStatus.NON_COMPLIANT:
                category = "unknown"  # Would extract from control metadata
                category_failures[category] = category_failures.get(category, 0) + 1
        
        # Generic recommendations based on common patterns
        non_compliant_count = len([a for a in assessments if a.status == ControlStatus.NON_COMPLIANT])
        low_automation = len([a for a in assessments if a.automation_coverage < 0.5])
        
        if non_compliant_count > 0:
            recommendations.append(f"Address {non_compliant_count} non-compliant controls with priority on critical business processes")
        
        if low_automation > 0:
            recommendations.append(f"Increase automation for {low_automation} controls to improve consistency and reduce manual effort")
        
        # Add standard recommendations
        recommendations.extend([
            "Implement continuous monitoring for high-risk controls",
            "Establish regular compliance training for all stakeholders",
            "Consider third-party security assessments for critical controls",
            "Develop incident response procedures for compliance violations"
        ])
        
        return recommendations
    
    async def _generate_executive_summary(self, framework: ComplianceFramework, 
                                        compliance_score: float, key_findings: List[str], 
                                        recommendations: List[str]) -> str:
        """Generate executive summary for compliance report"""
        
        status_description = "excellent" if compliance_score >= 0.95 else \
                           "good" if compliance_score >= 0.8 else \
                           "acceptable" if compliance_score >= 0.6 else "poor"
        
        summary = f"""
Executive Summary - {framework.value.upper()} Compliance Assessment

Overall Compliance Status: {status_description.title()} ({compliance_score:.1%})

Our assessment of {framework.value.upper()} compliance shows an overall score of {compliance_score:.1%}, 
indicating {status_description} adherence to the framework requirements.

Key Findings:
{chr(10).join(f"• {finding}" for finding in key_findings[:5])}

Priority Recommendations:
{chr(10).join(f"• {rec}" for rec in recommendations[:3])}

The compliance program demonstrates {"strong" if compliance_score >= 0.8 else "moderate" if compliance_score >= 0.6 else "weak"} 
controls implementation with {"minimal" if compliance_score >= 0.9 else "some" if compliance_score >= 0.7 else "significant"} 
areas requiring attention.

Next Steps:
• Address identified non-compliant controls within 30 days
• Implement recommended improvements to strengthen the compliance posture
• Schedule follow-up assessment in 90 days to validate remediation efforts
        """.strip()
        
        return summary
    
    async def _continuous_monitoring_loop(self):
        """Continuous monitoring loop for automated assessments"""
        self.logger.info("Starting continuous compliance monitoring")
        
        while self.continuous_monitoring:
            try:
                current_time = datetime.utcnow()
                
                # Check for scheduled assessments
                due_assessments = [
                    key for key, due_date in self.assessment_schedule.items()
                    if due_date <= current_time
                ]
                
                for assessment_key in due_assessments:
                    try:
                        framework_str, control_id = assessment_key.split(":", 1)
                        framework = ComplianceFramework(framework_str)
                        
                        # Perform assessment
                        await self.assess_compliance(framework, [control_id])
                        
                        self.logger.info(f"Completed scheduled assessment for {assessment_key}")
                        
                    except Exception as e:
                        self.logger.error(f"Scheduled assessment failed for {assessment_key}: {e}")
                
                # Sleep for monitoring interval
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(3600)
    
    def enable_continuous_monitoring(self):
        """Enable continuous monitoring"""
        if not self.continuous_monitoring:
            self.continuous_monitoring = True
            asyncio.create_task(self._continuous_monitoring_loop())
            self.logger.info("Continuous monitoring enabled")
    
    def disable_continuous_monitoring(self):
        """Disable continuous monitoring"""
        self.continuous_monitoring = False
        self.logger.info("Continuous monitoring disabled")
    
    def get_compliance_status(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Get current compliance status for a framework"""
        if framework not in self.frameworks:
            return {"error": f"Framework {framework.value} not supported"}
        
        # Get recent assessments
        recent_assessments = [
            assessment for assessment in self.assessments.values()
            if assessment.framework == framework and
            assessment.assessed_at > datetime.utcnow() - timedelta(days=30)
        ]
        
        if not recent_assessments:
            return {
                "framework": framework.value,
                "status": "no_recent_assessments",
                "message": "No assessments conducted in the last 30 days"
            }
        
        total_controls = len(recent_assessments)
        compliant = len([a for a in recent_assessments if a.status == ControlStatus.COMPLIANT])
        non_compliant = len([a for a in recent_assessments if a.status == ControlStatus.NON_COMPLIANT])
        
        return {
            "framework": framework.value,
            "overall_compliance_score": compliant / total_controls if total_controls > 0 else 0,
            "total_controls": total_controls,
            "compliant_controls": compliant,
            "non_compliant_controls": non_compliant,
            "last_assessment": max(a.assessed_at for a in recent_assessments).isoformat(),
            "status": "compliant" if compliant / total_controls >= 0.8 else "non_compliant",
            "continuous_monitoring": self.continuous_monitoring
        }
    
    def list_supported_frameworks(self) -> List[Dict[str, Any]]:
        """List all supported compliance frameworks"""
        return [
            {
                "framework": framework.value,
                "name": framework.value.replace("_", " ").title(),
                "controls_count": len(controls),
                "automation_coverage": sum(1 for c in controls.values() if c.automation_possible) / len(controls),
                "categories": list(set(c.category.value for c in controls.values()))
            }
            for framework, controls in self.frameworks.items()
        ]
    
    def get_assessment_history(self, control_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """Get assessment history for a specific control"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        history = [
            {
                "assessment_id": assessment.assessment_id,
                "assessed_at": assessment.assessed_at.isoformat(),
                "status": assessment.status.value,
                "confidence_score": assessment.confidence_score,
                "findings_count": len(assessment.findings),
                "assessed_by": assessment.assessed_by
            }
            for assessment in self.assessments.values()
            if assessment.control_id == control_id and assessment.assessed_at >= cutoff_date
        ]
        
        return sorted(history, key=lambda x: x["assessed_at"], reverse=True)
    
    async def health_check(self) -> 'ServiceHealth':
        """Perform health check"""
        try:
            checks = {
                "frameworks_loaded": len(self.frameworks),
                "total_assessments": len(self.assessments),
                "scheduled_assessments": len(self.assessment_schedule),
                "continuous_monitoring": self.continuous_monitoring,
                "ai_available": self.ai_available
            }
            
            status = ServiceStatus.HEALTHY if ServiceStatus else "healthy"
            
            health = ServiceHealth(
                status=status,
                message="Compliance Automation Engine is operational",
                timestamp=datetime.utcnow(),
                checks=checks
            ) if ServiceHealth else {
                "status": status,
                "message": "Compliance Automation Engine is operational",
                "timestamp": datetime.utcnow(),
                "checks": checks
            }
            
            return health
            
        except Exception as e:
            status = ServiceStatus.UNHEALTHY if ServiceStatus else "unhealthy"
            return ServiceHealth(
                status=status,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            ) if ServiceHealth else {
                "status": status,
                "message": f"Health check failed: {e}",
                "timestamp": datetime.utcnow(),
                "checks": {"error": str(e)}
            }

# Global compliance automation instance
_compliance_automation: Optional[EnterpriseComplianceAutomation] = None

async def get_compliance_automation() -> EnterpriseComplianceAutomation:
    """Get global compliance automation instance"""
    global _compliance_automation
    
    if _compliance_automation is None:
        _compliance_automation = EnterpriseComplianceAutomation()
        await _compliance_automation.initialize()
    
    return _compliance_automation

# Example usage
if __name__ == "__main__":
    async def demo():
        automation = EnterpriseComplianceAutomation()
        await automation.initialize()
        
        # List supported frameworks
        frameworks = automation.list_supported_frameworks()
        print(f"Supported frameworks: {len(frameworks)}")
        for framework in frameworks:
            print(f"- {framework['name']}: {framework['controls_count']} controls")
        
        # Assess SOC 2 compliance
        print("\nAssessing SOC 2 compliance...")
        soc2_assessments = await automation.assess_compliance(ComplianceFramework.SOC2_TYPE2)
        
        for control_id, assessment in soc2_assessments.items():
            print(f"Control {control_id}: {assessment.status.value} (confidence: {assessment.confidence_score:.2f})")
        
        # Generate compliance report
        print("\nGenerating compliance report...")
        report = await automation.generate_compliance_report(ComplianceFramework.SOC2_TYPE2)
        
        print(f"Report ID: {report.report_id}")
        print(f"Overall Compliance: {report.overall_compliance_score:.1%}")
        print(f"Compliant Controls: {report.compliant_controls}/{report.total_controls}")
        print(f"\nKey Findings:")
        for finding in report.key_findings[:3]:
            print(f"- {finding}")
        
        print(f"\nRecommendations:")
        for rec in report.recommendations[:3]:
            print(f"- {rec}")
        
        # Get compliance status
        status = automation.get_compliance_status(ComplianceFramework.SOC2_TYPE2)
        print(f"\nCurrent Status: {status}")
        
        await automation.shutdown()
    
    asyncio.run(demo())