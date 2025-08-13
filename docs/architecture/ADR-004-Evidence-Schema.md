# ADR-004: Evidence Schema for Chain of Custody

**Status:** Accepted  
**Date:** 2025-08-13  
**Deciders:** Chief Architect  

## Context

XORB platform requires comprehensive evidence collection and chain of custody for security scanning, vulnerability assessment, compliance reporting, and forensic investigations. Evidence must be tamper-evident, legally admissible, and traceable through the complete discovery-to-remediation lifecycle.

## Decision

### Evidence Schema Architecture

#### Core Evidence Types
1. **Discovery Evidence**: Network scans, service enumeration, asset fingerprints
2. **Vulnerability Evidence**: Security findings, exploit proofs, risk assessments  
3. **Threat Evidence**: Indicators of compromise, attack patterns, behavioral anomalies
4. **Compliance Evidence**: Policy violations, control effectiveness, audit trails
5. **Forensic Evidence**: Incident artifacts, preservation metadata, chain documentation

#### Evidence Collection Framework
```protobuf
message Evidence {
    string evidence_id = 1;
    string tenant_id = 2;
    EvidenceType type = 3;
    
    // Chain of custody
    ChainOfCustody chain = 4;
    
    // Evidence content
    google.protobuf.Any payload = 5;
    
    // Legal requirements
    LegalContext legal = 6;
    
    // Technical metadata
    TechnicalMetadata technical = 7;
}

message ChainOfCustody {
    string chain_id = 1;
    repeated CustodyRecord records = 2;
    DigitalSignature signature = 3;
    bool tamper_evident = 4;
    google.protobuf.Timestamp collection_time = 5;
}
```

### Evidence Lifecycle Management

#### Collection Phase
- **Automated Collection**: Discovery jobs, vulnerability scans, compliance checks
- **Manual Collection**: Incident response, forensic analysis, expert review
- **External Import**: Threat intelligence feeds, vendor reports, third-party scans
- **Timestamp Authority**: RFC 3161 timestamping for legal admissibility

#### Processing Phase  
- **Validation**: Cryptographic integrity, source authentication, completeness checks
- **Classification**: Sensitivity labeling, retention requirements, access controls
- **Correlation**: Cross-reference with related evidence, timeline reconstruction
- **Analysis**: Risk assessment, impact evaluation, remediation prioritization

#### Storage Phase
- **Immutable Storage**: WORM compliance, tamper-evident containers
- **Encryption**: AES-256 at rest, TLS 1.3 in transit, key escrow for legal holds
- **Backup**: Geographically distributed, legally compliant retention
- **Indexing**: Full-text search, metadata queries, timeline analysis

#### Disposition Phase
- **Retention Policies**: Industry-specific requirements (7 years SOX, 3 years GDPR)
- **Legal Holds**: Litigation preservation, regulatory investigation support
- **Destruction**: Secure deletion, certificate of destruction, audit trails
- **Transfer**: Evidence handoff to legal counsel, law enforcement, auditors

### Technical Implementation

#### Digital Signatures
```yaml
# Evidence signing configuration
signing:
  algorithm: "ECDSA-P256-SHA256"
  certificate_chain: true
  timestamp_authority: "http://timestamp.digicert.com"
  
verification:
  require_chain: true
  check_revocation: true
  validate_timestamp: true
```

#### Storage Backend
- **Primary**: PostgreSQL with row-level security for metadata
- **Artifacts**: S3-compatible object storage with versioning
- **Search**: Elasticsearch for full-text and metadata queries
- **Backup**: Cross-region replication with encryption

#### API Interface
```http
POST /api/v1/evidence
Content-Type: application/json
Authorization: Bearer {token}

{
  "type": "VULNERABILITY_EVIDENCE",
  "source": "nuclei-scan",
  "target": "192.168.1.100:443",
  "payload": {
    "cve_id": "CVE-2024-1234",
    "severity": "HIGH",
    "proof_of_concept": "..."
  },
  "collection_metadata": {
    "tool_version": "nuclei-3.0.1",
    "scan_time": "2025-08-13T10:30:00Z"
  }
}
```

### Compliance Integration

#### Legal Requirements
- **GDPR Article 5**: Lawfulness, fairness, transparency in evidence processing
- **SOX Section 404**: Internal controls over financial reporting evidence
- **HIPAA**: PHI protection in healthcare vulnerability evidence
- **FRE Rule 901**: Evidence authentication requirements

#### Audit Trail Requirements
```protobuf
message AuditTrail {
    string evidence_id = 1;
    repeated AccessRecord access_history = 2;
    repeated ModificationRecord modifications = 3;
    repeated TransferRecord transfers = 4;
    ComplianceAttestation compliance = 5;
}
```

#### Retention Policies
```yaml
retention_policies:
  vulnerability_evidence:
    default: "3_years"
    critical_findings: "7_years"
    compliance_scans: "7_years"
  
  forensic_evidence:
    default: "7_years"
    litigation_hold: "indefinite"
    law_enforcement: "indefinite"
    
  discovery_evidence:
    default: "1_year"
    baseline_scans: "3_years"
    change_detection: "2_years"
```

## Integration with XORB Architecture

### Discovery Service Integration
- Automatically generate discovery evidence for all scan results
- Include tool fingerprints, scan parameters, target validation
- Chain multiple scan evidences for comprehensive asset profiles

### Risk Management Integration  
- Evidence-backed risk scoring and prioritization
- Audit trail for risk decisions and remediation tracking
- Compliance evidence for regulatory reporting

### NATS JetStream Evidence Bus
```yaml
# Evidence streaming configuration
evidence_streams:
  subjects:
    - "evidence.collected.v1.{tenant_id}"
    - "evidence.analyzed.v1.{tenant_id}"
    - "evidence.exported.v1.{tenant_id}"
  
  retention:
    policy: "limits"
    max_age: "2555_days"  # 7 years
    max_bytes: "100_gb_per_tenant"
    storage: "file"
```

### Forensic Analysis Workflow
1. **Evidence Preservation**: Immutable collection with hash verification
2. **Chain Documentation**: Automated custody record generation
3. **Timeline Reconstruction**: Correlated evidence sequencing
4. **Export Preparation**: Legal format conversion and authentication
5. **Court Readiness**: Expert witness support documentation

## Consequences

### Positive
- **Legal Admissibility**: Court-ready evidence with proper chain of custody
- **Regulatory Compliance**: SOX, GDPR, HIPAA evidence requirements met
- **Forensic Capability**: Professional incident response and investigation
- **Audit Support**: Comprehensive trail for external audits
- **Risk Quantification**: Evidence-based risk assessment and reporting

### Negative
- **Storage Overhead**: Significant storage requirements for evidence retention
- **Performance Impact**: Cryptographic operations for signing and verification
- **Operational Complexity**: Chain of custody procedures and compliance workflows
- **Legal Liability**: Responsibility for evidence integrity and legal compliance

### Migration Strategy
- **Phase 1**: Implement evidence collection APIs and basic chain of custody
- **Phase 2**: Add digital signatures and immutable storage
- **Phase 3**: Integrate with discovery and vulnerability services
- **Phase 4**: Full forensic capabilities and legal export formats

## Implementation Files

### Required Artifacts
```
/proto/evidence/v1/evidence.proto
/services/evidence/collection/
/services/evidence/storage/
/services/evidence/forensics/
/infra/storage/evidence-vault/
/docs/compliance/evidence-procedures.md
/tests/evidence/chain-of-custody-tests.py
```

### Integration Points
- Discovery Service: Auto-evidence generation
- Risk Service: Evidence-backed assessments  
- Audit Service: Chain of custody logging
- Export Service: Legal format conversion
- Compliance Service: Retention policy enforcement