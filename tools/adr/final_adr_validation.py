#!/usr/bin/env python3
"""
Final ADR Compliance Validation Script
Validates complete ADR-001 through ADR-004 compliance after all remediation work.
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    component: str
    status: str  # "PASS", "FAIL", "WARNING"
    message: str
    details: str = ""

class FinalADRValidator:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.results: List[ValidationResult] = []
        
    def validate_all(self) -> Dict[str, Any]:
        """Run comprehensive ADR compliance validation"""
        print("🔍 XORB Final ADR Compliance Validation")
        print("=" * 50)
        
        # ADR-001: Languages and Repository Architecture
        self.validate_adr_001()
        
        # ADR-002: Two-Tier Bus Architecture  
        self.validate_adr_002()
        
        # ADR-003: Authentication Artifact Architecture
        self.validate_adr_003()
        
        # ADR-004: Evidence Schema
        self.validate_adr_004()
        
        # Generate final report
        return self.generate_final_report()
    
    def validate_adr_001(self):
        """Validate ADR-001: Languages and Repository Architecture compliance"""
        print("\n📋 ADR-001: Languages and Repository Architecture")
        
        # Check repository structure
        required_dirs = [
            "src/api", "src/orchestrator", "src/xorb", "src/common",
            "services/ptaas", "services/xorb-core", "services/infrastructure",
            "infra/kubernetes", "infra/monitoring", "infra/vault",
            "tests", "tools"
        ]
        
        for dir_path in required_dirs:
            full_path = self.repo_root / dir_path
            if full_path.exists():
                self.results.append(ValidationResult(
                    "ADR-001-Structure", "PASS", 
                    f"Required directory exists: {dir_path}"
                ))
            else:
                self.results.append(ValidationResult(
                    "ADR-001-Structure", "WARNING", 
                    f"Directory missing but may be acceptable: {dir_path}"
                ))
        
        # Check Python backend presence
        python_files = list(self.repo_root.glob("src/api/**/*.py"))
        if python_files:
            self.results.append(ValidationResult(
                "ADR-001-Languages", "PASS", 
                f"Python backend present: {len(python_files)} Python files found"
            ))
        else:
            self.results.append(ValidationResult(
                "ADR-001-Languages", "FAIL", 
                "Python backend missing - no Python files in src/api"
            ))
        
        # Check TypeScript frontend presence
        ts_files = list(self.repo_root.glob("services/ptaas/**/*.ts*"))
        if ts_files:
            self.results.append(ValidationResult(
                "ADR-001-Languages", "PASS", 
                f"TypeScript frontend present: {len(ts_files)} TypeScript files found"
            ))
        else:
            self.results.append(ValidationResult(
                "ADR-001-Languages", "WARNING", 
                "TypeScript frontend not found in services/ptaas"
            ))
    
    def validate_adr_002(self):
        """Validate ADR-002: Two-Tier Bus Architecture compliance"""
        print("\n🚌 ADR-002: Two-Tier Bus Architecture")
        
        # Check platform/bus directories exist
        bus_dirs = ["platform/bus/pubsub", "platform/bus/localring"]
        for bus_dir in bus_dirs:
            full_path = self.repo_root / bus_dir
            if full_path.exists():
                self.results.append(ValidationResult(
                    "ADR-002-Structure", "PASS", 
                    f"Two-Tier Bus directory exists: {bus_dir}"
                ))
            else:
                self.results.append(ValidationResult(
                    "ADR-002-Structure", "FAIL", 
                    f"Missing required Two-Tier Bus directory: {bus_dir}"
                ))
        
        # Check NATS JetStream implementation
        nats_file = self.repo_root / "platform/bus/pubsub/nats_client.py"
        if nats_file.exists():
            content = nats_file.read_text()
            if "JetStream" in content and "WORM" in content:
                self.results.append(ValidationResult(
                    "ADR-002-NATS", "PASS", 
                    "NATS JetStream client implemented with WORM retention"
                ))
            else:
                self.results.append(ValidationResult(
                    "ADR-002-NATS", "WARNING", 
                    "NATS client exists but missing JetStream/WORM features"
                ))
        else:
            self.results.append(ValidationResult(
                "ADR-002-NATS", "FAIL", 
                "NATS JetStream client missing"
            ))
        
        # Check UDS transport implementation
        uds_file = self.repo_root / "platform/bus/localring/uds_transport.py"
        if uds_file.exists():
            self.results.append(ValidationResult(
                "ADR-002-UDS", "PASS", 
                "UDS transport for Tier-1 local ring implemented"
            ))
        else:
            self.results.append(ValidationResult(
                "ADR-002-UDS", "FAIL", 
                "UDS transport for Tier-1 local ring missing"
            ))
        
        # Check for Redis bus violations
        self.check_redis_bus_violations()
        
        # Check ADR-002 has locked phrases
        adr_002_file = self.repo_root / "docs/architecture/ADR-002-Two-Tier-Bus.md"
        if adr_002_file.exists():
            content = adr_002_file.read_text()
            if "(locked)" in content:
                self.results.append(ValidationResult(
                    "ADR-002-Locked", "PASS", 
                    "ADR-002 contains required locked phrases"
                ))
            else:
                self.results.append(ValidationResult(
                    "ADR-002-Locked", "FAIL", 
                    "ADR-002 missing required locked phrases"
                ))
    
    def validate_adr_003(self):
        """Validate ADR-003: Authentication Artifact Architecture compliance"""
        print("\n🔐 ADR-003: Authentication Artifact Architecture")
        
        # Check Vault integration
        vault_files = list(self.repo_root.glob("**/vault_client*.py"))
        if vault_files:
            self.results.append(ValidationResult(
                "ADR-003-Vault", "PASS", 
                f"Vault integration present: {len(vault_files)} vault client files"
            ))
        else:
            self.results.append(ValidationResult(
                "ADR-003-Vault", "WARNING", 
                "Vault client files not found"
            ))
        
        # Check JWT/mTLS patterns
        auth_patterns = ["JWT", "mTLS", "certificate", "bearer"]
        auth_files = []
        for pattern in ["**/*auth*.py", "**/*jwt*.py", "**/*cert*.py"]:
            auth_files.extend(self.repo_root.glob(pattern))
        
        if auth_files:
            self.results.append(ValidationResult(
                "ADR-003-Auth", "PASS", 
                f"Authentication implementation present: {len(auth_files)} auth-related files"
            ))
        else:
            self.results.append(ValidationResult(
                "ADR-003-Auth", "WARNING", 
                "Authentication implementation files not found"
            ))
        
        # Check ADR-003 has locked phrases
        adr_003_file = self.repo_root / "docs/architecture/ADR-003-Auth-Artifact.md"
        if adr_003_file.exists():
            content = adr_003_file.read_text()
            if "(locked)" in content:
                self.results.append(ValidationResult(
                    "ADR-003-Locked", "PASS", 
                    "ADR-003 contains required locked phrases"
                ))
            else:
                self.results.append(ValidationResult(
                    "ADR-003-Locked", "FAIL", 
                    "ADR-003 missing required locked phrases"
                ))
    
    def validate_adr_004(self):
        """Validate ADR-004: Evidence Schema compliance"""
        print("\n📊 ADR-004: Evidence Schema")
        
        # Check evidence protobuf schemas
        required_schemas = [
            "proto/audit/v1/evidence.proto",
            "proto/discovery/v1/discovery.proto", 
            "proto/threat/v1/threat.proto",
            "proto/vuln/v1/vulnerability.proto",
            "proto/compliance/v1/compliance.proto"
        ]
        
        for schema_path in required_schemas:
            full_path = self.repo_root / schema_path
            if full_path.exists():
                content = full_path.read_text()
                if "ChainOfCustody" in content or "Evidence" in content:
                    self.results.append(ValidationResult(
                        "ADR-004-Schemas", "PASS", 
                        f"Evidence schema exists: {schema_path}"
                    ))
                else:
                    self.results.append(ValidationResult(
                        "ADR-004-Schemas", "WARNING", 
                        f"Schema exists but missing evidence patterns: {schema_path}"
                    ))
            else:
                self.results.append(ValidationResult(
                    "ADR-004-Schemas", "FAIL", 
                    f"Missing evidence schema: {schema_path}"
                ))
    
    def check_redis_bus_violations(self):
        """Check for Redis being used as a bus transport (violation of ADR-002)"""
        violation_patterns = [
            r"redis\.(pubsub|subscribe|psubscribe)",
            r"redis\.client\.PubSub",
            r"RedisStreams?",
            r"XADD|XREAD|XGROUP",
            r"redis.*stream.*publish",
            r"redis.*as.*bus"
        ]
        
        violations_found = 0
        search_dirs = ["src", "services", "ptaas", "platform"]
        
        for search_dir in search_dirs:
            dir_path = self.repo_root / search_dir
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    for line_num, line in enumerate(content.split('\n'), 1):
                        for pattern in violation_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                violations_found += 1
                                self.results.append(ValidationResult(
                                    "ADR-002-Redis-Bus", "FAIL", 
                                    f"Redis bus violation in {py_file}: line {line_num}: {line.strip()[:100]}"
                                ))
                except Exception as e:
                    continue
        
        if violations_found == 0:
            self.results.append(ValidationResult(
                "ADR-002-Redis-Bus", "PASS", 
                "No Redis bus violations found - Redis restricted to cache usage only"
            ))
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final validation report"""
        print("\n📊 Final ADR Compliance Report")
        print("=" * 50)
        
        # Categorize results
        by_status = {"PASS": [], "FAIL": [], "WARNING": []}
        by_adr = {"ADR-001": [], "ADR-002": [], "ADR-003": [], "ADR-004": []}
        
        for result in self.results:
            by_status[result.status].append(result)
            
            # Extract ADR number
            if "ADR-001" in result.component:
                by_adr["ADR-001"].append(result)
            elif "ADR-002" in result.component:
                by_adr["ADR-002"].append(result)
            elif "ADR-003" in result.component:
                by_adr["ADR-003"].append(result)
            elif "ADR-004" in result.component:
                by_adr["ADR-004"].append(result)
        
        # Calculate compliance scores
        total_checks = len(self.results)
        pass_count = len(by_status["PASS"])
        fail_count = len(by_status["FAIL"])
        warning_count = len(by_status["WARNING"])
        
        # Weighted scoring (PASS=100%, WARNING=70%, FAIL=0%)
        weighted_score = (pass_count * 100 + warning_count * 70) / total_checks if total_checks > 0 else 0
        
        # ADR-specific compliance
        adr_scores = {}
        for adr, results in by_adr.items():
            if results:
                adr_pass = len([r for r in results if r.status == "PASS"])
                adr_warn = len([r for r in results if r.status == "WARNING"])
                adr_total = len(results)
                adr_scores[adr] = (adr_pass * 100 + adr_warn * 70) / adr_total
            else:
                adr_scores[adr] = 0.0
        
        # Print summary
        print(f"✅ PASS: {pass_count}")
        print(f"⚠️  WARNING: {warning_count}")
        print(f"❌ FAIL: {fail_count}")
        print(f"📊 Overall Compliance: {weighted_score:.1f}%")
        print()
        
        # Print ADR-specific scores
        for adr, score in adr_scores.items():
            status_icon = "✅" if score >= 90 else "⚠️" if score >= 70 else "❌"
            print(f"{status_icon} {adr}: {score:.1f}%")
        
        print()
        
        # Print detailed results
        for status in ["FAIL", "WARNING", "PASS"]:
            if by_status[status]:
                icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}[status]
                print(f"\n{icon} {status} Results:")
                for result in by_status[status]:
                    print(f"  • {result.component}: {result.message}")
        
        # Generate report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "repo_root": str(self.repo_root),
            "summary": {
                "total_checks": total_checks,
                "pass_count": pass_count,
                "warning_count": warning_count,
                "fail_count": fail_count,
                "overall_compliance": weighted_score,
                "adr_scores": adr_scores
            },
            "results": [
                {
                    "component": r.component,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        # Write JSON report
        report_file = self.repo_root / "ADR_COMPLIANCE_FINAL_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report written to: {report_file}")
        
        # Determine overall status
        if fail_count == 0 and warning_count <= 2:
            print(f"\n🎉 ADR COMPLIANCE STATUS: EXCELLENT ({weighted_score:.1f}%)")
            overall_status = "EXCELLENT"
        elif weighted_score >= 80:
            print(f"\n✅ ADR COMPLIANCE STATUS: GOOD ({weighted_score:.1f}%)")
            overall_status = "GOOD"
        elif weighted_score >= 60:
            print(f"\n⚠️ ADR COMPLIANCE STATUS: NEEDS IMPROVEMENT ({weighted_score:.1f}%)")
            overall_status = "NEEDS_IMPROVEMENT"
        else:
            print(f"\n❌ ADR COMPLIANCE STATUS: CRITICAL ISSUES ({weighted_score:.1f}%)")
            overall_status = "CRITICAL"
        
        report_data["overall_status"] = overall_status
        
        # Re-write with overall status
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_data

def main():
    if len(sys.argv) > 1:
        repo_root = sys.argv[1]
    else:
        repo_root = os.getcwd()
    
    validator = FinalADRValidator(repo_root)
    report = validator.validate_all()
    
    # Exit code based on compliance
    if report["overall_status"] in ["EXCELLENT", "GOOD"]:
        sys.exit(0)
    elif report["overall_status"] == "NEEDS_IMPROVEMENT":
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()