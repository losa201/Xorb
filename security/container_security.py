#!/usr/bin/env python3
"""
Container Security Scanner for XORB
Implements multi-layer container security scanning and policy enforcement
"""

import json
import subprocess
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import os


logger = logging.getLogger(__name__)


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ScannerType(Enum):
    """Container security scanner types"""
    TRIVY = "trivy"
    GRYPE = "grype"
    DOCKER_SCOUT = "docker_scout"
    SNYK = "snyk"


@dataclass
class Vulnerability:
    """Container vulnerability"""
    cve_id: str
    severity: VulnerabilitySeverity
    package_name: str
    package_version: str
    fixed_version: Optional[str]
    description: str
    cvss_score: float
    vector: str


@dataclass
class SecurityPolicy:
    """Container security policy"""
    name: str
    description: str
    max_critical: int = 0
    max_high: int = 5
    max_medium: int = 20
    allowed_base_images: List[str] = None
    forbidden_packages: List[str] = None
    required_labels: List[str] = None
    enforce_non_root: bool = True
    enforce_read_only_fs: bool = True


@dataclass
class ScanResult:
    """Container scan result"""
    image_name: str
    image_tag: str
    image_digest: str
    scan_time: datetime
    scanner_type: ScannerType
    vulnerabilities: List[Vulnerability]
    policy_violations: List[str]
    compliance_status: str
    metadata: Dict[str, Any]


class ContainerSecurityScanner:
    """Multi-engine container security scanner"""
    
    def __init__(self, config_path: str = "security/container-security-config.yaml"):
        self.config = self._load_config(config_path)
        self.policies = self._load_policies()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load scanner configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "scanners": {
                    "trivy": {"enabled": True, "timeout": 300},
                    "grype": {"enabled": True, "timeout": 300},
                    "docker_scout": {"enabled": False, "timeout": 300}
                },
                "output": {
                    "format": "json",
                    "path": "security/scan-results"
                },
                "notifications": {
                    "slack_webhook": None,
                    "email_alerts": []
                }
            }
            
    def _load_policies(self) -> Dict[str, SecurityPolicy]:
        """Load security policies"""
        return {
            "production": SecurityPolicy(
                name="Production Security Policy",
                description="Strict security policy for production environments",
                max_critical=0,
                max_high=0,
                max_medium=5,
                allowed_base_images=[
                    "python:3.12-slim",
                    "node:18-alpine",
                    "nginx:alpine",
                    "postgres:15-alpine"
                ],
                forbidden_packages=[
                    "telnet", "ftp", "rsh", "rlogin"
                ],
                required_labels=[
                    "maintainer", "version", "security.scan"
                ],
                enforce_non_root=True,
                enforce_read_only_fs=True
            ),
            "staging": SecurityPolicy(
                name="Staging Security Policy",
                description="Moderate security policy for staging environments",
                max_critical=0,
                max_high=2,
                max_medium=10,
                enforce_non_root=True,
                enforce_read_only_fs=False
            ),
            "development": SecurityPolicy(
                name="Development Security Policy",
                description="Relaxed security policy for development",
                max_critical=1,
                max_high=5,
                max_medium=20,
                enforce_non_root=False,
                enforce_read_only_fs=False
            )
        }
        
    async def scan_image(
        self, 
        image_name: str, 
        policy_name: str = "production"
    ) -> ScanResult:
        """Scan container image for vulnerabilities"""
        
        logger.info(f"Starting security scan for image: {image_name}")
        
        # Get policy
        policy = self.policies.get(policy_name)
        if not policy:
            raise ValueError(f"Policy '{policy_name}' not found")
            
        # Run vulnerability scans
        vulnerabilities = []
        
        if self.config["scanners"]["trivy"]["enabled"]:
            trivy_vulns = await self._scan_with_trivy(image_name)
            vulnerabilities.extend(trivy_vulns)
            
        if self.config["scanners"]["grype"]["enabled"]:
            grype_vulns = await self._scan_with_grype(image_name)
            vulnerabilities.extend(grype_vulns)
            
        # Deduplicate vulnerabilities
        vulnerabilities = self._deduplicate_vulnerabilities(vulnerabilities)
        
        # Check policy compliance
        policy_violations = self._check_policy_compliance(
            image_name, vulnerabilities, policy
        )
        
        # Determine compliance status
        compliance_status = "compliant" if not policy_violations else "non_compliant"
        
        # Get image metadata
        metadata = await self._get_image_metadata(image_name)
        
        return ScanResult(
            image_name=image_name.split(':')[0],
            image_tag=image_name.split(':')[1] if ':' in image_name else 'latest',
            image_digest=metadata.get('digest', ''),
            scan_time=datetime.utcnow(),
            scanner_type=ScannerType.TRIVY,  # Primary scanner
            vulnerabilities=vulnerabilities,
            policy_violations=policy_violations,
            compliance_status=compliance_status,
            metadata=metadata
        )
        
    async def _scan_with_trivy(self, image_name: str) -> List[Vulnerability]:
        """Scan image with Trivy"""
        try:
            cmd = [
                "trivy", "image", "--format", "json",
                "--severity", "CRITICAL,HIGH,MEDIUM,LOW",
                image_name
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Trivy scan failed: {stderr.decode()}")
                return []
                
            result = json.loads(stdout.decode())
            return self._parse_trivy_output(result)
            
        except Exception as e:
            logger.error(f"Error running Trivy scan: {e}")
            return []
            
    async def _scan_with_grype(self, image_name: str) -> List[Vulnerability]:
        """Scan image with Grype"""
        try:
            cmd = [
                "grype", image_name, "-o", "json"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Grype scan failed: {stderr.decode()}")
                return []
                
            result = json.loads(stdout.decode())
            return self._parse_grype_output(result)
            
        except Exception as e:
            logger.error(f"Error running Grype scan: {e}")
            return []
            
    def _parse_trivy_output(self, trivy_result: Dict[str, Any]) -> List[Vulnerability]:
        """Parse Trivy output to vulnerability objects"""
        vulnerabilities = []
        
        for result in trivy_result.get("Results", []):
            for vuln in result.get("Vulnerabilities", []):
                vulnerabilities.append(Vulnerability(
                    cve_id=vuln.get("VulnerabilityID", ""),
                    severity=VulnerabilitySeverity(vuln.get("Severity", "unknown").lower()),
                    package_name=vuln.get("PkgName", ""),
                    package_version=vuln.get("InstalledVersion", ""),
                    fixed_version=vuln.get("FixedVersion"),
                    description=vuln.get("Description", ""),
                    cvss_score=vuln.get("CVSS", {}).get("nvd", {}).get("V3Score", 0.0),
                    vector=vuln.get("CVSS", {}).get("nvd", {}).get("V3Vector", "")
                ))
                
        return vulnerabilities
        
    def _parse_grype_output(self, grype_result: Dict[str, Any]) -> List[Vulnerability]:
        """Parse Grype output to vulnerability objects"""
        vulnerabilities = []
        
        for match in grype_result.get("matches", []):
            vuln = match.get("vulnerability", {})
            artifact = match.get("artifact", {})
            
            vulnerabilities.append(Vulnerability(
                cve_id=vuln.get("id", ""),
                severity=VulnerabilitySeverity(vuln.get("severity", "unknown").lower()),
                package_name=artifact.get("name", ""),
                package_version=artifact.get("version", ""),
                fixed_version=vuln.get("fix", {}).get("versions", [None])[0],
                description=vuln.get("description", ""),
                cvss_score=0.0,  # Grype doesn't always provide CVSS
                vector=""
            ))
            
        return vulnerabilities
        
    def _deduplicate_vulnerabilities(
        self, 
        vulnerabilities: List[Vulnerability]
    ) -> List[Vulnerability]:
        """Remove duplicate vulnerabilities"""
        seen = set()
        unique_vulns = []
        
        for vuln in vulnerabilities:
            key = (vuln.cve_id, vuln.package_name, vuln.package_version)
            if key not in seen:
                seen.add(key)
                unique_vulns.append(vuln)
                
        return unique_vulns
        
    def _check_policy_compliance(
        self,
        image_name: str,
        vulnerabilities: List[Vulnerability],
        policy: SecurityPolicy
    ) -> List[str]:
        """Check image compliance against security policy"""
        violations = []
        
        # Count vulnerabilities by severity
        severity_counts = {}
        for severity in VulnerabilitySeverity:
            severity_counts[severity] = len([
                v for v in vulnerabilities if v.severity == severity
            ])
            
        # Check vulnerability thresholds
        if severity_counts[VulnerabilitySeverity.CRITICAL] > policy.max_critical:
            violations.append(
                f"Critical vulnerabilities exceed limit: "
                f"{severity_counts[VulnerabilitySeverity.CRITICAL]} > {policy.max_critical}"
            )
            
        if severity_counts[VulnerabilitySeverity.HIGH] > policy.max_high:
            violations.append(
                f"High vulnerabilities exceed limit: "
                f"{severity_counts[VulnerabilitySeverity.HIGH]} > {policy.max_high}"
            )
            
        if severity_counts[VulnerabilitySeverity.MEDIUM] > policy.max_medium:
            violations.append(
                f"Medium vulnerabilities exceed limit: "
                f"{severity_counts[VulnerabilitySeverity.MEDIUM]} > {policy.max_medium}"
            )
            
        # Check base image compliance
        if policy.allowed_base_images:
            base_image_compliant = any(
                image_name.startswith(allowed) 
                for allowed in policy.allowed_base_images
            )
            if not base_image_compliant:
                violations.append(f"Base image not in allowed list: {image_name}")
                
        # Check for forbidden packages
        if policy.forbidden_packages:
            forbidden_found = [
                v.package_name for v in vulnerabilities 
                if v.package_name in policy.forbidden_packages
            ]
            if forbidden_found:
                violations.append(f"Forbidden packages found: {forbidden_found}")
                
        return violations
        
    async def _get_image_metadata(self, image_name: str) -> Dict[str, Any]:
        """Get image metadata using docker inspect"""
        try:
            cmd = ["docker", "inspect", image_name]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {}
                
            inspect_data = json.loads(stdout.decode())[0]
            
            return {
                "digest": inspect_data.get("Id", ""),
                "created": inspect_data.get("Created", ""),
                "size": inspect_data.get("Size", 0),
                "architecture": inspect_data.get("Architecture", ""),
                "os": inspect_data.get("Os", ""),
                "config": inspect_data.get("Config", {}),
                "labels": inspect_data.get("Config", {}).get("Labels", {})
            }
            
        except Exception as e:
            logger.error(f"Error getting image metadata: {e}")
            return {}
            
    def generate_security_report(
        self, 
        scan_results: List[ScanResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        total_images = len(scan_results)
        compliant_images = len([r for r in scan_results if r.compliance_status == "compliant"])
        
        # Vulnerability summary
        all_vulnerabilities = []
        for result in scan_results:
            all_vulnerabilities.extend(result.vulnerabilities)
            
        vuln_by_severity = {}
        for severity in VulnerabilitySeverity:
            vuln_by_severity[severity.value] = len([
                v for v in all_vulnerabilities if v.severity == severity
            ])
            
        # Top vulnerabilities
        top_cves = {}
        for vuln in all_vulnerabilities:
            if vuln.cve_id in top_cves:
                top_cves[vuln.cve_id] += 1
            else:
                top_cves[vuln.cve_id] = 1
                
        top_cves_list = sorted(top_cves.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "summary": {
                "total_images_scanned": total_images,
                "compliant_images": compliant_images,
                "compliance_rate": (compliant_images / total_images * 100) if total_images > 0 else 0,
                "total_vulnerabilities": len(all_vulnerabilities),
                "scan_timestamp": datetime.utcnow().isoformat()
            },
            "vulnerability_breakdown": vuln_by_severity,
            "top_cves": top_cves_list,
            "scan_results": [asdict(result) for result in scan_results],
            "recommendations": self._generate_recommendations(scan_results)
        }
        
    def _generate_recommendations(self, scan_results: List[ScanResult]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Check for high vulnerability counts
        high_vuln_images = [
            r for r in scan_results 
            if len([v for v in r.vulnerabilities if v.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]]) > 5
        ]
        
        if high_vuln_images:
            recommendations.append(
                f"Update base images for {len(high_vuln_images)} images with high vulnerability counts"
            )
            
        # Check for non-compliant images
        non_compliant = [r for r in scan_results if r.compliance_status == "non_compliant"]
        if non_compliant:
            recommendations.append(
                f"Address policy violations in {len(non_compliant)} non-compliant images"
            )
            
        # General recommendations
        recommendations.extend([
            "Implement automated vulnerability scanning in CI/CD pipeline",
            "Set up container image signing and verification",
            "Use minimal base images (Alpine, Distroless)",
            "Implement runtime security monitoring"
        ])
        
        return recommendations


async def scan_container_image(image_name: str, policy: str = "production") -> Dict[str, Any]:
    """API endpoint for scanning a single container image"""
    scanner = ContainerSecurityScanner()
    result = await scanner.scan_image(image_name, policy)
    return asdict(result)


async def scan_all_images(policy: str = "production") -> Dict[str, Any]:
    """Scan all images in the system"""
    scanner = ContainerSecurityScanner()
    
    # Get list of images (this would typically come from a registry)
    images = [
        "xorb/api:latest",
        "xorb/frontend:latest", 
        "xorb/orchestrator:latest"
    ]
    
    results = []
    for image in images:
        try:
            result = await scanner.scan_image(image, policy)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to scan {image}: {e}")
            
    return scanner.generate_security_report(results)


if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python container_security.py <image_name> [policy]")
            return
            
        image_name = sys.argv[1]
        policy = sys.argv[2] if len(sys.argv) > 2 else "production"
        
        result = await scan_container_image(image_name, policy)
        print(json.dumps(result, indent=2, default=str))
        
    asyncio.run(main())