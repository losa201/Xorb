#!/usr/bin/env python3
"""
XORB Platform Dependency Security Scanner
Batch 2: Automated dependency scanning implementation

This script provides automated security scanning for all dependencies across the platform.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

@dataclass
class Vulnerability:
    package: str
    version: str
    vulnerability_id: str
    severity: str
    description: str
    fix_version: Optional[str] = None

class DependencySecurityScanner:
    """Automated security scanner for platform dependencies."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.python_vulnerabilities: List[Vulnerability] = []
        self.npm_vulnerabilities: List[Vulnerability] = []
        self.report = {
            'scan_date': '',
            'python': {'total': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'npm': {'total': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'recommendations': []
        }
    
    def run_python_security_scan(self) -> List[Vulnerability]:
        """Run security scan on Python dependencies using safety and pip-audit."""
        print("🔍 Running Python dependency security scan...")
        
        vulnerabilities = []
        
        # Run safety check
        try:
            cmd = ["safety", "check", "--json", "--file", "requirements-unified.lock"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0 and result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    vulnerabilities.append(Vulnerability(
                        package=vuln.get('package', 'unknown'),
                        version=vuln.get('installed_version', 'unknown'),
                        vulnerability_id=vuln.get('id', 'unknown'),
                        severity=self._map_safety_severity(vuln.get('safety_grade', 'medium')),
                        description=vuln.get('advisory', 'Unknown vulnerability'),
                        fix_version=vuln.get('fix_version')
                    ))
            
            print(f"   • Safety scan: {len(vulnerabilities)} vulnerabilities found")
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"   ⚠️  Safety scan failed: {e}")
        
        # Run pip-audit (alternative scanner)
        try:
            cmd = ["pip-audit", "--format=json", "--requirement", "requirements-unified.lock"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0 and result.stdout:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data.get('vulnerabilities', []):
                    vulnerabilities.append(Vulnerability(
                        package=vuln.get('package', 'unknown'),
                        version=vuln.get('installed_version', 'unknown'),
                        vulnerability_id=vuln.get('id', 'unknown'),
                        severity=vuln.get('severity', 'medium').lower(),
                        description=vuln.get('summary', 'Unknown vulnerability'),
                        fix_version=vuln.get('fix_version')
                    ))
            
            print(f"   • Pip-audit scan: additional vulnerabilities checked")
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"   ⚠️  Pip-audit scan failed (may not be installed): {e}")
        
        return vulnerabilities
    
    def run_npm_security_scan(self) -> List[Vulnerability]:
        """Run security scan on NPM dependencies."""
        print("🔍 Running NPM dependency security scan...")
        
        vulnerabilities = []
        npm_dir = self.project_root / "services/ptaas/web"
        
        if not npm_dir.exists():
            print("   ⚠️  NPM directory not found, skipping")
            return vulnerabilities
        
        try:
            cmd = ["npm", "audit", "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=npm_dir)
            
            if result.stdout:
                audit_data = json.loads(result.stdout)
                
                # Parse npm audit format
                for vuln_id, vuln_data in audit_data.get('vulnerabilities', {}).items():
                    if isinstance(vuln_data, dict):
                        vulnerabilities.append(Vulnerability(
                            package=vuln_data.get('name', vuln_id),
                            version=vuln_data.get('range', 'unknown'),
                            vulnerability_id=vuln_id,
                            severity=vuln_data.get('severity', 'medium'),
                            description=vuln_data.get('title', 'Unknown vulnerability'),
                            fix_version=vuln_data.get('fixAvailable', {}).get('version') if isinstance(vuln_data.get('fixAvailable'), dict) else None
                        ))
            
            print(f"   • NPM audit: {len(vulnerabilities)} vulnerabilities found")
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"   ⚠️  NPM audit failed: {e}")
        
        return vulnerabilities
    
    def _map_safety_severity(self, safety_grade: str) -> str:
        """Map safety grades to standard severity levels."""
        mapping = {
            'A': 'low',
            'B': 'low', 
            'C': 'medium',
            'D': 'high',
            'E': 'critical',
            'F': 'critical'
        }
        return mapping.get(safety_grade.upper(), 'medium')
    
    def _get_severity_order(self, severity: str) -> int:
        """Get numeric order for severity sorting."""
        # Map NPM severities to standard ones first
        severity_map = {
            'moderate': 'medium',
            'info': 'low',
            'minor': 'low'
        }
        severity = severity_map.get(severity.lower(), severity.lower())
        
        order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        return order.get(severity, 1)  # Default to medium
    
    def categorize_vulnerabilities(self, vulnerabilities: List[Vulnerability]) -> Dict[str, int]:
        """Categorize vulnerabilities by severity."""
        categories = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        # Map NPM severity levels to standard levels
        severity_map = {
            'moderate': 'medium',
            'info': 'low',
            'minor': 'low'
        }
        
        for vuln in vulnerabilities:
            severity = vuln.severity.lower()
            # Map NPM severities to standard ones
            severity = severity_map.get(severity, severity)
            
            if severity in categories:
                categories[severity] += 1
            else:
                categories['medium'] += 1  # Default unknown to medium
        
        categories['total'] = len(vulnerabilities)
        return categories
    
    def generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        # Python recommendations
        if self.python_vulnerabilities:
            critical_python = sum(1 for v in self.python_vulnerabilities if v.severity == 'critical')
            high_python = sum(1 for v in self.python_vulnerabilities if v.severity == 'high')
            
            if critical_python > 0:
                recommendations.append(f"🚨 URGENT: {critical_python} critical Python vulnerabilities require immediate attention")
            
            if high_python > 0:
                recommendations.append(f"⚠️  HIGH: {high_python} high-severity Python vulnerabilities should be addressed within 24 hours")
            
            # Check for fixable vulnerabilities
            fixable_python = [v for v in self.python_vulnerabilities if v.fix_version]
            if fixable_python:
                recommendations.append(f"🔧 {len(fixable_python)} Python vulnerabilities have available fixes")
        
        # NPM recommendations
        if self.npm_vulnerabilities:
            critical_npm = sum(1 for v in self.npm_vulnerabilities if v.severity == 'critical')
            high_npm = sum(1 for v in self.npm_vulnerabilities if v.severity == 'high')
            
            if critical_npm > 0:
                recommendations.append(f"🚨 URGENT: {critical_npm} critical NPM vulnerabilities require immediate attention")
            
            if high_npm > 0:
                recommendations.append(f"⚠️  HIGH: {high_npm} high-severity NPM vulnerabilities should be addressed within 24 hours")
            
            recommendations.append("💡 Run 'npm audit fix' in services/ptaas/web/ to auto-fix compatible vulnerabilities")
        
        # General recommendations
        if len(self.python_vulnerabilities) + len(self.npm_vulnerabilities) == 0:
            recommendations.append("✅ No vulnerabilities detected - excellent security posture!")
        else:
            recommendations.append("📋 Enable automated dependency updates via Dependabot for continuous security")
            recommendations.append("🔄 Schedule weekly dependency security scans")
            recommendations.append("🛡️  Consider adding dependency pinning for critical packages")
        
        return recommendations
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report."""
        from datetime import datetime
        
        scan_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        report = [
            "# XORB Platform Dependency Security Report",
            f"**Scan Date**: {scan_date}",
            "**Batch 2: Automated Dependency Scanning**",
            "",
            "## 📊 Executive Summary",
            "",
        ]
        
        # Python summary
        python_stats = self.categorize_vulnerabilities(self.python_vulnerabilities)
        report.extend([
            f"### Python Dependencies",
            f"- **Total Vulnerabilities**: {python_stats['total']}",
            f"- **Critical**: {python_stats['critical']} 🚨",
            f"- **High**: {python_stats['high']} ⚠️",
            f"- **Medium**: {python_stats['medium']} 📋",
            f"- **Low**: {python_stats['low']} ℹ️",
            "",
        ])
        
        # NPM summary
        npm_stats = self.categorize_vulnerabilities(self.npm_vulnerabilities)
        report.extend([
            f"### NPM Dependencies",
            f"- **Total Vulnerabilities**: {npm_stats['total']}",
            f"- **Critical**: {npm_stats['critical']} 🚨",
            f"- **High**: {npm_stats['high']} ⚠️",
            f"- **Medium**: {npm_stats['medium']} 📋",
            f"- **Low**: {npm_stats['low']} ℹ️",
            "",
        ])
        
        # Overall risk assessment
        total_critical = python_stats['critical'] + npm_stats['critical']
        total_high = python_stats['high'] + npm_stats['high']
        total_vulnerabilities = python_stats['total'] + npm_stats['total']
        
        if total_critical > 0:
            risk_level = "🚨 **CRITICAL**"
        elif total_high > 0:
            risk_level = "⚠️  **HIGH**"
        elif total_vulnerabilities > 0:
            risk_level = "📋 **MEDIUM**"
        else:
            risk_level = "✅ **LOW**"
        
        report.extend([
            f"### Overall Risk Assessment: {risk_level}",
            "",
            f"**Total Vulnerabilities**: {total_vulnerabilities}",
            "",
        ])
        
        # Detailed vulnerabilities
        if self.python_vulnerabilities:
            report.extend([
                "## 🐍 Python Vulnerabilities",
                "",
                "| Package | Version | Severity | ID | Description | Fix Available |",
                "|---------|---------|----------|----|-----------|--------------| ",
            ])
            
            for vuln in sorted(self.python_vulnerabilities, key=lambda x: self._get_severity_order(x.severity)):
                fix_status = vuln.fix_version if vuln.fix_version else "No"
                report.append(f"| {vuln.package} | {vuln.version} | {vuln.severity.upper()} | {vuln.vulnerability_id} | {vuln.description[:60]}... | {fix_status} |")
            
            report.append("")
        
        if self.npm_vulnerabilities:
            report.extend([
                "## 📦 NPM Vulnerabilities",
                "",
                "| Package | Version | Severity | ID | Description | Fix Available |",
                "|---------|---------|----------|----|-----------|--------------| ",
            ])
            
            for vuln in sorted(self.npm_vulnerabilities, key=lambda x: self._get_severity_order(x.severity)):
                fix_status = vuln.fix_version if vuln.fix_version else "No"
                report.append(f"| {vuln.package} | {vuln.version} | {vuln.severity.upper()} | {vuln.vulnerability_id} | {vuln.description[:60]}... | {fix_status} |")
            
            report.append("")
        
        # Recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            report.extend([
                "## 💡 Security Recommendations",
                "",
            ])
            
            for rec in recommendations:
                report.append(f"- {rec}")
            
            report.append("")
        
        # Remediation commands
        report.extend([
            "## 🔧 Remediation Commands",
            "",
            "### Python Dependencies",
            "```bash",
            "# Install security scanning tools",
            "pip install safety pip-audit",
            "",
            "# Run security scan",
            "safety check --file requirements-unified.lock",
            "pip-audit --requirement requirements-unified.lock",
            "",
            "# Update to fix vulnerabilities (review changes carefully)",
            "pip install --upgrade <package_name>",
            "```",
            "",
            "### NPM Dependencies", 
            "```bash",
            "# Navigate to frontend directory",
            "cd services/ptaas/web",
            "",
            "# Run security audit",
            "npm audit",
            "",
            "# Auto-fix compatible vulnerabilities",
            "npm audit fix",
            "",
            "# Force fix (use with caution)",
            "npm audit fix --force",
            "```",
            "",
            "## 📋 Next Steps",
            "",
            "1. **Address Critical & High Vulnerabilities**: Prioritize immediate fixes",
            "2. **Test Updates**: Ensure fixes don't break functionality", 
            "3. **Update CI/CD**: Integrate security scanning into pipelines",
            "4. **Enable Automation**: Configure Dependabot for ongoing monitoring",
            "5. **Regular Scanning**: Schedule weekly security scans",
            "",
            f"---",
            f"*Report generated by XORB Platform Security Scanner v3.2.0*",
            f"*Scan completed: {scan_date}*"
        ])
        
        return "\n".join(report)
    
    def run_comprehensive_scan(self) -> None:
        """Run comprehensive security scan across all dependencies."""
        print("🚀 Starting XORB Platform Dependency Security Scan")
        print("=" * 60)
        
        # Python scan
        self.python_vulnerabilities = self.run_python_security_scan()
        
        # NPM scan
        self.npm_vulnerabilities = self.run_npm_security_scan()
        
        # Generate and save report
        report_content = self.generate_security_report()
        report_path = self.project_root / "DEPENDENCY_SECURITY_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Summary
        total_vulns = len(self.python_vulnerabilities) + len(self.npm_vulnerabilities)
        critical_vulns = sum(1 for v in self.python_vulnerabilities + self.npm_vulnerabilities if v.severity == 'critical')
        
        print(f"\n📊 Scan Results Summary:")
        print(f"   • Total vulnerabilities found: {total_vulns}")
        print(f"   • Critical vulnerabilities: {critical_vulns}")
        print(f"   • Python vulnerabilities: {len(self.python_vulnerabilities)}")
        print(f"   • NPM vulnerabilities: {len(self.npm_vulnerabilities)}")
        print(f"   • Report saved to: {report_path}")
        
        if critical_vulns > 0:
            print(f"\n🚨 URGENT: {critical_vulns} critical vulnerabilities require immediate attention!")
            return 1
        elif total_vulns > 0:
            print(f"\n⚠️  {total_vulns} vulnerabilities found - review and address as needed")
            return 1
        else:
            print(f"\n✅ No vulnerabilities detected - excellent security posture!")
            return 0

def main():
    """Main entry point for dependency security scanning."""
    scanner = DependencySecurityScanner()
    exit_code = scanner.run_comprehensive_scan()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()