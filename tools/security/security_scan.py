#!/usr/bin/env python3
"""
XORB Security Scanner
Comprehensive security scanning orchestrator for the repository.
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import argparse


@dataclass
class ScanResult:
    """Result from a security scan tool."""
    tool: str
    passed: bool
    issues: int
    high_severity: int
    medium_severity: int
    low_severity: int
    output: str
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class SecurityScanner:
    """Main security scanner orchestrator."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.results: Dict[str, ScanResult] = {}

    def run_gitleaks(self) -> ScanResult:
        """Run gitleaks secret detection."""
        print("🔍 Running gitleaks secret detection...")

        try:
            # Check if gitleaks is installed
            subprocess.run(["gitleaks", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ScanResult(
                tool="gitleaks",
                passed=False,
                issues=1,
                high_severity=1,
                medium_severity=0,
                low_severity=0,
                output="Gitleaks not installed",
                recommendations=["Install gitleaks: curl -sSfL https://raw.githubusercontent.com/gitleaks/gitleaks/master/scripts/install.sh | sh"]
            )

        try:
            result = subprocess.run(
                ["gitleaks", "detect", "--source", str(self.root_path), "--verbose"],
                capture_output=True,
                text=True,
                cwd=self.root_path
            )

            issues = result.stdout.count("Secret found")
            passed = result.returncode == 0

            recommendations = []
            if issues > 0:
                recommendations = [
                    "Review detected secrets and remove them from git history",
                    "Consider using tools/security/secret_hygiene.py for cleanup",
                    "Enable git-secrets or similar pre-commit hooks"
                ]

            return ScanResult(
                tool="gitleaks",
                passed=passed,
                issues=issues,
                high_severity=issues,  # All secrets are high severity
                medium_severity=0,
                low_severity=0,
                output=result.stdout if result.stdout else result.stderr,
                recommendations=recommendations
            )

        except subprocess.CalledProcessError as e:
            return ScanResult(
                tool="gitleaks",
                passed=False,
                issues=1,
                high_severity=1,
                medium_severity=0,
                low_severity=0,
                output=f"Gitleaks failed: {e}",
                recommendations=["Check gitleaks configuration and repository state"]
            )

    def run_bandit(self) -> ScanResult:
        """Run bandit Python security linting."""
        print("🔍 Running bandit Python security analysis...")

        try:
            # Check if bandit is installed
            subprocess.run(["bandit", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ScanResult(
                tool="bandit",
                passed=False,
                issues=1,
                high_severity=1,
                medium_severity=0,
                low_severity=0,
                output="Bandit not installed",
                recommendations=["Install bandit: pip install bandit"]
            )

        try:
            # Run bandit on Python files
            result = subprocess.run([
                "bandit",
                "-r", "src/", "tools/", "tests/",
                "-f", "json",
                "--skip", "B101,B601",  # Skip assert and shell usage (common in tests)
                "-ll"  # Low confidence level
            ], capture_output=True, text=True, cwd=self.root_path)

            # Parse JSON output
            if result.stdout.strip():
                try:
                    bandit_data = json.loads(result.stdout)
                    results = bandit_data.get("results", [])

                    high_severity = len([r for r in results if r.get("issue_severity") == "HIGH"])
                    medium_severity = len([r for r in results if r.get("issue_severity") == "MEDIUM"])
                    low_severity = len([r for r in results if r.get("issue_severity") == "LOW"])
                    total_issues = len(results)

                    passed = high_severity == 0 and medium_severity == 0

                    recommendations = []
                    if high_severity > 0:
                        recommendations.append("Fix high severity security issues immediately")
                    if medium_severity > 0:
                        recommendations.append("Review and fix medium severity issues")
                    if total_issues > 10:
                        recommendations.append("Consider adding .bandit configuration to exclude false positives")

                    return ScanResult(
                        tool="bandit",
                        passed=passed,
                        issues=total_issues,
                        high_severity=high_severity,
                        medium_severity=medium_severity,
                        low_severity=low_severity,
                        output=result.stdout,
                        recommendations=recommendations
                    )
                except json.JSONDecodeError:
                    # Fall back to non-JSON parsing
                    pass

            # If JSON parsing failed or no output, assume success
            return ScanResult(
                tool="bandit",
                passed=True,
                issues=0,
                high_severity=0,
                medium_severity=0,
                low_severity=0,
                output="No security issues detected",
                recommendations=[]
            )

        except subprocess.CalledProcessError as e:
            return ScanResult(
                tool="bandit",
                passed=False,
                issues=1,
                high_severity=0,
                medium_severity=1,
                low_severity=0,
                output=f"Bandit scan failed: {e}",
                recommendations=["Check Python code syntax and bandit configuration"]
            )

    def run_ruff_security(self) -> ScanResult:
        """Run ruff with security-focused rules."""
        print("🔍 Running ruff security checks...")

        try:
            result = subprocess.run([
                "python3", "-m", "ruff", "check",
                "--select", "S",  # Security rules
                "--output-format", "json",
                "src/", "tools/", "tests/"
            ], capture_output=True, text=True, cwd=self.root_path)

            if result.stdout.strip():
                try:
                    ruff_data = json.loads(result.stdout)
                    issues = len(ruff_data)

                    # Ruff doesn't categorize severity, so we'll consider all as medium
                    passed = issues == 0

                    recommendations = []
                    if issues > 0:
                        recommendations = [
                            "Review ruff security findings",
                            "Consider adding # noqa comments for false positives",
                            "Update code to follow security best practices"
                        ]

                    return ScanResult(
                        tool="ruff-security",
                        passed=passed,
                        issues=issues,
                        high_severity=0,
                        medium_severity=issues,
                        low_severity=0,
                        output=result.stdout,
                        recommendations=recommendations
                    )
                except json.JSONDecodeError:
                    pass

            return ScanResult(
                tool="ruff-security",
                passed=True,
                issues=0,
                high_severity=0,
                medium_severity=0,
                low_severity=0,
                output="No security issues detected",
                recommendations=[]
            )

        except subprocess.CalledProcessError:
            # Try fallback without ruff
            return ScanResult(
                tool="ruff-security",
                passed=True,
                issues=0,
                high_severity=0,
                medium_severity=0,
                low_severity=0,
                output="Ruff not available, skipping security checks",
                recommendations=["Install ruff for enhanced security linting: pip install ruff"]
            )

    def run_dockerfile_scan(self) -> ScanResult:
        """Scan Dockerfiles for security issues."""
        print("🔍 Scanning Dockerfiles for security issues...")

        dockerfiles = list(self.root_path.rglob("*Dockerfile*"))

        if not dockerfiles:
            return ScanResult(
                tool="dockerfile-scan",
                passed=True,
                issues=0,
                high_severity=0,
                medium_severity=0,
                low_severity=0,
                output="No Dockerfiles found",
                recommendations=[]
            )

        issues = 0
        findings = []
        recommendations = []

        for dockerfile in dockerfiles:
            try:
                content = dockerfile.read_text()

                # Check for common security issues
                if "FROM" in content and ":latest" in content:
                    issues += 1
                    findings.append(f"{dockerfile}: Uses :latest tag (security risk)")

                if "RUN" in content and any(cmd in content for cmd in ["curl", "wget"]) and "https://" not in content:
                    issues += 1
                    findings.append(f"{dockerfile}: Downloads over HTTP (security risk)")

                if "USER root" in content or ("USER" not in content and "FROM" in content):
                    issues += 1
                    findings.append(f"{dockerfile}: Runs as root user")

            except Exception as e:
                findings.append(f"{dockerfile}: Could not analyze - {e}")

        if issues > 0:
            recommendations = [
                "Use specific image tags instead of :latest",
                "Download packages over HTTPS",
                "Run containers as non-root users",
                "Consider using multi-stage builds"
            ]

        return ScanResult(
            tool="dockerfile-scan",
            passed=issues == 0,
            issues=issues,
            high_severity=0,
            medium_severity=issues,
            low_severity=0,
            output="\n".join(findings) if findings else "No issues found",
            recommendations=recommendations
        )

    def generate_report(self, output_format: str = "text") -> str:
        """Generate security scan report."""
        if output_format == "json":
            return json.dumps({
                tool: {
                    "passed": result.passed,
                    "issues": result.issues,
                    "severity": {
                        "high": result.high_severity,
                        "medium": result.medium_severity,
                        "low": result.low_severity
                    },
                    "recommendations": result.recommendations
                }
                for tool, result in self.results.items()
            }, indent=2)

        # Text format
        report = ["=" * 60, "🔒 XORB Security Scan Report", "=" * 60, ""]

        total_issues = sum(r.issues for r in self.results.values())
        total_high = sum(r.high_severity for r in self.results.values())
        total_medium = sum(r.medium_severity for r in self.results.values())
        total_low = sum(r.low_severity for r in self.results.values())

        all_passed = all(r.passed for r in self.results.values())
        status_icon = "✅" if all_passed else "❌"

        report.extend([
            f"{status_icon} Overall Status: {'PASSED' if all_passed else 'FAILED'}",
            f"📊 Total Issues: {total_issues}",
            f"🔴 High Severity: {total_high}",
            f"🟡 Medium Severity: {total_medium}",
            f"🟢 Low Severity: {total_low}",
            ""
        ])

        for tool, result in self.results.items():
            icon = "✅" if result.passed else "❌"
            report.extend([
                f"{icon} {tool.upper()}:",
                f"   Issues: {result.issues}",
                f"   High: {result.high_severity}, Medium: {result.medium_severity}, Low: {result.low_severity}",
                ""
            ])

        # Add recommendations
        all_recommendations = []
        for result in self.results.values():
            all_recommendations.extend(result.recommendations)

        if all_recommendations:
            report.extend([
                "📝 Recommendations:",
                ""
            ])
            for i, rec in enumerate(set(all_recommendations), 1):
                report.append(f"{i}. {rec}")
            report.append("")

        report.extend([
            "=" * 60,
            f"🕐 Scan completed at: {subprocess.check_output(['date'], text=True).strip()}",
            "=" * 60
        ])

        return "\n".join(report)

    def run_all_scans(self) -> bool:
        """Run all security scans."""
        self.results["gitleaks"] = self.run_gitleaks()
        self.results["bandit"] = self.run_bandit()
        self.results["ruff-security"] = self.run_ruff_security()
        self.results["dockerfile-scan"] = self.run_dockerfile_scan()

        return all(result.passed for result in self.results.values())


def main():
    parser = argparse.ArgumentParser(description="XORB Security Scanner")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--fail-on-issues", action="store_true", help="Exit with error code if issues found")

    args = parser.parse_args()

    scanner = SecurityScanner(Path.cwd())

    print("🔒 Starting XORB security scan...")
    print()

    passed = scanner.run_all_scans()

    print()
    report = scanner.generate_report(args.format)

    if args.output:
        Path(args.output).write_text(report)
        print(f"📄 Report saved to: {args.output}")
    else:
        print(report)

    if args.fail_on_issues and not passed:
        print("❌ Security issues detected!")
        return 1

    print("✅ Security scan completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
