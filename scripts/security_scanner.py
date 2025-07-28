#!/usr/bin/env python3
"""
Automated Security Scanner for Xorb 2.0
Implements comprehensive security checks and vulnerability detection
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class XorbSecurityScanner:
    """Comprehensive security scanner for Xorb platform"""

    def __init__(self, base_path: str = "/root/Xorb"):
        self.base_path = Path(base_path)
        self.results: dict[str, Any] = {
            'timestamp': time.time(),
            'scan_type': 'comprehensive',
            'findings': [],
            'summary': {},
            'score': 0
        }

    def scan_hardcoded_secrets(self) -> list[dict[str, Any]]:
        """Scan for hardcoded secrets and credentials"""
        print("🔍 Scanning for hardcoded secrets...")

        findings = []

        # Patterns for different types of secrets
        secret_patterns = {
            'api_key': r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
            'password': r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']([^"\']{8,})["\']',
            'jwt_secret': r'(?i)(jwt[_-]?secret|secret[_-]?key)\s*[:=]\s*["\']([^"\']{20,})["\']',
            'database_url': r'(?i)(database[_-]?url|db[_-]?url)\s*[:=]\s*["\']([^"\']+)["\']',
            'private_key': r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
            'aws_key': r'AKIA[0-9A-Z]{16}',
            'github_token': r'ghp_[a-zA-Z0-9]{36}',
        }

        # File types to scan
        scan_extensions = {'.py', '.js', '.ts', '.yaml', '.yml', '.json', '.env', '.conf', '.config'}

        for file_path in self.base_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in scan_extensions:
                # Skip certain directories
                if any(skip in str(file_path) for skip in ['.git', '__pycache__', 'node_modules', '.venv']):
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')

                    for pattern_name, pattern in secret_patterns.items():
                        matches = re.finditer(pattern, content, re.MULTILINE)

                        for match in matches:
                            # Skip obvious test/example values
                            matched_value = match.group(2) if len(match.groups()) > 1 else match.group(0)
                            if any(test_val in matched_value.lower() for test_val in [
                                'test', 'example', 'placeholder', 'your_', 'replace_', 'demo'
                            ]):
                                continue

                            findings.append({
                                'type': 'hardcoded_secret',
                                'severity': 'high' if pattern_name in ['api_key', 'private_key', 'aws_key'] else 'medium',
                                'file': str(file_path.relative_to(self.base_path)),
                                'line': content[:match.start()].count('\n') + 1,
                                'pattern': pattern_name,
                                'description': f'Potential {pattern_name} found in source code',
                                'recommendation': 'Move to environment variables or secure secret management'
                            })

                except Exception as e:
                    print(f"  ⚠️  Error scanning {file_path}: {e}")

        print(f"  Found {len(findings)} potential secrets")
        return findings

    def scan_insecure_configurations(self) -> list[dict[str, Any]]:
        """Scan for insecure configurations"""
        print("⚙️  Scanning for insecure configurations...")

        findings = []

        # Check Docker configurations
        docker_files = list(self.base_path.rglob('Dockerfile*')) + list(self.base_path.rglob('docker-compose*.yml'))

        for docker_file in docker_files:
            try:
                content = docker_file.read_text()

                # Check for running as root
                if 'USER root' in content or ('USER' not in content and 'Dockerfile' in docker_file.name):
                    findings.append({
                        'type': 'insecure_config',
                        'severity': 'medium',
                        'file': str(docker_file.relative_to(self.base_path)),
                        'description': 'Container may run as root user',
                        'recommendation': 'Create and use non-root user in Dockerfile'
                    })

                # Check for privileged mode
                if 'privileged: true' in content or '--privileged' in content:
                    findings.append({
                        'type': 'insecure_config',
                        'severity': 'high',
                        'file': str(docker_file.relative_to(self.base_path)),
                        'description': 'Container running in privileged mode',
                        'recommendation': 'Remove privileged mode unless absolutely necessary'
                    })

            except Exception as e:
                print(f"  ⚠️  Error scanning {docker_file}: {e}")

        # Check Kubernetes configurations
        k8s_files = list(self.base_path.rglob('*.yaml')) + list(self.base_path.rglob('*.yml'))

        for k8s_file in k8s_files:
            if 'kubernetes' in str(k8s_file) or 'k8s' in str(k8s_file):
                try:
                    content = k8s_file.read_text()

                    # Check for missing security contexts
                    if 'kind: Deployment' in content and 'securityContext' not in content:
                        findings.append({
                            'type': 'insecure_config',
                            'severity': 'medium',
                            'file': str(k8s_file.relative_to(self.base_path)),
                            'description': 'Kubernetes deployment missing security context',
                            'recommendation': 'Add securityContext with runAsNonRoot: true'
                        })

                    # Check for missing resource limits
                    if 'kind: Deployment' in content and 'resources:' not in content:
                        findings.append({
                            'type': 'insecure_config',
                            'severity': 'low',
                            'file': str(k8s_file.relative_to(self.base_path)),
                            'description': 'Kubernetes deployment missing resource limits',
                            'recommendation': 'Add resource requests and limits'
                        })

                except Exception as e:
                    print(f"  ⚠️  Error scanning {k8s_file}: {e}")

        print(f"  Found {len(findings)} configuration issues")
        return findings

    def scan_dependency_vulnerabilities(self) -> list[dict[str, Any]]:
        """Scan for known vulnerabilities in dependencies"""
        print("📦 Scanning dependencies for vulnerabilities...")

        findings = []

        # Check Python dependencies
        requirements_files = list(self.base_path.rglob('requirements*.txt')) + list(self.base_path.rglob('pyproject.toml'))

        for req_file in requirements_files:
            try:
                # Use pip-audit if available
                cmd = f"pip-audit --requirement {req_file} --format json"
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    try:
                        audit_data = json.loads(result.stdout)
                        for vuln in audit_data.get('vulnerabilities', []):
                            findings.append({
                                'type': 'dependency_vulnerability',
                                'severity': vuln.get('severity', 'medium').lower(),
                                'file': str(req_file.relative_to(self.base_path)),
                                'package': vuln.get('package', 'unknown'),
                                'version': vuln.get('installed_version', 'unknown'),
                                'vulnerability_id': vuln.get('id', 'unknown'),
                                'description': vuln.get('description', 'Known vulnerability in dependency'),
                                'recommendation': f"Update {vuln.get('package')} to version {vuln.get('fixed_versions', ['latest'])[0] if vuln.get('fixed_versions') else 'latest'}"
                            })
                    except json.JSONDecodeError:
                        pass

            except subprocess.TimeoutExpired:
                print(f"  ⚠️  Timeout scanning {req_file}")
            except FileNotFoundError:
                print("  ⚠️  pip-audit not available, skipping dependency scan")
                break
            except Exception as e:
                print(f"  ⚠️  Error scanning {req_file}: {e}")

        print(f"  Found {len(findings)} dependency vulnerabilities")
        return findings

    def scan_code_quality_issues(self) -> list[dict[str, Any]]:
        """Scan for code quality and security issues"""
        print("🔍 Scanning for code quality issues...")

        findings = []

        # Patterns for security anti-patterns
        security_patterns = {
            'sql_injection': r'(?i)(execute|query|cursor\.execute)\s*\(\s*["\'][^"\']*\+',
            'command_injection': r'(?i)(subprocess\.|os\.system|os\.popen)\s*\([^)]*\+',
            'weak_crypto': r'(?i)(md5|sha1)\s*\(',
            'debug_mode': r'(?i)debug\s*=\s*True',
            'insecure_random': r'random\.random\(\)',
            'eval_usage': r'(?i)eval\s*\(',
            'pickle_load': r'pickle\.loads?\s*\(',
        }

        python_files = list(self.base_path.rglob('*.py'))

        for py_file in python_files:
            if any(skip in str(py_file) for skip in ['.git', '__pycache__', 'venv', '.venv']):
                continue

            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')

                for pattern_name, pattern in security_patterns.items():
                    matches = re.finditer(pattern, content, re.MULTILINE)

                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1

                        findings.append({
                            'type': 'code_quality',
                            'severity': 'high' if pattern_name in ['sql_injection', 'command_injection', 'eval_usage'] else 'medium',
                            'file': str(py_file.relative_to(self.base_path)),
                            'line': line_num,
                            'issue': pattern_name,
                            'description': f'Potential {pattern_name.replace("_", " ")} detected',
                            'recommendation': self._get_security_recommendation(pattern_name)
                        })

            except Exception as e:
                print(f"  ⚠️  Error scanning {py_file}: {e}")

        print(f"  Found {len(findings)} code quality issues")
        return findings

    def _get_security_recommendation(self, issue_type: str) -> str:
        """Get security recommendation for specific issue type"""
        recommendations = {
            'sql_injection': 'Use parameterized queries or ORM with proper escaping',
            'command_injection': 'Use subprocess with explicit arguments, avoid shell=True',
            'weak_crypto': 'Use SHA-256 or stronger cryptographic hash functions',
            'debug_mode': 'Disable debug mode in production environments',
            'insecure_random': 'Use secrets.SystemRandom() for cryptographic purposes',
            'eval_usage': 'Avoid eval(), use ast.literal_eval() for safe evaluation',
            'pickle_load': 'Use JSON or other safe serialization formats',
        }
        return recommendations.get(issue_type, 'Review and fix security issue')

    def scan_network_security(self) -> list[dict[str, Any]]:
        """Scan for network security issues"""
        print("🌐 Scanning network security configurations...")

        findings = []

        # Check for insecure HTTP usage
        config_files = list(self.base_path.rglob('*.py')) + list(self.base_path.rglob('*.yaml')) + list(self.base_path.rglob('*.yml'))

        for config_file in config_files:
            try:
                content = config_file.read_text()

                # Check for HTTP instead of HTTPS
                http_matches = re.finditer(r'http://(?!localhost|127\.0\.0\.1)', content)
                for match in http_matches:
                    findings.append({
                        'type': 'network_security',
                        'severity': 'medium',
                        'file': str(config_file.relative_to(self.base_path)),
                        'line': content[:match.start()].count('\n') + 1,
                        'description': 'Insecure HTTP URL found',
                        'recommendation': 'Use HTTPS for external communications'
                    })

                # Check for overly permissive CORS
                if 'cors' in content.lower() and ('*' in content or 'allow_all' in content.lower()):
                    findings.append({
                        'type': 'network_security',
                        'severity': 'medium',
                        'file': str(config_file.relative_to(self.base_path)),
                        'description': 'Overly permissive CORS configuration',
                        'recommendation': 'Restrict CORS to specific origins'
                    })

            except Exception as e:
                print(f"  ⚠️  Error scanning {config_file}: {e}")

        print(f"  Found {len(findings)} network security issues")
        return findings

    def generate_security_report(self) -> dict[str, Any]:
        """Generate comprehensive security report"""
        print("\n🔒 Generating Security Report")
        print("=" * 50)

        # Run all scans
        all_findings = []
        all_findings.extend(self.scan_hardcoded_secrets())
        all_findings.extend(self.scan_insecure_configurations())
        all_findings.extend(self.scan_dependency_vulnerabilities())
        all_findings.extend(self.scan_code_quality_issues())
        all_findings.extend(self.scan_network_security())

        # Categorize findings
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        type_counts = {}

        for finding in all_findings:
            severity = finding.get('severity', 'medium')
            finding_type = finding.get('type', 'unknown')

            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[finding_type] = type_counts.get(finding_type, 0) + 1

        # Calculate security score
        total_issues = len(all_findings)
        critical_weight = severity_counts.get('critical', 0) * 20
        high_weight = severity_counts.get('high', 0) * 10
        medium_weight = severity_counts.get('medium', 0) * 5
        low_weight = severity_counts.get('low', 0) * 1

        total_weight = critical_weight + high_weight + medium_weight + low_weight

        # Score out of 100 (higher is better)
        if total_weight == 0:
            security_score = 100
        else:
            security_score = max(0, 100 - min(total_weight, 100))

        self.results.update({
            'findings': all_findings,
            'summary': {
                'total_issues': total_issues,
                'severity_breakdown': severity_counts,
                'type_breakdown': type_counts,
                'security_score': security_score,
                'scan_coverage': {
                    'hardcoded_secrets': True,
                    'configurations': True,
                    'dependencies': True,
                    'code_quality': True,
                    'network_security': True
                }
            },
            'score': security_score
        })

        # Print summary
        print("\n📊 Security Scan Summary")
        print(f"  Total Issues: {total_issues}")
        print(f"  Critical: {severity_counts.get('critical', 0)}")
        print(f"  High: {severity_counts.get('high', 0)}")
        print(f"  Medium: {severity_counts.get('medium', 0)}")
        print(f"  Low: {severity_counts.get('low', 0)}")
        print(f"  Security Score: {security_score}/100")

        status = "🟢 Excellent" if security_score >= 90 else "🟡 Good" if security_score >= 70 else "🟠 Fair" if security_score >= 50 else "🔴 Poor"
        print(f"  Status: {status}")

        return self.results

    def save_report(self, filename: str = None) -> str:
        """Save security report to file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"/root/Xorb/security_report_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n📄 Security report saved to {filename}")
        return filename

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Xorb 2.0 Security Scanner")
    parser.add_argument("--path", default="/root/Xorb", help="Path to scan")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--min-score", type=int, default=70, help="Minimum security score to pass")

    args = parser.parse_args()

    scanner = XorbSecurityScanner(base_path=args.path)
    results = scanner.generate_security_report()

    report_file = scanner.save_report(args.output)

    # Exit with error code if security score is too low
    security_score = results['score']
    if security_score < args.min_score:
        print(f"\n❌ Security score {security_score} below minimum {args.min_score}")
        sys.exit(1)
    else:
        print(f"\n✅ Security score {security_score} meets minimum {args.min_score}")
        sys.exit(0)

if __name__ == "__main__":
    main()
