#!/usr/bin/env python3
"""
XORB Comprehensive Security Hardening and Secret Scanner
Scans for hardcoded secrets, credentials, and security vulnerabilities
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityHardening:
    """Comprehensive security hardening and secret scanning"""

    def __init__(self, base_path: str = "/root/Xorb"):
        self.base_path = Path(base_path)
        self.secrets_found = []
        self.security_issues = []
        self.patterns = self._load_secret_patterns()
        self.exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}
        self.exclude_files = {'.pyc', '.pyo', '.so', '.dll', '.exe', '.bin'}

    def _load_secret_patterns(self) -> dict[str, re.Pattern]:
        """Load patterns for detecting secrets and credentials"""
        patterns = {
            'api_key': re.compile(r'(?i)(api[_-]?key|apikey)[\s]*[=:]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?'),
            'password': re.compile(r'(?i)(password|passwd|pwd)[\s]*[=:]\s*["\']([^"\']{4,})["\']'),
            'secret': re.compile(r'(?i)(secret|token)[\s]*[=:]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?'),
            'private_key': re.compile(r'-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----'),
            'connection_string': re.compile(r'(?i)(mongodb|mysql|postgres|redis)://[^\s"\']+'),
            'jwt_token': re.compile(r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*'),
            'aws_key': re.compile(r'AKIA[0-9A-Z]{16}'),
            'github_token': re.compile(r'ghp_[A-Za-z0-9]{36}'),
            'slack_token': re.compile(r'xox[baprs]-[A-Za-z0-9-]+'),
            'discord_token': re.compile(r'[MNO][A-Za-z\d]{23}\.[\w-]{6}\.[\w-]{27}'),
            'hardcoded_ip': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        }
        return patterns

    def scan_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Scan a single file for secrets and security issues"""
        findings = []

        try:
            if file_path.suffix.lower() in self.exclude_files:
                return findings

            with open(file_path, encoding='utf-8', errors='ignore') as f:
                content = f.read()

            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern_name, pattern in self.patterns.items():
                    matches = pattern.finditer(line)
                    for match in matches:
                        # Skip obvious false positives
                        if self._is_false_positive(pattern_name, match.group()):
                            continue

                        finding = {
                            'file': str(file_path.relative_to(self.base_path)),
                            'line': line_num,
                            'type': pattern_name,
                            'match': match.group()[:100],  # Truncate for safety
                            'severity': self._get_severity(pattern_name),
                            'recommendation': self._get_recommendation(pattern_name)
                        }
                        findings.append(finding)

        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")

        return findings

    def _is_false_positive(self, pattern_name: str, match: str) -> bool:
        """Check if a match is likely a false positive"""
        false_positives = {
            'password': ['password', 'your_password', 'example', 'test', '****', 'xxxx', 'changeme'],
            'api_key': ['your_api_key', 'api_key_here', 'example_key', 'test_key'],
            'secret': ['your_secret', 'secret_here', 'example_secret', 'test_secret'],
            'hardcoded_ip': ['127.0.0.1', '0.0.0.0', '255.255.255.255', '192.168.1.1', '10.0.0.1']
        }

        if pattern_name in false_positives:
            return any(fp in match.lower() for fp in false_positives[pattern_name])

        # Skip very short matches
        if len(match.strip()) < 4:
            return True

        return False

    def _get_severity(self, pattern_name: str) -> str:
        """Get severity level for pattern type"""
        severity_map = {
            'private_key': 'CRITICAL',
            'aws_key': 'CRITICAL',
            'github_token': 'CRITICAL',
            'jwt_token': 'HIGH',
            'api_key': 'HIGH',
            'password': 'HIGH',
            'secret': 'HIGH',
            'connection_string': 'MEDIUM',
            'slack_token': 'MEDIUM',
            'discord_token': 'MEDIUM',
            'hardcoded_ip': 'LOW',
            'email': 'LOW'
        }
        return severity_map.get(pattern_name, 'MEDIUM')

    def _get_recommendation(self, pattern_name: str) -> str:
        """Get security recommendation for pattern type"""
        recommendations = {
            'private_key': 'Move private keys to secure key management system',
            'aws_key': 'Rotate AWS keys immediately and use IAM roles',
            'github_token': 'Revoke GitHub token and use environment variables',
            'jwt_token': 'Remove hardcoded JWT tokens, use secure generation',
            'api_key': 'Move to environment variables or secret vault',
            'password': 'Use environment variables or encrypted configuration',
            'secret': 'Store in secure secret management system',
            'connection_string': 'Use environment variables for connection strings',
            'slack_token': 'Move to environment variables',
            'discord_token': 'Move to environment variables',
            'hardcoded_ip': 'Use configuration files or environment variables',
            'email': 'Consider if email exposure is necessary'
        }
        return recommendations.get(pattern_name, 'Review and secure if sensitive')

    def scan_directory(self) -> dict[str, Any]:
        """Scan entire directory for security issues"""
        logger.info(f"Starting security scan of {self.base_path}")

        findings = []
        file_count = 0

        for root, dirs, files in os.walk(self.base_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                file_path = Path(root) / file
                file_count += 1

                file_findings = self.scan_file(file_path)
                findings.extend(file_findings)

        # Categorize findings by severity
        critical = [f for f in findings if f['severity'] == 'CRITICAL']
        high = [f for f in findings if f['severity'] == 'HIGH']
        medium = [f for f in findings if f['severity'] == 'MEDIUM']
        low = [f for f in findings if f['severity'] == 'LOW']

        results = {
            'scan_time': datetime.now().isoformat(),
            'files_scanned': file_count,
            'total_findings': len(findings),
            'critical': len(critical),
            'high': len(high),
            'medium': len(medium),
            'low': len(low),
            'findings': findings,
            'summary': {
                'critical_issues': critical[:10],  # Show top 10 critical
                'high_issues': high[:10],  # Show top 10 high
                'recommendations': self._generate_recommendations(findings)
            }
        }

        logger.info(f"Scan completed: {len(findings)} findings across {file_count} files")
        return results

    def _generate_recommendations(self, findings: list[dict[str, Any]]) -> list[str]:
        """Generate security recommendations based on findings"""
        recommendations = set()

        if any(f['severity'] == 'CRITICAL' for f in findings):
            recommendations.add("🚨 CRITICAL: Immediately rotate all exposed credentials")
            recommendations.add("🔒 Implement comprehensive secret management system")

        if any(f['type'] == 'password' for f in findings):
            recommendations.add("🔐 Move all passwords to environment variables")

        if any(f['type'] == 'api_key' for f in findings):
            recommendations.add("🔑 Implement secure API key management")

        if any(f['type'] == 'connection_string' for f in findings):
            recommendations.add("🔗 Secure database connection strings")

        recommendations.add("📋 Create .env.template files for configuration")
        recommendations.add("🛡️  Implement pre-commit hooks for secret detection")
        recommendations.add("🔄 Regular security audits and secret rotation")

        return list(recommendations)

    def create_secure_templates(self) -> None:
        """Create secure configuration templates"""
        templates = {
            '.env.template': """# XORB Environment Configuration Template
# Copy to .env and fill in your values

# Database Configuration
POSTGRES_PASSWORD=your_secure_postgres_password
REDIS_PASSWORD=your_secure_redis_password

# API Keys
NVIDIA_API_KEY=your_nvidia_api_key
OPENAI_API_KEY=your_openai_api_key

# Security
JWT_SECRET=your_jwt_secret_32_chars_min
ENCRYPTION_KEY=your_encryption_key_here

# Monitoring
GRAFANA_ADMIN_PASSWORD=your_grafana_admin_password

# Network
ALLOWED_HOSTS=localhost,127.0.0.1
""",
            'secrets.json.template': """{
  "_comment": "XORB Secrets Template - Copy to secrets.json and fill values",
  "database": {
    "postgres_password": "your_secure_password",
    "redis_password": "your_secure_password"
  },
  "api_keys": {
    "nvidia": "your_nvidia_api_key",
    "openai": "your_openai_api_key"
  },
  "security": {
    "jwt_secret": "your_jwt_secret_minimum_32_characters",
    "encryption_key": "your_encryption_key_here"
  },
  "monitoring": {
    "grafana_admin_password": "your_secure_password"
  }
}""",
            'vault.yaml.template': """# XORB Vault Configuration Template
# Copy to vault.yaml and configure your secret management

vault:
  backend: "file"  # or "hashicorp", "aws", "azure"
  path: "/root/Xorb/secrets/vault"
  encryption: "aes256"

secrets:
  database:
    postgres_password: "{{ vault.secret }}"
    redis_password: "{{ vault.secret }}"

  api_keys:
    nvidia: "{{ vault.secret }}"
    openai: "{{ vault.secret }}"

  security:
    jwt_secret: "{{ vault.secret }}"
    encryption_key: "{{ vault.secret }}"
"""
        }

        for filename, content in templates.items():
            template_path = self.base_path / filename
            with open(template_path, 'w') as f:
                f.write(content)
            logger.info(f"Created secure template: {filename}")

    def fix_common_issues(self, findings: list[dict[str, Any]]) -> list[str]:
        """Attempt to fix common security issues"""
        fixes_applied = []

        # Group findings by file
        files_to_fix = {}
        for finding in findings:
            if finding['severity'] in ['CRITICAL', 'HIGH']:
                file_path = finding['file']
                if file_path not in files_to_fix:
                    files_to_fix[file_path] = []
                files_to_fix[file_path].append(finding)

        for file_path, file_findings in files_to_fix.items():
            full_path = self.base_path / file_path
            if self._should_auto_fix(file_path):
                fixes = self._apply_fixes(full_path, file_findings)
                fixes_applied.extend(fixes)

        return fixes_applied

    def _should_auto_fix(self, file_path: str) -> bool:
        """Determine if file should be auto-fixed"""
        # Only auto-fix configuration files and templates
        safe_files = ['.env', 'config.py', 'settings.py', '.yml', '.yaml']
        return any(file_path.endswith(ext) for ext in safe_files)

    def _apply_fixes(self, file_path: Path, findings: list[dict[str, Any]]) -> list[str]:
        """Apply automatic fixes to a file"""
        fixes = []

        try:
            with open(file_path) as f:
                content = f.read()

            original_content = content

            for finding in findings:
                if finding['type'] == 'password' and 'password=' in finding['match']:
                    # Replace hardcoded password with environment variable
                    content = content.replace(
                        finding['match'],
                        "password=os.getenv('PASSWORD', 'secure_default')"
                    )
                    fixes.append(f"Fixed hardcoded password in {file_path}")

                elif finding['type'] == 'api_key' and 'api_key=' in finding['match']:
                    # Replace hardcoded API key with environment variable
                    content = content.replace(
                        finding['match'],
                        "api_key=os.getenv('API_KEY', '')"
                    )
                    fixes.append(f"Fixed hardcoded API key in {file_path}")

            # Only write if changes were made
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)

        except Exception as e:
            logger.warning(f"Error applying fixes to {file_path}: {e}")

        return fixes

def main():
    """Main security hardening execution"""
    print("🔒 XORB Comprehensive Security Hardening")
    print("========================================")

    hardening = SecurityHardening()

    # Perform security scan
    print("🔍 Scanning for security vulnerabilities...")
    results = hardening.scan_directory()

    # Create secure templates
    print("📋 Creating secure configuration templates...")
    hardening.create_secure_templates()

    # Apply automatic fixes
    print("🔧 Applying automatic security fixes...")
    fixes = hardening.fix_common_issues(results['findings'])

    # Save results
    results_path = Path("/root/Xorb/logs/security_scan_results.json")
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n📊 Security Scan Summary:")
    print(f"Files scanned: {results['files_scanned']}")
    print(f"Total findings: {results['total_findings']}")
    print(f"Critical: {results['critical']}")
    print(f"High: {results['high']}")
    print(f"Medium: {results['medium']}")
    print(f"Low: {results['low']}")

    if fixes:
        print(f"\n🔧 Fixes applied: {len(fixes)}")
        for fix in fixes[:5]:  # Show first 5 fixes
            print(f"  - {fix}")

    print("\n📋 Recommendations:")
    for rec in results['summary']['recommendations'][:5]:
        print(f"  {rec}")

    print(f"\n📄 Full report saved to: {results_path}")

    # Determine security score
    total_issues = results['total_findings']
    critical_weight = results['critical'] * 10
    high_weight = results['high'] * 5
    medium_weight = results['medium'] * 2
    low_weight = results['low'] * 1

    weighted_score = critical_weight + high_weight + medium_weight + low_weight
    max_score = results['files_scanned'] * 2  # Assume 2 potential issues per file

    security_score = max(0, 100 - (weighted_score / max_score * 100)) if max_score > 0 else 100

    print(f"\n🛡️  Security Score: {security_score:.1f}%")

    if security_score >= 90:
        print("✅ EXCELLENT - Enterprise security standards met")
    elif security_score >= 80:
        print("✅ GOOD - Minor security improvements needed")
    elif security_score >= 70:
        print("⚠️  FAIR - Moderate security hardening required")
    else:
        print("❌ POOR - Significant security improvements needed")

    return security_score

if __name__ == "__main__":
    main()
