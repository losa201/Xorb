#!/usr/bin/env python3
"""
XORB Security Configuration Validator
Validates security configurations and identifies potential security issues
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SecurityIssue:
    """Represents a security configuration issue"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    file_path: str
    line_number: int = 0
    recommendation: str = ""


class SecurityConfigValidator:
    """Validates XORB security configurations"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.issues: List[SecurityIssue] = []
        
        # Security patterns to detect
        self.insecure_patterns = {
            "dummy_credentials": [
                r"dummy_key",
                r"placeholder_key", 
                r"test_secret",
                r"changeme",
                r"password.*=.*password",
                r"secret.*=.*secret"
            ],
            "hardcoded_secrets": [
                r"api_key.*=.*['\"][a-zA-Z0-9]{20,}['\"]",
                r"password.*=.*['\"][^'\"]+['\"]",
                r"secret.*=.*['\"][^'\"]+['\"]",
                r"token.*=.*['\"][a-zA-Z0-9]{20,}['\"]"
            ],
            "weak_crypto": [
                r"md5|sha1",
                r"tls_disable.*=.*true",
                r"ssl_verify.*=.*false",
                r"verify_ssl.*=.*false"
            ],
            "debug_modes": [
                r"debug.*=.*true",
                r"development.*=.*true",
                r"test_mode.*=.*true"
            ]
        }
    
    def validate_all(self) -> List[SecurityIssue]:
        """Run all security validations"""
        self.issues = []
        
        print("🔍 Starting security configuration validation...")
        
        # Validate source code
        self._validate_source_files()
        
        # Validate configuration files
        self._validate_config_files()
        
        # Validate environment files
        self._validate_env_files()
        
        # Validate Docker configurations
        self._validate_docker_configs()
        
        # Validate Vault configurations
        self._validate_vault_configs()
        
        # Check for exposed secrets
        self._check_exposed_secrets()
        
        print(f"✅ Security validation complete. Found {len(self.issues)} issues.")
        
        return self.issues
    
    def _validate_source_files(self):
        """Validate Python source files for security issues"""
        print("📝 Validating source code...")
        
        for py_file in self.root_path.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                    
                self._check_patterns_in_file(py_file, lines)
                    
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")
    
    def _validate_config_files(self):
        """Validate configuration files"""
        print("⚙️ Validating configuration files...")
        
        config_patterns = ["*.yml", "*.yaml", "*.json", "*.toml", "*.ini"]
        
        for pattern in config_patterns:
            for config_file in self.root_path.rglob(pattern):
                if "venv" in str(config_file):
                    continue
                    
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.splitlines()
                        
                    self._check_patterns_in_file(config_file, lines)
                    
                    # Specific checks for different config types
                    if config_file.suffix in ['.yml', '.yaml']:
                        self._validate_yaml_security(config_file, content)
                    elif config_file.suffix == '.json':
                        self._validate_json_security(config_file, content)
                        
                except Exception as e:
                    print(f"Warning: Could not read {config_file}: {e}")
    
    def _validate_env_files(self):
        """Validate environment files"""
        print("🌍 Validating environment files...")
        
        env_files = [
            ".env", ".env.local", ".env.development", ".env.production",
            ".env.staging", ".env.test"
        ]
        
        for env_file in env_files:
            env_path = self.root_path / env_file
            if env_path.exists():
                try:
                    with open(env_path, 'r') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        # Check for weak passwords
                        if '=' in line:
                            key, value = line.split('=', 1)
                            if any(keyword in key.lower() for keyword in ['password', 'secret', 'key', 'token']):
                                if len(value) < 16:
                                    self.issues.append(SecurityIssue(
                                        severity="MEDIUM",
                                        category="weak_credentials",
                                        description=f"Potentially weak credential: {key}",
                                        file_path=str(env_path),
                                        line_number=i,
                                        recommendation="Use strong, randomly generated credentials (min 16 chars)"
                                    ))
                                
                                if value.lower() in ['password', 'secret', 'changeme', 'admin']:
                                    self.issues.append(SecurityIssue(
                                        severity="CRITICAL",
                                        category="default_credentials",
                                        description=f"Default/weak credential detected: {key}={value}",
                                        file_path=str(env_path),
                                        line_number=i,
                                        recommendation="Replace with strong, unique credential"
                                    ))
                    
                except Exception as e:
                    print(f"Warning: Could not read {env_path}: {e}")
    
    def _validate_docker_configs(self):
        """Validate Docker configurations"""
        print("🐳 Validating Docker configurations...")
        
        docker_files = list(self.root_path.rglob("Dockerfile*")) + list(self.root_path.rglob("docker-compose*.yml"))
        
        for docker_file in docker_files:
            try:
                with open(docker_file, 'r') as f:
                    content = f.read()
                    lines = content.splitlines()
                
                self._check_patterns_in_file(docker_file, lines)
                
                # Docker-specific security checks
                if "Dockerfile" in docker_file.name:
                    self._validate_dockerfile_security(docker_file, lines)
                elif "docker-compose" in docker_file.name:
                    self._validate_compose_security(docker_file, lines)
                    
            except Exception as e:
                print(f"Warning: Could not read {docker_file}: {e}")
    
    def _validate_vault_configs(self):
        """Validate HashiCorp Vault configurations"""
        print("🔐 Validating Vault configurations...")
        
        vault_configs = list(self.root_path.rglob("vault*.hcl"))
        
        for vault_file in vault_configs:
            try:
                with open(vault_file, 'r') as f:
                    content = f.read()
                    lines = content.splitlines()
                
                # Check for TLS disabled
                if "tls_disable = true" in content:
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="insecure_transport",
                        description="TLS disabled in Vault configuration",
                        file_path=str(vault_file),
                        recommendation="Enable TLS for production: tls_disable = false"
                    ))
                
                # Check for development mode
                if "disable_mlock = true" in content and "production" in str(vault_file):
                    self.issues.append(SecurityIssue(
                        severity="MEDIUM",
                        category="development_config",
                        description="Development setting in production Vault config",
                        file_path=str(vault_file),
                        recommendation="Set disable_mlock = false for production"
                    ))
                    
            except Exception as e:
                print(f"Warning: Could not read {vault_file}: {e}")
    
    def _check_exposed_secrets(self):
        """Check for potentially exposed secrets using git"""
        print("🕵️ Checking for exposed secrets...")
        
        try:
            # Use git to check for potential secrets in history
            result = subprocess.run(
                ["git", "log", "--all", "--grep=password", "--grep=secret", "--grep=key", "-i"],
                cwd=self.root_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                self.issues.append(SecurityIssue(
                    severity="MEDIUM",
                    category="potential_exposure",
                    description="Git history contains commits mentioning secrets/passwords",
                    file_path="git_history",
                    recommendation="Review git history and consider secret rotation"
                ))
                
        except Exception:
            # Git not available or not a git repo
            pass
    
    def _check_patterns_in_file(self, file_path: Path, lines: List[str]):
        """Check security patterns in file content"""
        for category, patterns in self.insecure_patterns.items():
            for pattern in patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        severity = self._get_severity_for_category(category)
                        self.issues.append(SecurityIssue(
                            severity=severity,
                            category=category,
                            description=f"Insecure pattern detected: {pattern}",
                            file_path=str(file_path),
                            line_number=i,
                            recommendation=self._get_recommendation_for_category(category)
                        ))
    
    def _validate_dockerfile_security(self, file_path: Path, lines: List[str]):
        """Validate Dockerfile security"""
        has_user = False
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check for USER instruction
            if line.startswith("USER "):
                has_user = True
                user = line.split()[1]
                if user == "root":
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="privilege_escalation",
                        description="Container runs as root user",
                        file_path=str(file_path),
                        line_number=i,
                        recommendation="Use non-root user for security"
                    ))
        
        if not has_user:
            self.issues.append(SecurityIssue(
                severity="MEDIUM",
                category="missing_security",
                description="No USER instruction found - container may run as root",
                file_path=str(file_path),
                recommendation="Add USER instruction to run as non-root"
            ))
    
    def _validate_compose_security(self, file_path: Path, lines: List[str]):
        """Validate docker-compose security"""
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check for privileged mode
            if "privileged: true" in line:
                self.issues.append(SecurityIssue(
                    severity="HIGH",
                    category="privilege_escalation",
                    description="Container runs in privileged mode",
                    file_path=str(file_path),
                    line_number=i,
                    recommendation="Avoid privileged mode unless absolutely necessary"
                ))
    
    def _validate_yaml_security(self, file_path: Path, content: str):
        """Validate YAML security"""
        # Check for exposed ports
        if "ports:" in content and "0.0.0.0:" in content:
            self.issues.append(SecurityIssue(
                severity="MEDIUM",
                category="network_exposure",
                description="Services bound to 0.0.0.0 (all interfaces)",
                file_path=str(file_path),
                recommendation="Bind to specific interfaces when possible"
            ))
    
    def _validate_json_security(self, file_path: Path, content: str):
        """Validate JSON security"""
        # Check for potential secrets in JSON
        if any(keyword in content.lower() for keyword in ['password', 'secret', 'key', 'token']):
            self.issues.append(SecurityIssue(
                severity="MEDIUM",
                category="potential_secrets",
                description="JSON file may contain sensitive information",
                file_path=str(file_path),
                recommendation="Ensure no secrets are stored in JSON files"
            ))
    
    def _get_severity_for_category(self, category: str) -> str:
        """Get severity level for security category"""
        severity_map = {
            "dummy_credentials": "CRITICAL",
            "hardcoded_secrets": "HIGH",
            "weak_crypto": "HIGH",
            "debug_modes": "MEDIUM"
        }
        return severity_map.get(category, "MEDIUM")
    
    def _get_recommendation_for_category(self, category: str) -> str:
        """Get recommendation for security category"""
        recommendations = {
            "dummy_credentials": "Replace dummy credentials with real values or use environment variables",
            "hardcoded_secrets": "Move secrets to environment variables or secret management system",
            "weak_crypto": "Use strong cryptographic algorithms and enable TLS/SSL",
            "debug_modes": "Disable debug modes in production environments"
        }
        return recommendations.get(category, "Review and fix security issue")
    
    def generate_report(self) -> str:
        """Generate security validation report"""
        if not self.issues:
            return "✅ No security issues found!"
        
        # Group issues by severity
        by_severity = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
        for issue in self.issues:
            by_severity[issue.severity].append(issue)
        
        report = []
        report.append("# XORB Security Configuration Validation Report")
        report.append(f"**Total Issues Found**: {len(self.issues)}")
        report.append("")
        
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            issues = by_severity[severity]
            if issues:
                report.append(f"## {severity} Issues ({len(issues)})")
                report.append("")
                
                for issue in issues:
                    report.append(f"### {issue.description}")
                    report.append(f"- **File**: `{issue.file_path}`")
                    if issue.line_number:
                        report.append(f"- **Line**: {issue.line_number}")
                    report.append(f"- **Category**: {issue.category}")
                    report.append(f"- **Recommendation**: {issue.recommendation}")
                    report.append("")
        
        return "\n".join(report)


def main():
    """Main execution function"""
    validator = SecurityConfigValidator()
    issues = validator.validate_all()
    
    # Generate and save report
    report = validator.generate_report()
    
    report_file = Path("security_validation_report.md")
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"📊 Security report saved to: {report_file}")
    
    # Print summary
    if issues:
        critical = sum(1 for i in issues if i.severity == "CRITICAL")
        high = sum(1 for i in issues if i.severity == "HIGH")
        medium = sum(1 for i in issues if i.severity == "MEDIUM")
        low = sum(1 for i in issues if i.severity == "LOW")
        
        print(f"\n🚨 Security Issues Summary:")
        print(f"  Critical: {critical}")
        print(f"  High: {high}")
        print(f"  Medium: {medium}")
        print(f"  Low: {low}")
        
        if critical > 0:
            print("\n⚠️ CRITICAL issues found! Please address immediately.")
            return 1
        elif high > 0:
            print("\n⚠️ HIGH severity issues found! Please address soon.")
            return 1
    else:
        print("\n✅ No security issues found!")
    
    return 0


if __name__ == "__main__":
    exit(main())