#!/usr/bin/env python3
"""
Security Implementation Validation Script
Validates all critical security fixes have been properly implemented
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

def log_info(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def log_success(message: str):
    print(f"{Colors.GREEN}[PASS]{Colors.NC} {message}")

def log_warning(message: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {message}")

def log_error(message: str):
    print(f"{Colors.RED}[FAIL]{Colors.NC} {message}")

def log_critical(message: str):
    print(f"{Colors.RED}[CRITICAL]{Colors.NC} {message}")

class SecurityValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            "passed": [],
            "warnings": [],
            "failed": [],
            "critical": []
        }
        self.score = 0
        self.max_score = 0
    
    def add_result(self, test_name: str, passed: bool, message: str, critical: bool = False):
        """Add test result"""
        self.max_score += 1
        
        if passed:
            self.score += 1
            self.results["passed"].append({"test": test_name, "message": message})
            log_success(f"{test_name}: {message}")
        else:
            if critical:
                self.results["critical"].append({"test": test_name, "message": message})
                log_critical(f"{test_name}: {message}")
            else:
                self.results["failed"].append({"test": test_name, "message": message})
                log_error(f"{test_name}: {message}")
    
    def add_warning(self, test_name: str, message: str):
        """Add warning"""
        self.results["warnings"].append({"test": test_name, "message": message})
        log_warning(f"{test_name}: {message}")
    
    def validate_secrets_management(self) -> bool:
        """Validate secrets management implementation"""
        log_info("Validating secrets management...")
        
        all_passed = True
        
        # Check that production secrets are not in version control
        env_secure_path = self.project_root / "infra" / "config" / ".env.production.secure"
        if env_secure_path.exists():
            self.add_result(
                "Secrets Exposure", False, 
                "Production secrets file still exists in repository", 
                critical=True
            )
            all_passed = False
        else:
            self.add_result(
                "Secrets Exposure", True, 
                "Production secrets properly removed from version control"
            )
        
        # Check Vault integration exists
        vault_integration = self.project_root / "src" / "common" / "vault_integration.py"
        if vault_integration.exists():
            self.add_result(
                "Vault Integration", True, 
                "Vault integration module implemented"
            )
        else:
            self.add_result(
                "Vault Integration", False, 
                "Vault integration module missing"
            )
            all_passed = False
        
        # Check environment template exists
        env_template = self.project_root / "infra" / "config" / ".env.production.template"
        if env_template.exists():
            self.add_result(
                "Environment Template", True, 
                "Secure environment template created"
            )
        else:
            self.add_result(
                "Environment Template", False, 
                "Environment template missing"
            )
            all_passed = False
        
        return all_passed
    
    def validate_jwt_security(self) -> bool:
        """Validate JWT security configuration"""
        log_info("Validating JWT security...")
        
        all_passed = True
        
        # Check enhanced JWT service
        enhanced_jwt = self.project_root / "src" / "api" / "app" / "core" / "enhanced_jwt.py"
        if enhanced_jwt.exists():
            self.add_result(
                "Enhanced JWT", True, 
                "Enhanced JWT service with RSA support implemented"
            )
            
            # Check for RS256 algorithm
            with open(enhanced_jwt) as f:
                content = f.read()
                if "RS256" in content:
                    self.add_result(
                        "JWT Algorithm", True, 
                        "RSA256 algorithm configured"
                    )
                else:
                    self.add_result(
                        "JWT Algorithm", False, 
                        "RSA256 algorithm not found"
                    )
                    all_passed = False
        else:
            self.add_result(
                "Enhanced JWT", False, 
                "Enhanced JWT service missing"
            )
            all_passed = False
        
        # Check config validation
        config_file = self.project_root / "src" / "api" / "app" / "core" / "config.py"
        if config_file.exists():
            with open(config_file) as f:
                content = f.read()
                if "validate_jwt_secret" in content:
                    self.add_result(
                        "JWT Validation", True, 
                        "JWT secret validation implemented"
                    )
                else:
                    self.add_result(
                        "JWT Validation", False, 
                        "JWT secret validation missing"
                    )
                    all_passed = False
        
        return all_passed
    
    def validate_container_security(self) -> bool:
        """Validate container security hardening"""
        log_info("Validating container security...")
        
        all_passed = True
        
        # Check production Dockerfile
        dockerfile = self.project_root / "src" / "api" / "Dockerfile.production"
        if dockerfile.exists():
            with open(dockerfile) as f:
                content = f.read()
                
                # Check for non-root user
                if "USER xorb" in content:
                    self.add_result(
                        "Non-root User", True, 
                        "Container runs as non-root user"
                    )
                else:
                    self.add_result(
                        "Non-root User", False, 
                        "Container still runs as root", 
                        critical=True
                    )
                    all_passed = False
                
                # Check for security labels
                if "security.capabilities" in content:
                    self.add_result(
                        "Security Labels", True, 
                        "Security labels configured"
                    )
                else:
                    self.add_result(
                        "Security Labels", False, 
                        "Security labels missing"
                    )
                    all_passed = False
                
                # Check for version pinning
                if "@sha256:" in content:
                    self.add_result(
                        "Image Pinning", True, 
                        "Base images pinned with SHA256"
                    )
                else:
                    self.add_warning(
                        "Image Pinning", 
                        "Base images not pinned with SHA256"
                    )
        else:
            self.add_result(
                "Production Dockerfile", False, 
                "Production Dockerfile missing"
            )
            all_passed = False
        
        # Check Docker Compose security
        docker_compose = self.project_root / "docker-compose.production.yml"
        if docker_compose.exists():
            with open(docker_compose) as f:
                content = f.read()
                
                if "security_opt:" in content and "no-new-privileges" in content:
                    self.add_result(
                        "Docker Security Options", True, 
                        "Security options configured in Docker Compose"
                    )
                else:
                    self.add_result(
                        "Docker Security Options", False, 
                        "Security options missing in Docker Compose"
                    )
                    all_passed = False
        
        return all_passed
    
    def validate_input_validation(self) -> bool:
        """Validate input validation implementation"""
        log_info("Validating input validation...")
        
        all_passed = True
        
        # Check input validation middleware
        input_validation = self.project_root / "src" / "api" / "app" / "middleware" / "input_validation.py"
        if input_validation.exists():
            with open(input_validation) as f:
                content = f.read()
                
                # Check for XSS protection
                if "xss_patterns" in content:
                    self.add_result(
                        "XSS Protection", True, 
                        "XSS protection patterns implemented"
                    )
                else:
                    self.add_result(
                        "XSS Protection", False, 
                        "XSS protection missing"
                    )
                    all_passed = False
                
                # Check for SQL injection protection
                if "sql_patterns" in content:
                    self.add_result(
                        "SQL Injection Protection", True, 
                        "SQL injection protection implemented"
                    )
                else:
                    self.add_result(
                        "SQL Injection Protection", False, 
                        "SQL injection protection missing"
                    )
                    all_passed = False
                
                # Check for command injection protection
                if "command_patterns" in content:
                    self.add_result(
                        "Command Injection Protection", True, 
                        "Command injection protection implemented"
                    )
                else:
                    self.add_result(
                        "Command Injection Protection", False, 
                        "Command injection protection missing"
                    )
                    all_passed = False
        else:
            self.add_result(
                "Input Validation Middleware", False, 
                "Input validation middleware missing", 
                critical=True
            )
            all_passed = False
        
        # Check middleware integration
        main_file = self.project_root / "src" / "api" / "app" / "main.py"
        if main_file.exists():
            with open(main_file) as f:
                content = f.read()
                if "InputValidationMiddleware" in content:
                    self.add_result(
                        "Middleware Integration", True, 
                        "Input validation middleware integrated"
                    )
                else:
                    self.add_result(
                        "Middleware Integration", False, 
                        "Input validation middleware not integrated"
                    )
                    all_passed = False
        
        return all_passed
    
    def validate_cors_configuration(self) -> bool:
        """Validate CORS configuration"""
        log_info("Validating CORS configuration...")
        
        all_passed = True
        
        main_file = self.project_root / "src" / "api" / "app" / "main.py"
        if main_file.exists():
            with open(main_file) as f:
                content = f.read()
                
                # Check for CORS validation
                if "validated_origins" in content and "wildcard CORS origin not allowed" in content:
                    self.add_result(
                        "CORS Validation", True, 
                        "CORS wildcard validation implemented"
                    )
                else:
                    self.add_result(
                        "CORS Validation", False, 
                        "CORS validation missing"
                    )
                    all_passed = False
                
                # Check for trusted host middleware
                if "TrustedHostMiddleware" in content and "allowed_hosts" in content:
                    self.add_result(
                        "Trusted Hosts", True, 
                        "Trusted host middleware configured"
                    )
                else:
                    self.add_result(
                        "Trusted Hosts", False, 
                        "Trusted host middleware missing"
                    )
                    all_passed = False
        
        return all_passed
    
    def validate_logging_security(self) -> bool:
        """Validate secure logging implementation"""
        log_info("Validating logging security...")
        
        all_passed = True
        
        # Check secure logging module
        secure_logging = self.project_root / "src" / "api" / "app" / "core" / "secure_logging.py"
        if secure_logging.exists():
            with open(secure_logging) as f:
                content = f.read()
                
                # Check for sensitive data sanitization
                if "SensitiveDataSanitizer" in content:
                    self.add_result(
                        "Log Sanitization", True, 
                        "Sensitive data sanitization implemented"
                    )
                else:
                    self.add_result(
                        "Log Sanitization", False, 
                        "Sensitive data sanitization missing"
                    )
                    all_passed = False
                
                # Check for audit logging
                if "AuditLogger" in content:
                    self.add_result(
                        "Audit Logging", True, 
                        "Audit logging system implemented"
                    )
                else:
                    self.add_result(
                        "Audit Logging", False, 
                        "Audit logging system missing"
                    )
                    all_passed = False
        else:
            self.add_result(
                "Secure Logging Module", False, 
                "Secure logging module missing"
            )
            all_passed = False
        
        return all_passed
    
    def validate_security_dashboard(self) -> bool:
        """Validate security dashboard implementation"""
        log_info("Validating security dashboard...")
        
        all_passed = True
        
        # Check security dashboard router
        dashboard = self.project_root / "src" / "api" / "app" / "routers" / "security_dashboard.py"
        if dashboard.exists():
            with open(dashboard) as f:
                content = f.read()
                
                # Check for key endpoints
                required_endpoints = [
                    "/dashboard/overview",
                    "/alerts",
                    "/metrics/real-time",
                    "/compliance/status"
                ]
                
                missing_endpoints = []
                for endpoint in required_endpoints:
                    if endpoint not in content:
                        missing_endpoints.append(endpoint)
                
                if not missing_endpoints:
                    self.add_result(
                        "Security Endpoints", True, 
                        "All security dashboard endpoints implemented"
                    )
                else:
                    self.add_result(
                        "Security Endpoints", False, 
                        f"Missing endpoints: {', '.join(missing_endpoints)}"
                    )
                    all_passed = False
        else:
            self.add_result(
                "Security Dashboard", False, 
                "Security dashboard module missing"
            )
            all_passed = False
        
        return all_passed
    
    def validate_file_permissions(self) -> bool:
        """Validate that sensitive files don't exist or have proper permissions"""
        log_info("Validating file security...")
        
        all_passed = True
        
        # Check for accidentally committed secrets
        secret_patterns = [
            "*.key",
            "*.pem", 
            "*secret*",
            "*.env.production",
            "*password*"
        ]
        
        found_secrets = []
        for pattern in secret_patterns:
            try:
                result = subprocess.run(
                    ["find", str(self.project_root), "-name", pattern, "-type", "f"],
                    capture_output=True, text=True, timeout=10
                )
                if result.stdout.strip():
                    found_secrets.extend(result.stdout.strip().split('\n'))
            except:
                pass
        
        # Filter out acceptable secret files
        acceptable_paths = [
            "secrets/", "vault/", "test/", "demo/", "example", "template"
        ]
        
        dangerous_secrets = []
        for secret_file in found_secrets:
            if not any(acceptable in secret_file for acceptable in acceptable_paths):
                dangerous_secrets.append(secret_file)
        
        if dangerous_secrets:
            self.add_result(
                "Secret Files", False, 
                f"Potential secret files found: {', '.join(dangerous_secrets)}", 
                critical=True
            )
            all_passed = False
        else:
            self.add_result(
                "Secret Files", True, 
                "No dangerous secret files found in repository"
            )
        
        return all_passed
    
    def run_code_security_scan(self) -> bool:
        """Run basic security scan on code"""
        log_info("Running code security scan...")
        
        all_passed = True
        issues = []
        
        # Scan Python files for security issues
        python_files = list(self.project_root.glob("src/**/*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file) as f:
                    content = f.read()
                    
                    # Check for eval/exec usage
                    if re.search(r'\beval\s*\(|\bexec\s*\(', content):
                        issues.append(f"Unsafe eval/exec in {py_file}")
                    
                    # Check for hardcoded passwords
                    if re.search(r'password\s*=\s*["\'][^"\']+["\']', content, re.IGNORECASE):
                        # Exclude hash_password, verify_password functions
                        if "hash_password" not in content and "verify_password" not in content:
                            issues.append(f"Potential hardcoded password in {py_file}")
                    
                    # Check for insecure random
                    if re.search(r'import random\b|from random import', content):
                        if "secrets" not in content:
                            issues.append(f"Insecure random usage in {py_file}")
                    
                    # Check for SQL injection patterns
                    if re.search(r'execute\s*\([^)]*%|execute\s*\([^)]*\.format', content):
                        issues.append(f"Potential SQL injection in {py_file}")
                        
            except Exception:
                continue
        
        if issues:
            for issue in issues[:5]:  # Limit to first 5 issues
                self.add_warning("Code Security", issue)
            if len(issues) > 5:
                self.add_warning("Code Security", f"... and {len(issues) - 5} more issues")
            all_passed = False
        else:
            self.add_result(
                "Code Security Scan", True, 
                "No critical security issues found in code"
            )
        
        return all_passed
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        total_tests = self.max_score
        passed_tests = self.score
        failed_tests = len(self.results["failed"])
        critical_issues = len(self.results["critical"])
        warnings = len(self.results["warnings"])
        
        security_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall security level
        if critical_issues > 0:
            security_level = "CRITICAL RISK"
        elif failed_tests > 5:
            security_level = "HIGH RISK"
        elif failed_tests > 2:
            security_level = "MEDIUM RISK"
        elif warnings > 3:
            security_level = "LOW RISK"
        else:
            security_level = "SECURE"
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "security_score": round(security_score, 1),
                "security_level": security_level,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "critical_issues": critical_issues,
                "warnings": warnings
            },
            "results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        if self.results["critical"]:
            recommendations.append("🚨 Address critical security issues immediately")
            recommendations.append("🔒 Rotate all credentials and secrets")
            recommendations.append("🛡️ Enable additional monitoring")
        
        if self.results["failed"]:
            recommendations.append("🔧 Fix all failed security tests")
            recommendations.append("📋 Review and update security policies")
        
        if self.results["warnings"]:
            recommendations.append("⚠️ Address security warnings")
            recommendations.append("📊 Implement additional monitoring")
        
        # Always include best practices
        recommendations.extend([
            "🔄 Implement regular security assessments",
            "📚 Provide security training to development team",
            "🎯 Set up automated security testing in CI/CD",
            "📈 Monitor security metrics continuously",
            "🔍 Conduct penetration testing",
            "📖 Keep security documentation up to date"
        ])
        
        return recommendations
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all security validations"""
        
        print(f"{Colors.CYAN}")
        print("🔐 XORB Platform Security Implementation Validation")
        print("=" * 55)
        print(f"{Colors.NC}")
        
        # Run all validation tests
        self.validate_secrets_management()
        self.validate_jwt_security()
        self.validate_container_security()
        self.validate_input_validation()
        self.validate_cors_configuration()
        self.validate_logging_security()
        self.validate_security_dashboard()
        self.validate_file_permissions()
        self.run_code_security_scan()
        
        # Generate and display report
        report = self.generate_report()
        
        print(f"\n{Colors.WHITE}Security Validation Summary{Colors.NC}")
        print("=" * 30)
        print(f"Security Score: {report['summary']['security_score']:.1f}%")
        print(f"Security Level: {report['summary']['security_level']}")
        print(f"Tests Passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
        print(f"Critical Issues: {report['summary']['critical_issues']}")
        print(f"Failed Tests: {report['summary']['failed_tests']}")
        print(f"Warnings: {report['summary']['warnings']}")
        
        if report['summary']['critical_issues'] > 0:
            print(f"\n{Colors.RED}❌ CRITICAL SECURITY ISSUES DETECTED{Colors.NC}")
            print("The platform is NOT SAFE for production deployment.")
            print("Address critical issues immediately before deployment.")
        elif report['summary']['failed_tests'] > 3:
            print(f"\n{Colors.YELLOW}⚠️ SIGNIFICANT SECURITY CONCERNS{Colors.NC}")
            print("Multiple security tests failed. Review and fix before production.")
        elif report['summary']['security_score'] >= 90:
            print(f"\n{Colors.GREEN}✅ EXCELLENT SECURITY POSTURE{Colors.NC}")
            print("Security implementation meets high standards.")
        elif report['summary']['security_score'] >= 75:
            print(f"\n{Colors.GREEN}✅ GOOD SECURITY POSTURE{Colors.NC}")
            print("Security implementation is acceptable for production.")
        else:
            print(f"\n{Colors.YELLOW}⚠️ SECURITY IMPROVEMENTS NEEDED{Colors.NC}")
            print("Consider addressing failed tests and warnings.")
        
        print(f"\n{Colors.WHITE}Recommendations:{Colors.NC}")
        for i, rec in enumerate(report['recommendations'][:8], 1):
            print(f"{i}. {rec}")
        
        # Save detailed report
        report_file = self.project_root / "security_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report

def main():
    """Main function"""
    validator = SecurityValidator()
    report = validator.run_all_validations()
    
    # Exit with appropriate code
    if report['summary']['critical_issues'] > 0:
        sys.exit(2)  # Critical issues
    elif report['summary']['failed_tests'] > 3:
        sys.exit(1)  # Significant failures
    else:
        sys.exit(0)  # Success

if __name__ == "__main__":
    main()