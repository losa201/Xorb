"""
Security Testing Framework

This module implements comprehensive security testing capabilities for the Xorb platform.
"""

import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
import aiohttp
from urllib.parse import urlparse

from xorb.shared.config import Config
from xorb.security.audit import SecurityAudit

logger = logging.getLogger(__name__)

@dataclass
class SecurityTestResult:
    """Represents the result of a security test"""
    test_id: str
    test_name: str
    severity: str  # low/medium/high/critical
    description: str
    evidence: str
    timestamp: datetime
    remediation: str
    component: str
    status: str  # passed/failed/skipped

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'severity': self.severity,
            'description': self.description,
            'evidence': self.evidence,
            'timestamp': self.timestamp.isoformat(),
            'remediation': self.remediation,
            'component': self.component,
            'status': self.status
        }

class SecurityTester:
    """
    Comprehensive security testing framework for Xorb services
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security tester with configuration"""
        self.config = config or Config().get('security_testing', {})
        self.audit = SecurityAudit()
        self.session = requests.Session()
        self.test_results: List[SecurityTestResult] = []
        self.severity_weights = {
            'critical': 10,
            'high': 5,
            'medium': 2,
            'low': 1
        }

    def run_security_tests(self) -> List[SecurityTestResult]:
        """Run all security tests and return results"""
        logger.info("Starting security testing suite")
        
        # Run all test categories
        test_results = []
        test_results.extend(self._test_authentication())
        test_results.extend(self._test_authorization())
        test_results.extend(self._test_data_protection())
        test_results.extend(self._test_input_validation())
        test_results.extend(self._test_api_security())
        test_results.extend(self._test_third_party_integrations())
        test_results.extend(self._test_compliance())
        
        self.test_results = test_results
        self._log_summary()
        
        # Store audit record
        self.audit.log_event(
            'security_testing_complete',
            {'total_tests': len(test_results), 'failed': len([r for r in test_results if r.status == 'failed']}
        )
        
        return test_results

    def _log_summary(self):
        """Log summary of security test results"""
        total = len(self.test_results)
        failed = len([r for r in self.test_results if r.status == 'failed'])
        high_risk = len([r for r in self.test_results if r.severity in ['critical', 'high'] and r.status == 'failed'])
        
        logger.info(f"Security testing complete: {total} tests executed")
        logger.info(f"Failed tests: {failed} ({high_risk} high/critical risk)")
        
        if failed > 0:
            logger.warning("Security test failures detected - recommend immediate remediation")
        
    def _test_authentication(self) -> List[SecurityTestResult]:
        """Test authentication mechanisms"""
        results = []
        
        # Test 1: Weak password policy
        test_id = "AUTH-001"
        try:
            # Check password policy
            pwd_policy = self.config.get('password_policy', {})
            min_length = pwd_policy.get('min_length', 12)
            
            if min_length < 12:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="Weak Password Policy",
                    severity="high",
                    description=f"Password minimum length is {min_length}, recommended minimum is 12",
                    evidence=f"Current policy: {json.dumps(pwd_policy)}",
                    timestamp=datetime.now(),
                    remediation="Update password policy to require at least 12 characters with mixed case, numbers, and special characters",
                    component="authentication",
                    status="failed"
                ))
            else:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="Weak Password Policy",
                    severity="low",
                    description="Password policy meets minimum requirements",
                    evidence="Password minimum length is 12 or more",
                    timestamp=datetime.now(),
                    remediation="",
                    component="authentication",
                    status="passed"
                ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="Weak Password Policy",
                severity="critical",
                description="Error testing password policy",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate authentication configuration",
                component="authentication",
                status="failed"
            ))
        
        # Test 2: MFA enforcement
        test_id = "AUTH-002"
        try:
            # Check MFA configuration
            mfa_required = self.config.get('mfa', {}).get('required', False)
            
            if not mfa_required:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="MFA Not Enforced",
                    severity="high",
                    description="Multi-factor authentication is not required for user accounts",
                    evidence=f"MFA configuration: {json.dumps(self.config.get('mfa', {}))}",
                    timestamp=datetime.now(),
                    remediation="Enable and enforce MFA for all user accounts",
                    component="authentication",
                    status="failed"
                ))
            else:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="MFA Not Enforced",
                    severity="low",
                    description="Multi-factor authentication is enforced",
                    evidence="MFA is required for all accounts",
                    timestamp=datetime.now(),
                    remediation="",
                    component="authentication",
                    status="passed"
                ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="MFA Not Enforced",
                severity="critical",
                description="Error testing MFA configuration",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate authentication configuration",
                component="authentication",
                status="failed"
            ))
        
        return results

    def _test_authorization(self) -> List[SecurityTestResult]:
        """Test authorization mechanisms"""
        results = []
        
        # Test 1: Insecure Direct Object References (IDOR)
        test_id = "AUTHZ-001"
        try:
            # Simulate IDOR attempt
            test_url = self.config.get('test_endpoints', {}).get('user_profile', '/api/v1/users/{user_id}')
            
            # Try to access another user's profile
            response = self.session.get(test_url.format(user_id=1), headers={'Authorization': 'Bearer test_token'})
            
            # If we can access another user's profile, it's a vulnerability
            if response.status_code == 200:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="Insecure Direct Object Reference (IDOR)",
                    severity="high",
                    description="User can access another user's profile without authorization",
                    evidence=f"Successfully accessed user 1's profile with status code {response.status_code}",
                    timestamp=datetime.now(),
                    remediation="Implement proper access control checks to ensure users can only access their own resources",
                    component="authorization",
                    status="failed"
                ))
            else:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="Insecure Direct Object Reference (IDOR)",
                    severity="low",
                    description="Access control appears to be properly implemented",
                    evidence=f"Access to other user's profile denied with status code {response.status_code}",
                    timestamp=datetime.now(),
                    remediation="",
                    component="authorization",
                    status="passed"
                ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="Insecure Direct Object Reference (IDOR)",
                severity="critical",
                description="Error testing IDOR vulnerability",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate authorization configuration",
                component="authorization",
                status="failed"
            ))
        
        return results

    def _test_data_protection(self) -> List[SecurityTestResult]:
        """Test data protection mechanisms"""
        results = []
        
        # Test 1: Sensitive data in logs
        test_id = "DATA-001"
        try:
            # Check if sensitive data is being logged
            log_config = self.config.get('logging', {})
            
            if log_config.get('log_sensitive_data', False):
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="Sensitive Data in Logs",
                    severity="high",
                    description="Application is configured to log sensitive data",
                    evidence=f"Logging configuration: {json.dumps(log_config)}",
                    timestamp=datetime.now(),
                    remediation="Disable logging of sensitive data and implement data masking",
                    component="data_protection",
                    status="failed"
                ))
            else:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="Sensitive Data in Logs",
                    severity="low",
                    description="Application is not configured to log sensitive data",
                    evidence="Sensitive data logging is disabled",
                    timestamp=datetime.now(),
                    remediation="",
                    component="data_protection",
                    status="passed"
                ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="Sensitive Data in Logs",
                severity="critical",
                description="Error testing data protection configuration",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate data protection configuration",
                component="data_protection",
                status="failed"
            ))
        
        return results

    def _test_input_validation(self) -> List[SecurityTestResult]:
        """Test input validation mechanisms"""
        results = []
        
        # Test 1: SQL Injection
        test_id = "INPUT-001"
        try:
            # Simulate SQL injection attempt
            test_url = self.config.get('test_endpoints', {}).get('search', '/api/v1/search')
            
            # Try SQL injection payload
            response = self.session.get(test_url, params={'q': "' OR '1'='1"})
            
            # Check if SQL injection was successful
            if response.status_code == 500 and "SQL" in response.text:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="SQL Injection Vulnerability",
                    severity="high",
                    description="Application is vulnerable to SQL injection attacks",
                    evidence=f"SQL error message returned: {response.text[:200]}...",
                    timestamp=datetime.now(),
                    remediation="Implement proper input validation and use parameterized queries",
                    component="input_validation",
                    status="failed"
                ))
            else:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="SQL Injection Vulnerability",
                    severity="low",
                    description="Application appears to be protected against SQL injection",
                    evidence=f"Response status code {response.status_code}, no SQL error message",
                    timestamp=datetime.now(),
                    remediation="",
                    component="input_validation",
                    status="passed"
                ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="SQL Injection Vulnerability",
                severity="critical",
                description="Error testing SQL injection vulnerability",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate input validation configuration",
                component="input_validation",
                status="failed"
            ))
        
        return results

    def _test_api_security(self) -> List[SecurityTestResult]:
        """Test API security mechanisms"""
        results = []
        
        # Test 1: Rate limiting
        test_id = "API-001"
        try:
            # Check rate limiting configuration
            rate_limit = self.config.get('rate_limiting', {})
            
            if not rate_limit.get('enabled', False):
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="Rate Limiting Not Enforced",
                    severity="medium",
                    description="API rate limiting is not enabled",
                    evidence=f"Rate limiting configuration: {json.dumps(rate_limit)}",
                    timestamp=datetime.now(),
                    remediation="Enable rate limiting to prevent abuse and DDoS attacks",
                    component="api_security",
                    status="failed"
                ))
            else:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="Rate Limiting Not Enforced",
                    severity="low",
                    description="API rate limiting is enabled",
                    evidence="Rate limiting is configured and enforced",
                    timestamp=datetime.now(),
                    remediation="",
                    component="api_security",
                    status="passed"
                ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="Rate Limiting Not Enforced",
                severity="critical",
                description="Error testing API rate limiting",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate API security configuration",
                component="api_security",
                status="failed"
            ))
        
        return results

    def _test_third_party_integrations(self) -> List[SecurityTestResult]:
        """Test third-party integration security"""
        results = []
        
        # Test 1: Insecure third-party API keys
        test_id = "THIRD-001"
        try:
            # Check if third-party API keys are exposed
            third_party = self.config.get('third_party', {})
            
            for service, config in third_party.items():
                if 'api_key' in config and config['api_key'] in ['test_key', '1234567890', 'default_key']:
                    results.append(SecurityTestResult(
                        test_id=f"{test_id}-{service}",
                        test_name=f"Insecure {service} API Key",
                        severity="high",
                        description=f"Using default or weak API key for {service} integration",
                        evidence=f"API key: {config['api_key']}",
                        timestamp=datetime.now(),
                        remediation=f"Replace {service} API key with a strong, randomly generated value",
                        component="third_party_integrations",
                        status="failed"
                    ))
                else:
                    results.append(SecurityTestResult(
                        test_id=f"{test_id}-{service}",
                        test_name=f"Insecure {service} API Key",
                        severity="low",
                        description=f"{service} API key appears to be secure",
                        evidence="API key is not a default value",
                        timestamp=datetime.now(),
                        remediation="",
                        component="third_party_integrations",
                        status="passed"
                    ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="Third-Party API Key Security",
                severity="critical",
                description="Error testing third-party API key security",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate third-party integration configuration",
                component="third_party_integrations",
                status="failed"
            ))
        
        return results

    def _test_compliance(self) -> List[SecurityTestResult]:
        """Test compliance with security standards"""
        results = []
        
        # Test 1: GDPR compliance
        test_id = "COMPLIANCE-001"
        try:
            # Check GDPR compliance configuration
            compliance_config = self.config.get('compliance', {})
            
            if not compliance_config.get('gdpr', {}).get('enabled', False):
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="GDPR Compliance Not Enforced",
                    severity="high",
                    description="GDPR compliance features are not enabled",
                    evidence=f"Compliance configuration: {json.dumps(compliance_config)}",
                    timestamp=datetime.now(),
                    remediation="Enable GDPR compliance features to protect EU user data",
                    component="compliance",
                    status="failed"
                ))
            else:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="GDPR Compliance Not Enforced",
                    severity="low",
                    description="GDPR compliance features are enabled",
                    evidence="GDPR compliance is configured and enforced",
                    timestamp=datetime.now(),
                    remediation="",
                    component="compliance",
                    status="passed"
                ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="GDPR Compliance Not Enforced",
                severity="critical",
                description="Error testing compliance configuration",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate compliance configuration",
                component="compliance",
                status="failed"
            ))
        
        return results

    async def _test_api_security_async(self) -> List[SecurityTestResult]:
        """Async version of API security tests"""
        results = []
        
        # Test 2: CORS misconfiguration
        test_id = "API-002"
        try:
            # Check CORS configuration
            cors_config = self.config.get('cors', {})
            
            if cors_config.get('allow_credentials', False) and '*' in cors_config.get('origins', []):
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="CORS Misconfiguration",
                    severity="high",
                    description="CORS is configured to allow credentials from any origin",
                    evidence=f"CORS configuration: {json.dumps(cors_config)}",
                    timestamp=datetime.now(),
                    remediation="Restrict allowed origins to specific domains and disable allow_credentials unless absolutely necessary",
                    component="api_security",
                    status="failed"
                ))
            else:
                results.append(SecurityTestResult(
                    test_id=test_id,
                    test_name="CORS Misconfiguration",
                    severity="low",
                    description="CORS configuration appears to be secure",
                    evidence="CORS is configured with restricted origins",
                    timestamp=datetime.now(),
                    remediation="",
                    component="api_security",
                    status="passed"
                ))
        except Exception as e:
            logger.error(f"Error running test {test_id}: {str(e)}")
            results.append(SecurityTestResult(
                test_id=test_id,
                test_name="CORS Misconfiguration",
                severity="critical",
                description="Error testing CORS configuration",
                evidence=str(e),
                timestamp=datetime.now(),
                remediation="Investigate API security configuration",
                component="api_security",
                status="failed"
            ))
        
        return results

    def run_security_tests_async(self) -> List[SecurityTestResult]:
        """Run async security tests and return results"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._test_api_security_async())

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report"""
        if not self.test_results:
            self.run_security_tests()
        
        # Calculate security score
        total_score = 0
        max_score = 0
        
        for result in self.test_results:
            max_score += self.severity_weights.get(result.severity, 1)
            if result.status == 'failed':
                total_score += self.severity_weights.get(result.severity, 1)
        
        security_score = 100 - (total_score / max_score * 100) if max_score > 0 else 100
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.test_results),
            'passed': len([r for r in self.test_results if r.status == 'passed']),
            'failed': len([r for r in self.test_results if r.status == 'failed']),
            'security_score': round(security_score, 2),
            'high_risk_issues': len([r for r in self.test_results if r.severity in ['critical', 'high'] and r.status == 'failed']),
            'test_results': [r.to_dict() for r in self.test_results],
            'summary': {
                'security_score': round(security_score, 2),
                'status': 'healthy' if security_score >= 90 else 'needs_improvement' if security_score >= 70 else 'critical'
            }
        }

    def save_security_report(self, file_path: str) -> None:
        """Save security report to file"""
        report = self.generate_security_report()
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Security report saved to {file_path}")
        
        # Store audit record
        self.audit.log_event(
            'security_report_generated',
            {'file_path': file_path, 'security_score': report['summary']['security_score']}
        )

    def run_and_save_security_report(self, file_path: str) -> None:
        """Run security tests and save report"""
        self.run_security_tests()
        self.save_security_report(file_path)

if __name__ == '__main__':
    # Example usage
    tester = SecurityTester()
    tester.run_and_save_security_report('security_report.json')
    
    # Run async tests
    tester.run_security_tests_async()
    
    # Generate and print report
    report = tester.generate_security_report()
    print(json.dumps(report, indent=2))

    # Run security tests and generate report
    tester = SecurityTester()
    tester.run_and_save_security_report('security_report.json')
    
    # Print summary
    print("\nSecurity Score:", report['summary']['security_score'])
    print("Status:", report['summary']['status'].upper())
    print("High Risk Issues:", report['high_risk_issues'])
    print("Total Tests:", report['total_tests'])
    print("Passed:", report['passed'])
    print("Failed:", report['failed'])
    
    # Exit with error code if there are failed tests
    if report['failed'] > 0:
        exit(1)
    else:
        exit(0)