"""
Comprehensive Security Testing Framework for XORB Platform
Tests authentication, authorization, encryption, and security vulnerabilities
"""

import asyncio
import pytest
import time
import hashlib
import secrets
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import re
import base64
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis
import jwt
from cryptography.fernet import Fernet

from src.api.app.services.unified_auth_service_consolidated import UnifiedAuthService
from src.api.app.domain.entities import User, AuthToken
from src.api.app.domain.exceptions import InvalidCredentials, AccountLocked, ValidationError
from src.common.jwt_manager import JWTManager


class SecurityTestType(Enum):
    """Types of security tests"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    INPUT_VALIDATION = "input_validation"
    SESSION_MANAGEMENT = "session_management"
    PASSWORD_SECURITY = "password_security"
    TOKEN_SECURITY = "token_security"
    RATE_LIMITING = "rate_limiting"
    INJECTION_ATTACKS = "injection_attacks"
    BRUTE_FORCE = "brute_force"


class SecurityTestSeverity(Enum):
    """Security test severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityTestResult:
    """Result of a security test"""
    test_name: str
    test_type: SecurityTestType
    severity: SecurityTestSeverity
    passed: bool
    description: str
    details: Dict[str, Any]
    recommendations: List[str]
    execution_time: float
    timestamp: datetime


@dataclass
class VulnerabilityReport:
    """Security vulnerability report"""
    vulnerability_id: str
    title: str
    severity: SecurityTestSeverity
    description: str
    impact: str
    affected_components: List[str]
    reproduction_steps: List[str]
    remediation: List[str]
    references: List[str]


class SecurityTestFramework:
    """Comprehensive security testing framework"""
    
    def __init__(self):
        self.test_results: List[SecurityTestResult] = []
        self.vulnerabilities: List[VulnerabilityReport] = []
        
    async def setup(self):
        """Setup security testing environment"""
        # Setup Redis client for testing
        self.redis_client = redis.from_url("redis://localhost:6379/3")  # Security test DB
        await self.redis_client.flushdb()
        
        # Setup mock repositories
        self.mock_user_repo = AsyncMock()
        self.mock_token_repo = AsyncMock()
        
        # Setup unified auth service
        self.auth_service = UnifiedAuthService(
            user_repository=self.mock_user_repo,
            token_repository=self.mock_token_repo,
            redis_client=self.redis_client,
            secret_key="security-test-secret-key",
            algorithm="HS256",
            access_token_expire_minutes=30
        )
        
        # Setup JWT manager
        self.jwt_manager = JWTManager()
        
        print("🔒 Security testing environment setup complete")
    
    async def teardown(self):
        """Cleanup security testing environment"""
        await self.redis_client.flushdb()
        await self.redis_client.close()
        print("🔒 Security testing environment cleaned up")
    
    def _create_test_result(
        self,
        test_name: str,
        test_type: SecurityTestType,
        severity: SecurityTestSeverity,
        passed: bool,
        description: str,
        details: Dict[str, Any] = None,
        recommendations: List[str] = None,
        execution_time: float = 0.0
    ) -> SecurityTestResult:
        """Create a security test result"""
        return SecurityTestResult(
            test_name=test_name,
            test_type=test_type,
            severity=severity,
            passed=passed,
            description=description,
            details=details or {},
            recommendations=recommendations or [],
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )
    
    async def test_password_security(self) -> List[SecurityTestResult]:
        """Test password security mechanisms"""
        results = []
        
        # Test 1: Password strength validation
        start_time = time.time()
        weak_passwords = [
            "123456",
            "password",
            "abc123",
            "qwerty",
            "admin",
            "Password123",  # Missing special chars
            "password123!",  # Missing uppercase
            "PASSWORD123!",  # Missing lowercase
            "Passw0rd!"  # Too short for our requirements
        ]
        
        weak_password_accepted = []
        for weak_password in weak_passwords:
            try:
                await self.auth_service.hash_password(weak_password)
                weak_password_accepted.append(weak_password)
            except ValidationError:
                pass  # Expected behavior
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="password_strength_validation",
            test_type=SecurityTestType.PASSWORD_SECURITY,
            severity=SecurityTestSeverity.HIGH,
            passed=len(weak_password_accepted) == 0,
            description="Test that weak passwords are rejected",
            details={
                "weak_passwords_tested": len(weak_passwords),
                "weak_passwords_accepted": weak_password_accepted
            },
            recommendations=[
                "Ensure all weak passwords are rejected",
                "Implement comprehensive password complexity rules"
            ] if weak_password_accepted else [],
            execution_time=execution_time
        ))
        
        # Test 2: Password hashing security
        start_time = time.time()
        test_password = "SecureTestPassword123!"
        
        # Test that same password produces different hashes
        hash1 = await self.auth_service.hash_password(test_password)
        hash2 = await self.auth_service.hash_password(test_password)
        
        # Test hash format (should be Argon2)
        is_argon2 = hash1.startswith("$argon2")
        different_hashes = hash1 != hash2
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="password_hashing_security",
            test_type=SecurityTestType.ENCRYPTION,
            severity=SecurityTestSeverity.CRITICAL,
            passed=is_argon2 and different_hashes,
            description="Test password hashing uses secure algorithms",
            details={
                "uses_argon2": is_argon2,
                "different_hashes_for_same_password": different_hashes,
                "hash_example": hash1[:50] + "..." if len(hash1) > 50 else hash1
            },
            recommendations=[
                "Use Argon2 for password hashing",
                "Ensure salts are unique for each hash"
            ] if not (is_argon2 and different_hashes) else [],
            execution_time=execution_time
        ))
        
        # Test 3: Password verification timing attack resistance
        start_time = time.time()
        correct_password = "CorrectPassword123!"
        wrong_password = "WrongPassword123!"
        
        # Hash a password
        correct_hash = await self.auth_service.hash_password(correct_password)
        
        # Time verification of correct password
        correct_times = []
        for _ in range(10):
            verify_start = time.perf_counter()
            await self.auth_service.verify_password(correct_password, correct_hash)
            verify_end = time.perf_counter()
            correct_times.append(verify_end - verify_start)
        
        # Time verification of wrong password
        wrong_times = []
        for _ in range(10):
            verify_start = time.perf_counter()
            await self.auth_service.verify_password(wrong_password, correct_hash)
            verify_end = time.perf_counter()
            wrong_times.append(verify_end - verify_start)
        
        # Calculate timing difference
        avg_correct_time = sum(correct_times) / len(correct_times)
        avg_wrong_time = sum(wrong_times) / len(wrong_times)
        timing_difference = abs(avg_correct_time - avg_wrong_time)
        
        execution_time = time.time() - start_time
        
        # Small timing differences are acceptable
        timing_attack_resistant = timing_difference < 0.01  # 10ms threshold
        
        results.append(self._create_test_result(
            test_name="password_timing_attack_resistance",
            test_type=SecurityTestType.PASSWORD_SECURITY,
            severity=SecurityTestSeverity.MEDIUM,
            passed=timing_attack_resistant,
            description="Test resistance to timing attacks in password verification",
            details={
                "avg_correct_verification_time": avg_correct_time,
                "avg_wrong_verification_time": avg_wrong_time,
                "timing_difference": timing_difference,
                "threshold": 0.01
            },
            recommendations=[
                "Ensure password verification has constant time complexity",
                "Use secure comparison functions"
            ] if not timing_attack_resistant else [],
            execution_time=execution_time
        ))
        
        return results
    
    async def test_jwt_security(self) -> List[SecurityTestResult]:
        """Test JWT token security"""
        results = []
        
        # Test 1: JWT token structure and claims
        start_time = time.time()
        test_user = User(
            id="jwt-test-user",
            username="jwttest",
            email="jwt@test.com",
            roles=["user"]
        )
        
        access_token, _ = self.auth_service.create_access_token(test_user)
        
        # Decode without verification to inspect structure
        try:
            header = jwt.get_unverified_header(access_token)
            payload = jwt.decode(access_token, options={"verify_signature": False})
            
            has_required_claims = all(
                claim in payload for claim in ["sub", "exp", "iat"]
            )
            uses_secure_algorithm = header.get("alg") in ["HS256", "RS256", "ES256"]
            has_expiration = "exp" in payload and payload["exp"] > time.time()
            
            execution_time = time.time() - start_time
            
            results.append(self._create_test_result(
                test_name="jwt_structure_validation",
                test_type=SecurityTestType.TOKEN_SECURITY,
                severity=SecurityTestSeverity.HIGH,
                passed=has_required_claims and uses_secure_algorithm and has_expiration,
                description="Test JWT token structure and security claims",
                details={
                    "algorithm": header.get("alg"),
                    "has_required_claims": has_required_claims,
                    "uses_secure_algorithm": uses_secure_algorithm,
                    "has_expiration": has_expiration,
                    "payload_keys": list(payload.keys())
                },
                recommendations=[
                    "Use secure JWT algorithms (HS256, RS256, ES256)",
                    "Include required claims: sub, exp, iat",
                    "Set appropriate expiration times"
                ] if not (has_required_claims and uses_secure_algorithm and has_expiration) else [],
                execution_time=execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            results.append(self._create_test_result(
                test_name="jwt_structure_validation",
                test_type=SecurityTestType.TOKEN_SECURITY,
                severity=SecurityTestSeverity.CRITICAL,
                passed=False,
                description="Failed to decode JWT token",
                details={"error": str(e)},
                recommendations=["Fix JWT token generation"],
                execution_time=execution_time
            ))
        
        # Test 2: JWT signature verification
        start_time = time.time()
        
        # Test valid token verification
        valid_verification = self.auth_service.verify_token(access_token) is not None
        
        # Test invalid signature detection
        tampered_token = access_token[:-10] + "tampered123"
        try:
            invalid_verification = self.auth_service.verify_token(tampered_token)
            signature_security = invalid_verification is None
        except:
            signature_security = True  # Exception is expected
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="jwt_signature_verification",
            test_type=SecurityTestType.TOKEN_SECURITY,
            severity=SecurityTestSeverity.CRITICAL,
            passed=valid_verification and signature_security,
            description="Test JWT signature verification security",
            details={
                "valid_token_verified": valid_verification,
                "tampered_token_rejected": signature_security
            },
            recommendations=[
                "Ensure signature verification is enforced",
                "Reject tokens with invalid signatures"
            ] if not (valid_verification and signature_security) else [],
            execution_time=execution_time
        ))
        
        # Test 3: Token expiration handling
        start_time = time.time()
        
        # Create expired token (mock by setting past expiration)
        expired_payload = {
            "sub": "expired-user",
            "exp": int(time.time()) - 3600,  # Expired 1 hour ago
            "iat": int(time.time()) - 7200   # Issued 2 hours ago
        }
        
        expired_token = jwt.encode(
            expired_payload, 
            self.auth_service.secret_key, 
            algorithm=self.auth_service.algorithm
        )
        
        try:
            expired_result = self.auth_service.verify_token(expired_token)
            expiration_handled = expired_result is None
        except jwt.ExpiredSignatureError:
            expiration_handled = True
        except Exception:
            expiration_handled = False
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="jwt_expiration_handling",
            test_type=SecurityTestType.TOKEN_SECURITY,
            severity=SecurityTestSeverity.HIGH,
            passed=expiration_handled,
            description="Test JWT token expiration is properly enforced",
            details={
                "expired_token_rejected": expiration_handled
            },
            recommendations=[
                "Properly handle token expiration",
                "Reject expired tokens"
            ] if not expiration_handled else [],
            execution_time=execution_time
        ))
        
        return results
    
    async def test_authentication_security(self) -> List[SecurityTestResult]:
        """Test authentication security mechanisms"""
        results = []
        
        # Test 1: Account lockout mechanism
        start_time = time.time()
        
        user_id = "lockout-test-user"
        client_ip = "192.168.1.200"
        
        # Simulate failed login attempts
        for i in range(6):  # One more than the threshold
            await self.auth_service.record_failed_attempt(user_id, client_ip)
        
        # Check if account is locked
        is_locked = await self.auth_service.check_account_lockout(user_id, client_ip)
        
        # Check lockout data in Redis
        lock_key = f"account_lock:{user_id}"
        lockout_data = await self.redis_client.get(lock_key)
        has_lockout_data = lockout_data is not None
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="account_lockout_mechanism",
            test_type=SecurityTestType.AUTHENTICATION,
            severity=SecurityTestSeverity.HIGH,
            passed=is_locked and has_lockout_data,
            description="Test account lockout after failed login attempts",
            details={
                "account_locked": is_locked,
                "lockout_data_stored": has_lockout_data,
                "failed_attempts_threshold": 5
            },
            recommendations=[
                "Implement account lockout after multiple failed attempts",
                "Store lockout information securely",
                "Set appropriate lockout duration"
            ] if not (is_locked and has_lockout_data) else [],
            execution_time=execution_time
        ))
        
        # Test 2: Brute force attack resistance
        start_time = time.time()
        
        # Setup test user
        test_user = User(
            id="brute-force-test",
            username="brutetest",
            email="brute@test.com",
            password_hash=await self.auth_service.hash_password("SecurePassword123!"),
            roles=["user"],
            is_active=True
        )
        
        self.mock_user_repo.get_by_username.return_value = test_user
        
        # Attempt rapid authentication attempts
        rapid_attempts = []
        for i in range(10):
            attempt_start = time.perf_counter()
            try:
                credentials = {
                    "username": "brutetest",
                    "password": f"wrong_password_{i}",
                    "client_ip": "192.168.1.201"
                }
                result = await self.auth_service.authenticate_user(credentials)
                attempt_end = time.perf_counter()
                rapid_attempts.append(attempt_end - attempt_start)
            except Exception:
                attempt_end = time.perf_counter()
                rapid_attempts.append(attempt_end - attempt_start)
        
        # Check if later attempts take longer (indicating rate limiting)
        early_avg = sum(rapid_attempts[:3]) / 3
        late_avg = sum(rapid_attempts[-3:]) / 3
        shows_rate_limiting = late_avg > early_avg * 1.5  # 50% slower
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="brute_force_resistance",
            test_type=SecurityTestType.BRUTE_FORCE,
            severity=SecurityTestSeverity.HIGH,
            passed=shows_rate_limiting,
            description="Test resistance to brute force attacks",
            details={
                "early_avg_time": early_avg,
                "late_avg_time": late_avg,
                "shows_rate_limiting": shows_rate_limiting,
                "all_attempt_times": rapid_attempts
            },
            recommendations=[
                "Implement progressive delays for failed attempts",
                "Use rate limiting for authentication endpoints",
                "Consider CAPTCHA after multiple failures"
            ] if not shows_rate_limiting else [],
            execution_time=execution_time
        ))
        
        return results
    
    async def test_session_management(self) -> List[SecurityTestResult]:
        """Test session management security"""
        results = []
        
        # Test 1: Token blacklisting
        start_time = time.time()
        
        test_user = User(id="session-test", username="sessiontest", email="session@test.com")
        access_token, _ = self.auth_service.create_access_token(test_user)
        
        # Verify token works initially
        initial_validation = self.auth_service.verify_token(access_token) is not None
        
        # Revoke token
        revocation_success = await self.auth_service.revoke_token(access_token)
        
        # Verify token is blacklisted
        post_revocation_validation = await self.auth_service.validate_token(access_token) is None
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="token_blacklisting",
            test_type=SecurityTestType.SESSION_MANAGEMENT,
            severity=SecurityTestSeverity.HIGH,
            passed=initial_validation and revocation_success and post_revocation_validation,
            description="Test token revocation and blacklisting",
            details={
                "initial_validation": initial_validation,
                "revocation_success": revocation_success,
                "post_revocation_blocked": post_revocation_validation
            },
            recommendations=[
                "Implement token blacklisting for revoked tokens",
                "Ensure revoked tokens cannot be used"
            ] if not (revocation_success and post_revocation_validation) else [],
            execution_time=execution_time
        ))
        
        # Test 2: API key security
        start_time = time.time()
        
        user_id = "api-key-test"
        key_name = "test-api-key"
        scopes = ["read", "write"]
        
        # Create API key
        raw_key, key_hash = await self.auth_service.create_api_key(user_id, key_name, scopes)
        
        # Test key format
        proper_format = raw_key.startswith("xorb_") and len(raw_key) > 40
        
        # Test key validation
        key_data = await self.auth_service.validate_api_key(raw_key)
        valid_data = key_data is not None and key_data["user_id"] == user_id
        
        # Test invalid key rejection
        invalid_key_data = await self.auth_service.validate_api_key("invalid_key")
        invalid_rejected = invalid_key_data is None
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="api_key_security",
            test_type=SecurityTestType.SESSION_MANAGEMENT,
            severity=SecurityTestSeverity.MEDIUM,
            passed=proper_format and valid_data and invalid_rejected,
            description="Test API key generation and validation security",
            details={
                "proper_key_format": proper_format,
                "valid_key_accepted": valid_data,
                "invalid_key_rejected": invalid_rejected,
                "key_length": len(raw_key)
            },
            recommendations=[
                "Use secure API key format with sufficient entropy",
                "Validate API keys properly",
                "Reject invalid API keys"
            ] if not (proper_format and valid_data and invalid_rejected) else [],
            execution_time=execution_time
        ))
        
        return results
    
    async def test_input_validation(self) -> List[SecurityTestResult]:
        """Test input validation security"""
        results = []
        
        # Test 1: SQL injection attempts in authentication
        start_time = time.time()
        
        sql_injection_payloads = [
            "admin'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'/*",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --"
        ]
        
        injection_vulnerabilities = []
        
        for payload in sql_injection_payloads:
            try:
                credentials = {
                    "username": payload,
                    "password": "testpassword",
                    "client_ip": "192.168.1.250"
                }
                
                # This should not cause any SQL injection since we use proper ORM
                result = await self.auth_service.authenticate_user(credentials)
                
                # Check if any unexpected behavior occurred
                if hasattr(result, 'user') and result.user:
                    injection_vulnerabilities.append(payload)
                    
            except Exception as e:
                # Exceptions are expected for malicious input
                pass
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="sql_injection_protection",
            test_type=SecurityTestType.INPUT_VALIDATION,
            severity=SecurityTestSeverity.CRITICAL,
            passed=len(injection_vulnerabilities) == 0,
            description="Test protection against SQL injection attacks",
            details={
                "payloads_tested": len(sql_injection_payloads),
                "vulnerabilities_found": injection_vulnerabilities
            },
            recommendations=[
                "Use parameterized queries or ORM",
                "Validate and sanitize all user input",
                "Implement input length limits"
            ] if injection_vulnerabilities else [],
            execution_time=execution_time
        ))
        
        # Test 2: Cross-site scripting (XSS) protection
        start_time = time.time()
        
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "'><script>alert('xss')</script>",
            "<svg onload=alert('xss')>"
        ]
        
        xss_vulnerabilities = []
        
        for payload in xss_payloads:
            try:
                # Test if payload gets stored and returned unescaped
                user_id = "xss-test"
                key_name = payload  # Try to inject XSS in API key name
                
                raw_key, _ = await self.auth_service.create_api_key(user_id, key_name, ["read"])
                key_data = await self.auth_service.validate_api_key(raw_key)
                
                if key_data and payload in str(key_data):
                    xss_vulnerabilities.append(payload)
                    
            except Exception:
                # Exceptions for malicious input are good
                pass
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="xss_protection",
            test_type=SecurityTestType.INPUT_VALIDATION,
            severity=SecurityTestSeverity.HIGH,
            passed=len(xss_vulnerabilities) == 0,
            description="Test protection against XSS attacks",
            details={
                "payloads_tested": len(xss_payloads),
                "vulnerabilities_found": xss_vulnerabilities
            },
            recommendations=[
                "Escape output data",
                "Validate input for malicious scripts",
                "Use Content Security Policy headers"
            ] if xss_vulnerabilities else [],
            execution_time=execution_time
        ))
        
        return results
    
    async def test_encryption_security(self) -> List[SecurityTestResult]:
        """Test encryption and cryptographic security"""
        results = []
        
        # Test 1: Random number generation quality
        start_time = time.time()
        
        # Generate multiple random values and test for patterns
        random_values = []
        for _ in range(1000):
            # Test the secrets module used in token generation
            random_val = secrets.token_hex(32)
            random_values.append(random_val)
        
        # Check for duplicates (should be extremely rare)
        unique_values = set(random_values)
        no_duplicates = len(unique_values) == len(random_values)
        
        # Check entropy (basic test)
        concatenated = ''.join(random_values)
        char_distribution = {}
        for char in concatenated:
            char_distribution[char] = char_distribution.get(char, 0) + 1
        
        # Calculate basic entropy metric
        total_chars = len(concatenated)
        entropy_score = 0
        for count in char_distribution.values():
            frequency = count / total_chars
            if frequency > 0:
                entropy_score -= frequency * (frequency.bit_length() - 1)
        
        good_entropy = entropy_score > 3.5  # Threshold for good entropy
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="random_number_generation",
            test_type=SecurityTestType.ENCRYPTION,
            severity=SecurityTestSeverity.HIGH,
            passed=no_duplicates and good_entropy,
            description="Test quality of random number generation",
            details={
                "samples_generated": len(random_values),
                "unique_values": len(unique_values),
                "no_duplicates": no_duplicates,
                "entropy_score": entropy_score,
                "good_entropy": good_entropy
            },
            recommendations=[
                "Use cryptographically secure random number generators",
                "Ensure sufficient entropy in random values"
            ] if not (no_duplicates and good_entropy) else [],
            execution_time=execution_time
        ))
        
        # Test 2: Key derivation security
        start_time = time.time()
        
        # Test that API key hashing is secure
        test_key = "test_api_key_12345"
        
        # Hash the same key multiple times
        hash1 = hashlib.sha256(test_key.encode()).hexdigest()
        hash2 = hashlib.sha256(test_key.encode()).hexdigest()
        
        # Hashes should be identical for same input
        consistent_hashing = hash1 == hash2
        
        # Hash should be different from input
        different_from_input = hash1 != test_key
        
        # Hash should have proper length (SHA256 = 64 hex chars)
        proper_length = len(hash1) == 64
        
        execution_time = time.time() - start_time
        
        results.append(self._create_test_result(
            test_name="key_derivation_security",
            test_type=SecurityTestType.ENCRYPTION,
            severity=SecurityTestSeverity.MEDIUM,
            passed=consistent_hashing and different_from_input and proper_length,
            description="Test key derivation and hashing security",
            details={
                "consistent_hashing": consistent_hashing,
                "different_from_input": different_from_input,
                "proper_hash_length": proper_length,
                "hash_length": len(hash1)
            },
            recommendations=[
                "Use secure hashing algorithms",
                "Ensure consistent key derivation"
            ] if not (consistent_hashing and proper_length) else [],
            execution_time=execution_time
        ))
        
        return results
    
    async def run_comprehensive_security_tests(self) -> Dict[str, Any]:
        """Run comprehensive security test suite"""
        print("🔒 Starting comprehensive security test suite...")
        
        await self.setup()
        
        try:
            all_results = []
            
            # Run all security test categories
            password_results = await self.test_password_security()
            jwt_results = await self.test_jwt_security()
            auth_results = await self.test_authentication_security()
            session_results = await self.test_session_management()
            input_results = await self.test_input_validation()
            encryption_results = await self.test_encryption_security()
            
            all_results.extend(password_results)
            all_results.extend(jwt_results)
            all_results.extend(auth_results)
            all_results.extend(session_results)
            all_results.extend(input_results)
            all_results.extend(encryption_results)
            
            self.test_results = all_results
            
            # Generate summary report
            summary = self._generate_security_summary()
            
            return {
                "test_results": all_results,
                "summary": summary,
                "vulnerabilities": self.vulnerabilities
            }
            
        finally:
            await self.teardown()
    
    def _generate_security_summary(self) -> Dict[str, Any]:
        """Generate security test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        # Group by severity
        by_severity = {}
        for result in self.test_results:
            severity = result.severity.value
            if severity not in by_severity:
                by_severity[severity] = {"total": 0, "passed": 0, "failed": 0}
            
            by_severity[severity]["total"] += 1
            if result.passed:
                by_severity[severity]["passed"] += 1
            else:
                by_severity[severity]["failed"] += 1
        
        # Group by test type
        by_type = {}
        for result in self.test_results:
            test_type = result.test_type.value
            if test_type not in by_type:
                by_type[test_type] = {"total": 0, "passed": 0, "failed": 0}
            
            by_type[test_type]["total"] += 1
            if result.passed:
                by_type[test_type]["passed"] += 1
            else:
                by_type[test_type]["failed"] += 1
        
        # Calculate overall security score
        critical_failed = by_severity.get("critical", {}).get("failed", 0)
        high_failed = by_severity.get("high", {}).get("failed", 0)
        
        if critical_failed > 0:
            security_score = 0  # Critical failures = 0 score
        elif high_failed > 0:
            security_score = max(0, 100 - (high_failed * 20))  # -20 for each high failure
        else:
            security_score = max(0, 100 - (failed_tests * 5))  # -5 for each failure
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "security_score": security_score,
            "by_severity": by_severity,
            "by_type": by_type,
            "critical_failures": critical_failed,
            "high_failures": high_failed
        }
    
    def print_security_report(self, results: Dict[str, Any]):
        """Print formatted security test report"""
        print("\n" + "="*80)
        print("🔒 COMPREHENSIVE SECURITY TEST REPORT")
        print("="*80)
        
        summary = results["summary"]
        
        # Overall summary
        print(f"\n📊 OVERALL SECURITY SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ({summary['pass_rate']:.1f}%)")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Security Score: {summary['security_score']:.0f}/100")
        
        if summary['critical_failures'] > 0:
            print(f"   ⚠️  CRITICAL FAILURES: {summary['critical_failures']}")
        
        # By severity
        print(f"\n🔴 BY SEVERITY:")
        for severity, data in summary["by_severity"].items():
            status = "✅" if data["failed"] == 0 else "❌"
            print(f"   {status} {severity.upper()}: {data['passed']}/{data['total']} passed")
        
        # By test type
        print(f"\n🔍 BY TEST TYPE:")
        for test_type, data in summary["by_type"].items():
            status = "✅" if data["failed"] == 0 else "❌"
            print(f"   {status} {test_type.replace('_', ' ').title()}: {data['passed']}/{data['total']} passed")
        
        # Failed tests details
        failed_tests = [r for r in results["test_results"] if not r.passed]
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test.test_name} ({test.severity.value.upper()})")
                print(f"     {test.description}")
                if test.recommendations:
                    print(f"     Recommendations: {'; '.join(test.recommendations)}")
        
        # Security recommendations
        if failed_tests:
            print(f"\n🛠️  SECURITY RECOMMENDATIONS:")
            all_recommendations = set()
            for test in failed_tests:
                all_recommendations.update(test.recommendations)
            
            for i, rec in enumerate(sorted(all_recommendations), 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)


async def run_security_tests():
    """Main function to run security tests"""
    framework = SecurityTestFramework()
    results = await framework.run_comprehensive_security_tests()
    framework.print_security_report(results)
    return results


if __name__ == "__main__":
    # Run security tests
    asyncio.run(run_security_tests())