"""
Unit tests for security core functionality
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.api.app.core.security import (
    SecurityService, SecurityConfig, PasswordValidator,
    CryptographyService, JWTService, RateLimitService
)


class TestPasswordValidator:
    """Test password validation functionality"""
    
    def test_password_validation_success(self, test_security_config):
        validator = PasswordValidator(test_security_config)
        
        is_valid, errors = validator.validate_password("SecurePassword123!", "testuser")
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_password_too_short(self, test_security_config):
        validator = PasswordValidator(test_security_config)
        
        is_valid, errors = validator.validate_password("Short1!", "testuser")
        
        assert is_valid is False
        assert any("at least" in error for error in errors)
    
    def test_password_missing_uppercase(self, test_security_config):
        validator = PasswordValidator(test_security_config)
        
        is_valid, errors = validator.validate_password("securepassword123!", "testuser")
        
        assert is_valid is False
        assert any("uppercase" in error for error in errors)
    
    def test_password_missing_special_chars(self, test_security_config):
        validator = PasswordValidator(test_security_config)
        
        is_valid, errors = validator.validate_password("SecurePassword123", "testuser")
        
        assert is_valid is False
        assert any("special character" in error for error in errors)
    
    def test_password_contains_username(self, test_security_config):
        validator = PasswordValidator(test_security_config)
        
        is_valid, errors = validator.validate_password("testuserPassword123!", "testuser")
        
        assert is_valid is False
        assert any("username" in error for error in errors)
    
    def test_password_strength_calculation(self, test_security_config):
        validator = PasswordValidator(test_security_config)
        
        # Strong password
        score, strength = validator.calculate_strength_score("VerySecureP@ssw0rd123!")
        assert score >= 80
        assert strength == "STRONG"
        
        # Weak password
        score, strength = validator.calculate_strength_score("password")
        assert score < 40
        assert strength == "VERY_WEAK"


class TestCryptographyService:
    """Test cryptography functionality"""
    
    def test_password_hashing_and_verification(self, test_security_config):
        crypto_service = CryptographyService(test_security_config)
        
        password = "TestPassword123!"
        hashed = crypto_service.hash_password(password)
        
        assert hashed != password
        assert crypto_service.verify_password(password, hashed) is True
        assert crypto_service.verify_password("wrong_password", hashed) is False
    
    def test_secure_token_generation(self, test_security_config):
        crypto_service = CryptographyService(test_security_config)
        
        token1 = crypto_service.generate_secure_token(32)
        token2 = crypto_service.generate_secure_token(32)
        
        assert len(token1) > 0
        assert len(token2) > 0
        assert token1 != token2
    
    def test_data_encryption_decryption(self, test_security_config):
        crypto_service = CryptographyService(test_security_config)
        
        original_data = "sensitive information"
        encrypted = crypto_service.encrypt_sensitive_data(original_data)
        decrypted = crypto_service.decrypt_sensitive_data(encrypted)
        
        assert encrypted != original_data
        assert decrypted == original_data
    
    def test_webhook_signature_creation_and_verification(self, test_security_config):
        crypto_service = CryptographyService(test_security_config)
        
        payload = '{"event": "test", "data": "sample"}'
        secret = "webhook_secret"
        
        signature = crypto_service.create_webhook_signature(payload, secret)
        
        assert crypto_service.verify_webhook_signature(payload, signature, secret) is True
        assert crypto_service.verify_webhook_signature(payload, signature, "wrong_secret") is False
        assert crypto_service.verify_webhook_signature("wrong_payload", signature, secret) is False


class TestJWTService:
    """Test JWT token functionality"""
    
    def test_access_token_creation_and_verification(self, test_security_config):
        jwt_service = JWTService(test_security_config)
        
        user_id = "test_user_123"
        additional_claims = {"username": "testuser", "roles": ["user"]}
        
        token = jwt_service.create_access_token(user_id, additional_claims)
        payload = jwt_service.verify_token(token, "access")
        
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["username"] == "testuser"
        assert payload["type"] == "access"
    
    def test_refresh_token_creation_and_verification(self, test_security_config):
        jwt_service = JWTService(test_security_config)
        
        user_id = "test_user_123"
        
        refresh_token = jwt_service.create_refresh_token(user_id)
        payload = jwt_service.verify_token(refresh_token, "refresh")
        
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
    
    def test_token_refresh_flow(self, test_security_config):
        jwt_service = JWTService(test_security_config)
        
        user_id = "test_user_123"
        refresh_token = jwt_service.create_refresh_token(user_id)
        
        new_access_token = jwt_service.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        
        new_payload = jwt_service.verify_token(new_access_token, "access")
        assert new_payload["sub"] == user_id
    
    def test_invalid_token_handling(self, test_security_config):
        jwt_service = JWTService(test_security_config)
        
        # Invalid token
        assert jwt_service.verify_token("invalid_token") is None
        
        # Wrong token type
        refresh_token = jwt_service.create_refresh_token("user_123")
        assert jwt_service.verify_token(refresh_token, "access") is None


class TestRateLimitService:
    """Test rate limiting functionality"""
    
    def test_rate_limit_enforcement(self, test_security_config):
        rate_limit_service = RateLimitService(test_security_config)
        
        key = "test_user"
        max_attempts = 3
        
        # First attempts should not be rate limited
        for i in range(max_attempts):
            assert rate_limit_service.is_rate_limited(key, max_attempts) is False
            rate_limit_service.record_attempt(key, success=False)
        
        # Should be rate limited after max attempts
        assert rate_limit_service.is_rate_limited(key, max_attempts) is True
    
    def test_rate_limit_reset_on_success(self, test_security_config):
        rate_limit_service = RateLimitService(test_security_config)
        
        key = "test_user"
        max_attempts = 3
        
        # Record failed attempts
        for i in range(max_attempts - 1):
            rate_limit_service.record_attempt(key, success=False)
        
        # Successful attempt should reset
        rate_limit_service.record_attempt(key, success=True)
        
        assert rate_limit_service.is_rate_limited(key, max_attempts) is False
    
    def test_remaining_attempts_calculation(self, test_security_config):
        rate_limit_service = RateLimitService(test_security_config)
        
        key = "test_user"
        max_attempts = 5
        
        # Initially should have max attempts
        assert rate_limit_service.get_remaining_attempts(key, max_attempts) == max_attempts
        
        # After one failed attempt
        rate_limit_service.record_attempt(key, success=False)
        assert rate_limit_service.get_remaining_attempts(key, max_attempts) == max_attempts - 1
        
        # After reaching limit
        for i in range(max_attempts - 1):
            rate_limit_service.record_attempt(key, success=False)
        
        assert rate_limit_service.get_remaining_attempts(key, max_attempts) == 0


class TestSecurityService:
    """Test integrated security service functionality"""
    
    def test_password_validation_and_hashing(self, test_security_service):
        password = "SecureTestPassword123!"
        username = "testuser"
        
        is_valid, password_hash, errors = test_security_service.validate_and_hash_password(password, username)
        
        assert is_valid is True
        assert password_hash != password
        assert len(errors) == 0
    
    def test_user_authentication_with_rate_limiting(self, test_security_service):
        username = "testuser"
        password = "SecureTestPassword123!"
        
        # Create password hash
        _, password_hash, _ = test_security_service.validate_and_hash_password(password, username)
        
        # Successful authentication
        assert test_security_service.authenticate_user(username, password, password_hash) is True
        
        # Failed authentication
        assert test_security_service.authenticate_user(username, "wrong_password", password_hash) is False
    
    def test_user_token_creation_and_verification(self, test_security_service):
        user_id = "test_user_123"
        additional_claims = {"username": "testuser", "roles": ["user"]}
        
        tokens = test_security_service.create_user_tokens(user_id, additional_claims)
        
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"
        
        # Verify access token
        payload = test_security_service.verify_access_token(tokens["access_token"])
        assert payload is not None
        assert payload["sub"] == user_id
    
    def test_token_refresh(self, test_security_service):
        user_id = "test_user_123"
        tokens = test_security_service.create_user_tokens(user_id)
        
        new_access_token = test_security_service.refresh_tokens(tokens["refresh_token"])
        
        assert new_access_token is not None
        
        # Verify new token
        payload = test_security_service.verify_access_token(new_access_token)
        assert payload["sub"] == user_id


@pytest.mark.asyncio
async def test_security_service_integration():
    """Integration test for security service components"""
    config = SecurityConfig(
        jwt_secret_key="test-integration-secret",
        min_password_length=10,
        max_login_attempts=3
    )
    
    security_service = SecurityService(config)
    
    # Test complete authentication flow
    username = "integration_user"
    password = "IntegrationTestPassword123!"
    
    # Validate and hash password
    is_valid, password_hash, errors = security_service.validate_and_hash_password(password, username)
    assert is_valid is True
    
    # Authenticate user
    auth_result = security_service.authenticate_user(username, password, password_hash)
    assert auth_result is True
    
    # Create tokens
    tokens = security_service.create_user_tokens(username)
    assert tokens["access_token"] is not None
    
    # Verify token
    payload = security_service.verify_access_token(tokens["access_token"])
    assert payload["sub"] == username