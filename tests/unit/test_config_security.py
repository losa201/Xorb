"""
Security tests for configuration management
"""

import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError

from src.api.app.core.config import AppSettings, get_config_manager


class TestJWTSecretSecurity:
    """Test JWT secret security enforcement"""
    
    def test_jwt_secret_required(self):
        """Test that JWT secret is required"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()
            
            error = str(exc_info.value)
            assert "JWT_SECRET" in error or "jwt_secret_key" in error
    
    def test_jwt_secret_minimum_length(self):
        """Test minimum length requirement for JWT secret"""
        from tests.fixtures.secure_credentials import get_secure_jwt_secret
        short_secret = "short"
        
        with patch.dict(os.environ, {"JWT_SECRET": short_secret}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()
            
            error = str(exc_info.value)
            assert "32 characters" in error
    
    def test_jwt_secret_rejects_common_defaults(self):
        """Test that common insecure defaults are rejected"""
        insecure_defaults = [
            "change-me-in-production",
            "dev-jwt-secret-key-change-in-production-12345678901234567890",
            "default",
            "secret",
            "key",
            "jwt-secret",
            "your-secret-key",
            "12345678901234567890123456789012"
        ]
        
        for insecure_secret in insecure_defaults:
            with patch.dict(os.environ, {"JWT_SECRET": insecure_secret}, clear=True):
                with pytest.raises(ValidationError) as exc_info:
                    AppSettings()
                
                error = str(exc_info.value)
                assert "common default value" in error
    
    def test_jwt_secret_production_requirements(self):
        """Test production-specific JWT secret requirements"""
        # Short secret in production should fail
        short_secret = "a" * 32  # 32 chars but less than 64 - using generated pattern
        
        with patch.dict(os.environ, {
            "JWT_SECRET": short_secret,
            "ENVIRONMENT": "production"
        }, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()
            
            error = str(exc_info.value)
            assert "64 characters in production" in error
    
    def test_jwt_secret_production_entropy(self):
        """Test entropy requirement in production"""
        # Secret with insufficient entropy
        # Generate low entropy secret for testing
        low_entropy_secret = "a" * 64  # All same character - test pattern only
        
        with patch.dict(os.environ, {
            "JWT_SECRET": low_entropy_secret,
            "ENVIRONMENT": "production"
        }, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()
            
            error = str(exc_info.value)
            assert "16 unique characters" in error
    
    def test_jwt_secret_valid_development(self):
        """Test that valid secret works in development"""
        # Generate secure test secret
        from tests.fixtures.secure_credentials import get_secure_jwt_secret
        valid_secret = get_secure_jwt_secret()[:32]  # Truncate for test
        
        with patch.dict(os.environ, {
            "JWT_SECRET": valid_secret,
            "ENVIRONMENT": "development"
        }, clear=True):
            settings = AppSettings()
            assert settings.jwt_secret_key == valid_secret
    
    def test_jwt_secret_valid_production(self):
        """Test that valid secret works in production"""
        # Generate secure test secret
        valid_secret = get_secure_jwt_secret()  # Full length secure secret
        
        with patch.dict(os.environ, {
            "JWT_SECRET": valid_secret,
            "ENVIRONMENT": "production"
        }, clear=True):
            settings = AppSettings()
            assert settings.jwt_secret_key == valid_secret


class TestConfigurationSecurity:
    """Test overall configuration security"""
    
    def test_production_validation(self):
        """Test production environment validation"""
        # Generate secure JWT secret for testing
        valid_jwt_secret = get_secure_jwt_secret()
        
        with patch.dict(os.environ, {
            "JWT_SECRET": valid_jwt_secret,
            "ENVIRONMENT": "production",
            "DEBUG": "true",  # Should trigger validation warning
            "CORS_ALLOW_ORIGINS": "*"  # Should trigger validation warning
        }, clear=True):
            config_manager = get_config_manager()
            issues = config_manager.validate_configuration()
            
            # Should identify security issues
            assert len(issues) > 0
            assert any("debug" in issue.lower() for issue in issues)
            assert any("cors" in issue.lower() for issue in issues)
    
    def test_development_allows_relaxed_settings(self):
        """Test that development environment allows relaxed settings"""
        # Generate valid JWT secret
        valid_jwt_secret = get_secure_jwt_secret()
        
        with patch.dict(os.environ, {
            "JWT_SECRET": valid_jwt_secret,
            "ENVIRONMENT": "development",
            "DEBUG": "true",
            "CORS_ALLOW_ORIGINS": "*"
        }, clear=True):
            settings = AppSettings()
            assert settings.debug is True
            assert settings.cors_allow_origins == "*"
    
    def test_configuration_summary_masks_secrets(self):
        """Test that configuration summary masks sensitive data"""
        # Generate valid JWT secret
        valid_jwt_secret = get_secure_jwt_secret()
        
        with patch.dict(os.environ, {
            "JWT_SECRET": valid_jwt_secret,
            "DATABASE_URL": "postgresql://user:password@localhost/db"
        }, clear=True):
            config_manager = get_config_manager()
            summary = config_manager.get_configuration_summary()
            
            # JWT secret should not appear in summary
            assert valid_jwt_secret not in str(summary)
            
            # Database password should be masked
            assert "password" not in summary["database"]["url_masked"]


class TestSecurityConfigIntegration:
    """Test integration with security configuration"""
    
    def test_security_config_uses_jwt_secret(self):
        """Test that security config properly uses JWT secret"""
        valid_jwt_secret = "abcdefghijklmnopqrstuvwxyz123456"
        
        with patch.dict(os.environ, {
            "JWT_SECRET": valid_jwt_secret
        }, clear=True):
            config_manager = get_config_manager()
            security_config = config_manager.security_config
            
            assert security_config.jwt_secret_key == valid_jwt_secret
    
    def test_security_config_validation_blocks_startup(self):
        """Test that invalid configuration blocks application startup"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):  # Should fail to create settings
                get_config_manager().security_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])