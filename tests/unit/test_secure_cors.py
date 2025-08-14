"""
Tests for secure CORS configuration
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import Request

from src.api.app.middleware.secure_cors import (
    SecureCORSConfig, 
    create_secure_cors_middleware,
    validate_cors_configuration
)


class TestSecureCORSConfig:
    """Test cases for SecureCORSConfig"""
    
    def test_production_rejects_wildcard(self):
        """Test that production environment rejects wildcard origins"""
        config = SecureCORSConfig("production")
        
        # Wildcard should be rejected in production
        assert not config._validate_origin("*")
        
        # Valid HTTPS origin should be accepted
        assert config._validate_origin("https://app.xorb.enterprise")
    
    def test_development_allows_localhost(self):
        """Test that development environment allows localhost"""
        config = SecureCORSConfig("development")
        
        # Localhost should be allowed in development
        assert config._validate_origin("http://localhost:3000")
        assert config._validate_origin("http://127.0.0.1:8080")
        
        # Wildcard should be allowed in development
        assert config._validate_origin("*")
    
    def test_production_requires_https(self):
        """Test that production requires HTTPS for external origins"""
        config = SecureCORSConfig("production")
        
        # HTTP should be rejected in production (except localhost)
        assert not config._validate_origin("http://example.com")
        
        # HTTPS should be accepted
        assert config._validate_origin("https://app.xorb.enterprise")
    
    def test_hostname_validation(self):
        """Test hostname validation logic"""
        config = SecureCORSConfig("production")
        
        # Valid hostnames
        assert config._validate_hostname("app.xorb.enterprise")
        assert config._validate_hostname("localhost")
        assert config._validate_hostname("127.0.0.1")
        
        # Invalid hostnames
        assert not config._validate_hostname("")
        assert not config._validate_hostname("invalid..domain")
        assert not config._validate_hostname("domain with spaces")
    
    def test_domain_whitelist_production(self):
        """Test domain whitelist enforcement in production"""
        config = SecureCORSConfig("production")
        
        # Whitelisted domains should pass
        assert config._check_domain_whitelist("app.xorb.enterprise")
        assert config._check_domain_whitelist("api.xorb.enterprise")
        
        # Non-whitelisted domains should fail
        assert not config._check_domain_whitelist("malicious.com")
        assert not config._check_domain_whitelist("fake.xorb.com")
    
    def test_cors_violation_logging(self):
        """Test CORS violation logging"""
        config = SecureCORSConfig("production")
        
        # Mock request
        request = Mock(spec=Request)
        request.headers = {"origin": "http://malicious.com", "user-agent": "test"}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/test"
        
        # Should log violation without raising exception
        config.log_cors_violation(request, "Origin not allowed")
    
    def test_validate_and_set_origins_with_valid_list(self):
        """Test validation and setting of valid origins"""
        config = SecureCORSConfig("development")
        
        origins_string = "http://localhost:3000,https://app.xorb.enterprise"
        validated = config.validate_and_set_origins(origins_string)
        
        assert "http://localhost:3000" in validated
        assert "https://app.xorb.enterprise" in validated
        assert len(validated) == 2
    
    def test_validate_and_set_origins_filters_invalid(self):
        """Test that invalid origins are filtered out"""
        config = SecureCORSConfig("production")
        
        origins_string = "https://app.xorb.enterprise,http://malicious.com,*"
        validated = config.validate_and_set_origins(origins_string)
        
        # Only valid origin should remain
        assert "https://app.xorb.enterprise" in validated
        assert "http://malicious.com" not in validated
        assert "*" not in validated
    
    def test_empty_origins_uses_defaults(self):
        """Test that empty origins string uses secure defaults"""
        config = SecureCORSConfig("production")
        
        validated = config.validate_and_set_origins("")
        
        # Should fall back to secure defaults
        assert "https://app.xorb.enterprise" in validated


class TestCreateSecureCORSMiddleware:
    """Test cases for create_secure_cors_middleware function"""
    
    def test_production_configuration(self):
        """Test production CORS middleware configuration"""
        cors_config, middleware_config = create_secure_cors_middleware(
            "production", 
            "https://app.xorb.enterprise"
        )
        
        assert cors_config.environment == "production"
        assert "https://app.xorb.enterprise" in middleware_config["allow_origins"]
        assert "*" not in middleware_config["allow_origins"]
        assert middleware_config["allow_credentials"] is True
    
    def test_development_configuration(self):
        """Test development CORS middleware configuration"""
        cors_config, middleware_config = create_secure_cors_middleware(
            "development",
            "http://localhost:3000,*"
        )
        
        assert cors_config.environment == "development"
        assert "http://localhost:3000" in middleware_config["allow_origins"]
        assert "*" in middleware_config["allow_origins"]
    
    def test_invalid_origins_fallback(self):
        """Test fallback to defaults when all origins are invalid"""
        cors_config, middleware_config = create_secure_cors_middleware(
            "production",
            "http://malicious.com,*,invalid-url"
        )
        
        # Should fall back to secure defaults
        origins = middleware_config["allow_origins"]
        assert "https://app.xorb.enterprise" in origins
        assert "http://malicious.com" not in origins


class TestValidateCORSConfiguration:
    """Test cases for validate_cors_configuration function"""
    
    def test_production_validation_rejects_wildcard(self):
        """Test that production validation rejects wildcard"""
        origins = ["https://app.xorb.enterprise", "*"]
        
        # Should return False due to wildcard in production
        assert not validate_cors_configuration("production", origins)
    
    def test_production_validation_rejects_http(self):
        """Test that production validation rejects HTTP"""
        origins = ["https://app.xorb.enterprise", "http://example.com"]
        
        # Should return False due to HTTP in production
        assert not validate_cors_configuration("production", origins)
    
    def test_production_validation_passes_valid_config(self):
        """Test that production validation passes valid configuration"""
        origins = ["https://app.xorb.enterprise", "https://api.xorb.enterprise"]
        
        # Should return True for valid HTTPS origins
        assert validate_cors_configuration("production", origins)
    
    def test_development_validation_allows_flexibility(self):
        """Test that development validation is more flexible"""
        origins = ["http://localhost:3000", "*", "https://app.xorb.enterprise"]
        
        # Should return True in development
        assert validate_cors_configuration("development", origins)
    
    @patch('src.api.app.middleware.secure_cors.logger')
    def test_suspicious_origin_warning(self, mock_logger):
        """Test that suspicious origins generate warnings"""
        origins = ["https://app.xorb.enterprise", "https://test.ngrok.io"]
        
        validate_cors_configuration("production", origins)
        
        # Should log warning for ngrok domain
        mock_logger.warning.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])