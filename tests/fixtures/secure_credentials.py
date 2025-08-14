"""
Secure Test Credential Generation

This module provides secure credential generation for testing without
exposing hardcoded secrets in the codebase.

Features:
- Cryptographically secure credential generation
- No hardcoded test credentials
- Proper entropy validation
- Test isolation with unique credentials per test
"""

import secrets
import string
import pytest
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class TestCredentials:
    """Container for test credentials"""
    username: str
    password: str
    email: str
    jwt_secret: str
    api_key: str
    tenant_id: str


class SecureTestCredentialGenerator:
    """Generate secure test credentials dynamically"""
    
    def __init__(self):
        """Initialize credential generator"""
        self.min_password_length = 32
        self.min_jwt_secret_length = 64
        self.min_api_key_length = 32
    
    def generate_username(self, prefix: str = "test_user") -> str:
        """Generate unique test username"""
        suffix = secrets.token_hex(8)
        return f"{prefix}_{suffix}"
    
    def generate_password(self) -> str:
        """Generate secure test password"""
        # Use a mix of character types for high entropy
        chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
        
        # Generate password with high entropy
        password = ''.join(secrets.choice(chars) for _ in range(self.min_password_length))
        
        # Ensure it contains at least one of each character type
        if not any(c.islower() for c in password):
            password = password[:-1] + secrets.choice(string.ascii_lowercase)
        if not any(c.isupper() for c in password):
            password = password[:-1] + secrets.choice(string.ascii_uppercase)
        if not any(c.isdigit() for c in password):
            password = password[:-1] + secrets.choice(string.digits)
        if not any(c in "!@#$%^&*()_+-=" for c in password):
            password = password[:-1] + secrets.choice("!@#$%^&*()_+-=")
        
        return password
    
    def generate_email(self, domain: str = "test.xorb.local") -> str:
        """Generate test email address"""
        username = secrets.token_hex(8)
        return f"test_{username}@{domain}"
    
    def generate_jwt_secret(self) -> str:
        """Generate secure JWT secret for testing"""
        return secrets.token_urlsafe(self.min_jwt_secret_length)
    
    def generate_api_key(self) -> str:
        """Generate secure API key for testing"""
        return secrets.token_urlsafe(self.min_api_key_length)
    
    def generate_tenant_id(self) -> str:
        """Generate unique tenant ID"""
        return f"tenant_{secrets.token_hex(16)}"
    
    def generate_full_credentials(self, username_prefix: str = "test_user") -> TestCredentials:
        """Generate complete set of test credentials"""
        return TestCredentials(
            username=self.generate_username(username_prefix),
            password=self.generate_password(),
            email=self.generate_email(),
            jwt_secret=self.generate_jwt_secret(),
            api_key=self.generate_api_key(),
            tenant_id=self.generate_tenant_id()
        )


# Global generator instance
_generator = SecureTestCredentialGenerator()


@pytest.fixture
def secure_test_credentials() -> TestCredentials:
    """Pytest fixture for secure test credentials"""
    return _generator.generate_full_credentials()


@pytest.fixture
def admin_test_credentials() -> TestCredentials:
    """Pytest fixture for admin test credentials"""
    return _generator.generate_full_credentials("admin_user")


@pytest.fixture
def test_jwt_secret() -> str:
    """Pytest fixture for JWT secret"""
    return _generator.generate_jwt_secret()


@pytest.fixture
def test_api_key() -> str:
    """Pytest fixture for API key"""
    return _generator.generate_api_key()


@pytest.fixture
def test_tenant_credentials() -> Dict[str, TestCredentials]:
    """Pytest fixture for multi-tenant test credentials"""
    return {
        "tenant_a": _generator.generate_full_credentials("tenant_a_user"),
        "tenant_b": _generator.generate_full_credentials("tenant_b_user"),
        "tenant_c": _generator.generate_full_credentials("tenant_c_user")
    }


# Convenience functions for direct usage
def get_test_user_credentials() -> TestCredentials:
    """Get test user credentials"""
    return _generator.generate_full_credentials()


def get_admin_credentials() -> TestCredentials:
    """Get admin credentials"""
    return _generator.generate_full_credentials("admin")


def get_secure_jwt_secret() -> str:
    """Get secure JWT secret for testing"""
    return _generator.generate_jwt_secret()


def get_test_database_url() -> str:
    """Get test database URL with secure credentials"""
    db_password = _generator.generate_password()
    return f"postgresql://test_user:{db_password}@localhost:5432/xorb_test"


def get_test_redis_url() -> str:
    """Get test Redis URL with secure credentials"""
    redis_password = _generator.generate_password()
    return f"redis://:{redis_password}@localhost:6379/1"


def validate_credential_security(credentials: TestCredentials) -> bool:
    """Validate that credentials meet security requirements"""
    checks = [
        len(credentials.password) >= 32,
        len(credentials.jwt_secret) >= 64,
        len(credentials.api_key) >= 32,
        any(c.islower() for c in credentials.password),
        any(c.isupper() for c in credentials.password),
        any(c.isdigit() for c in credentials.password),
        any(c in "!@#$%^&*()_+-=" for c in credentials.password),
        "@" in credentials.email,
        "test" in credentials.username
    ]
    
    return all(checks)


# Example usage for manual testing
if __name__ == "__main__":
    # Generate test credentials
    creds = get_test_user_credentials()
    print(f"Generated secure test credentials:")
    print(f"Username: {creds.username}")
    print(f"Password length: {len(creds.password)}")
    print(f"JWT secret length: {len(creds.jwt_secret)}")
    print(f"API key length: {len(creds.api_key)}")
    print(f"Security validation: {validate_credential_security(creds)}")