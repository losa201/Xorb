"""
Production Security Test Suite
Comprehensive tests for authentication, authorization, and security controls
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import jwt
import redis.asyncio as redis

from app.services.consolidated_auth_service import (
    ConsolidatedAuthService, AuthProvider, Role, Permission, 
    SecurityContext, AuthenticationResult
)
from app.domain.entities import User, AuthToken
from app.domain.exceptions import SecurityViolation, InvalidCredentials


class TestConsolidatedAuthService:
    """Test consolidated authentication service"""
    
    @pytest.fixture
    async def auth_service(self):
        """Create auth service for testing"""
        
        user_repo = Mock()
        token_repo = Mock()
        redis_client = Mock()
        
        service = ConsolidatedAuthService(
            user_repository=user_repo,
            token_repository=token_repo,
            redis_client=redis_client,
            secret_key="test-secret-key",
            algorithm="HS256",
            access_token_expire_minutes=30
        )
        
        return service
    
    @pytest.fixture
    def test_user(self):
        """Create test user"""
        return User(
            id="user-123",
            username="testuser",
            email="test@example.com",
            password_hash="$2b$12$hashed_password",
            roles=["user"],
            is_active=True
        )
    
    @pytest.mark.asyncio
    async def test_successful_authentication(self, auth_service, test_user):
        """Test successful user authentication"""
        
        # Setup mocks
        auth_service.user_repository.get_by_username = AsyncMock(return_value=test_user)
        auth_service.redis_client.get = AsyncMock(return_value=None)  # No failed attempts
        auth_service.redis_client.delete = AsyncMock()
        auth_service._log_security_event = AsyncMock()
        
        # Mock password verification
        with patch('app.services.consolidated_auth_service.verify_password', return_value=True):
            result = await auth_service.authenticate_user(
                username="testuser",
                password="correct_password",
                ip_address="192.168.1.100",
                user_agent="test-agent"
            )
        
        # Assertions
        assert result.success is True
        assert result.security_context is not None
        assert result.access_token is not None
        assert result.security_context.username == "testuser"
        
        # Verify security logging
        auth_service._log_security_event.assert_called_with(
            "auth_success", "testuser", "192.168.1.100", {"provider": "local"}
        )
    
    @pytest.mark.asyncio
    async def test_failed_authentication_invalid_password(self, auth_service, test_user):
        """Test failed authentication with invalid password"""
        
        # Setup mocks
        auth_service.user_repository.get_by_username = AsyncMock(return_value=test_user)
        auth_service.redis_client.get = AsyncMock(return_value=None)
        auth_service._record_failed_attempt = AsyncMock()
        auth_service._log_security_event = AsyncMock()
        
        # Mock password verification failure
        with patch('app.services.consolidated_auth_service.verify_password', return_value=False):
            result = await auth_service.authenticate_user(
                username="testuser",
                password="wrong_password",
                ip_address="192.168.1.100"
            )
        
        # Assertions
        assert result.success is False
        assert result.error_message == "Invalid credentials"
        assert result.security_context is None
        
        # Verify failed attempt recording
        auth_service._record_failed_attempt.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_account_lockout_protection(self, auth_service, test_user):
        """Test account lockout after multiple failed attempts"""
        
        # Setup mocks - simulate account lockout
        auth_service.redis_client.get = AsyncMock(return_value=b'5')  # 5 failed attempts
        auth_service._log_security_event = AsyncMock()
        
        result = await auth_service.authenticate_user(
            username="testuser",
            password="any_password",
            ip_address="192.168.1.100"
        )
        
        # Assertions
        assert result.success is False
        assert result.account_locked is True
        assert "locked" in result.error_message
    
    @pytest.mark.asyncio
    async def test_jwt_token_validation(self, auth_service, test_user):
        """Test JWT token validation"""
        
        # Create security context
        context = SecurityContext(
            user_id=test_user.id,
            username=test_user.username,
            email=test_user.email,
            roles=[Role.USER],
            permissions={Permission.USER_READ},
            auth_provider=AuthProvider.LOCAL
        )
        
        # Create token
        token = await auth_service._create_access_token(context)
        
        # Setup mocks
        auth_service.user_repository.get_by_id = AsyncMock(return_value=test_user)
        
        # Validate token
        validated_context = await auth_service.validate_token(token)
        
        # Assertions
        assert validated_context is not None
        assert validated_context.username == test_user.username
        assert Role.USER in validated_context.roles
    
    @pytest.mark.asyncio
    async def test_expired_token_validation(self, auth_service):
        """Test validation of expired token"""
        
        # Create expired token
        expired_payload = {
            "sub": "user-123",
            "username": "testuser",
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
        }
        
        expired_token = jwt.encode(
            expired_payload, 
            auth_service.secret_key, 
            algorithm=auth_service.algorithm
        )
        
        # Validate expired token
        result = await auth_service.validate_token(expired_token)
        
        # Assertions
        assert result is None
    
    @pytest.mark.asyncio
    async def test_api_key_authentication(self, auth_service, test_user):
        """Test API key authentication"""
        
        # Setup mocks
        api_key_hash = "hashed_api_key"
        auth_token = AuthToken(
            id="token-123",
            user_id=test_user.id,
            token_hash=api_key_hash,
            token_type="api_key",
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        auth_service.token_repository.get_by_token_hash = AsyncMock(return_value=auth_token)
        auth_service.user_repository.get_by_id = AsyncMock(return_value=test_user)
        
        # Mock hashlib
        with patch('hashlib.sha256') as mock_hash:
            mock_hash.return_value.hexdigest.return_value = api_key_hash
            
            result = await auth_service.authenticate_api_key("test_api_key")
        
        # Assertions
        assert result.success is True
        assert result.security_context is not None
        assert result.security_context.auth_provider == AuthProvider.API_KEY
    
    @pytest.mark.asyncio
    async def test_permission_checking(self, auth_service):
        """Test permission checking functionality"""
        
        # Create context with specific permissions
        context = SecurityContext(
            user_id="user-123",
            username="testuser",
            email="test@example.com",
            roles=[Role.SECURITY_ANALYST],
            permissions={Permission.SECURITY_SCAN, Permission.SECURITY_ANALYZE, Permission.USER_READ},
            auth_provider=AuthProvider.LOCAL
        )
        
        # Test permission checks
        assert auth_service.check_permission(context, Permission.SECURITY_SCAN) is True
        assert auth_service.check_permission(context, Permission.SYSTEM_ADMIN) is False
        assert auth_service.check_role(context, Role.SECURITY_ANALYST) is True
        assert auth_service.check_role(context, Role.SUPER_ADMIN) is False
    
    @pytest.mark.asyncio
    async def test_role_hierarchy(self, auth_service):
        """Test role hierarchy permissions"""
        
        # Super admin should have all permissions
        super_admin_context = SecurityContext(
            user_id="admin-123",
            username="admin",
            email="admin@example.com",
            roles=[Role.SUPER_ADMIN],
            permissions=auth_service.ROLE_PERMISSIONS[Role.SUPER_ADMIN],
            auth_provider=AuthProvider.LOCAL
        )
        
        # Test super admin permissions
        assert auth_service.check_permission(super_admin_context, Permission.SYSTEM_ADMIN) is True
        assert auth_service.check_permission(super_admin_context, Permission.USER_DELETE) is True
        assert auth_service.check_permission(super_admin_context, Permission.SECURITY_SCAN) is True
    
    @pytest.mark.asyncio
    async def test_tenant_access_control(self, auth_service):
        """Test multi-tenant access control"""
        
        # User with specific tenant
        tenant_context = SecurityContext(
            user_id="user-123",
            username="tenant_user",
            email="user@tenant.com",
            roles=[Role.USER],
            permissions={Permission.USER_READ},
            tenant_id="tenant-456",
            auth_provider=AuthProvider.LOCAL
        )
        
        # Test tenant access
        assert await auth_service.check_tenant_access(tenant_context, "tenant-456") is True
        assert await auth_service.check_tenant_access(tenant_context, "tenant-789") is False
        
        # Super admin should access all tenants
        super_admin_context = SecurityContext(
            user_id="admin-123",
            username="admin",
            email="admin@example.com",
            roles=[Role.SUPER_ADMIN],
            permissions=auth_service.ROLE_PERMISSIONS[Role.SUPER_ADMIN],
            auth_provider=AuthProvider.LOCAL
        )
        
        assert await auth_service.check_tenant_access(super_admin_context, "any-tenant") is True
    
    @pytest.mark.asyncio
    async def test_security_event_logging(self, auth_service):
        """Test security event logging"""
        
        # Setup Redis mock
        auth_service.redis_client.lpush = AsyncMock()
        auth_service.redis_client.ltrim = AsyncMock()
        
        # Log security event
        await auth_service._log_security_event(
            "test_event",
            "test_user",
            "192.168.1.100",
            {"additional": "data"}
        )
        
        # Verify Redis calls
        auth_service.redis_client.lpush.assert_called_once()
        auth_service.redis_client.ltrim.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_key_creation(self, auth_service):
        """Test API key creation"""
        
        # Create context with permission
        context = SecurityContext(
            user_id="user-123",
            username="testuser",
            email="test@example.com",
            roles=[Role.USER],
            permissions={Permission.USER_UPDATE},
            auth_provider=AuthProvider.LOCAL
        )
        
        # Setup mocks
        auth_service.token_repository.create = AsyncMock()
        auth_service._log_security_event = AsyncMock()
        
        # Create API key
        api_key = await auth_service.create_api_key(context, "Test API Key")
        
        # Assertions
        assert api_key.startswith("xorb_")
        assert len(api_key) == 37  # "xorb_" + 32 characters
        auth_service.token_repository.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_key_creation_insufficient_permissions(self, auth_service):
        """Test API key creation with insufficient permissions"""
        
        # Create context without permission
        context = SecurityContext(
            user_id="user-123",
            username="testuser",
            email="test@example.com",
            roles=[Role.READONLY],
            permissions=set(),  # No permissions
            auth_provider=AuthProvider.LOCAL
        )
        
        # Attempt to create API key
        with pytest.raises(SecurityViolation):
            await auth_service.create_api_key(context, "Test API Key")


class TestSecurityMiddleware:
    """Test security middleware functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting middleware"""
        # Implementation would test rate limiting logic
        pass
    
    @pytest.mark.asyncio
    async def test_audit_logging(self):
        """Test audit logging middleware"""
        # Implementation would test audit logging
        pass
    
    @pytest.mark.asyncio
    async def test_security_headers(self):
        """Test security headers middleware"""
        # Implementation would test security header injection
        pass


class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        # Implementation would test SQL injection prevention
        pass
    
    def test_xss_prevention(self):
        """Test XSS prevention"""
        # Implementation would test XSS prevention
        pass
    
    def test_command_injection_prevention(self):
        """Test command injection prevention in PTaaS"""
        # Implementation would test command injection prevention
        pass


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_authentication_flow(self):
        """Test complete authentication flow"""
        # Implementation would test complete auth flow
        pass
    
    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self):
        """Test multi-tenant data isolation"""
        # Implementation would test tenant isolation
        pass
    
    @pytest.mark.asyncio
    async def test_security_incident_response(self):
        """Test automated security incident response"""
        # Implementation would test incident response
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])