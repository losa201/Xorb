"""
Integration tests for the unified authentication system
Tests all authentication flows, JWT management, and security features
"""

import pytest
import asyncio
import redis.asyncio as redis
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import json

from src.api.app.services.unified_auth_service_consolidated import UnifiedAuthService
from src.api.app.domain.entities import User, AuthToken
from src.api.app.domain.exceptions import InvalidCredentials, AccountLocked, ValidationError
from src.common.jwt_manager import JWTManager


class TestUnifiedAuthIntegration:
    """Integration tests for unified authentication system"""
    
    @pytest.fixture
    async def redis_client(self):
        """Create test Redis client"""
        client = redis.from_url("redis://localhost:6379/1")  # Test DB
        await client.flushdb()  # Clean test database
        yield client
        await client.flushdb()
        await client.close()
    
    @pytest.fixture
    def mock_user_repository(self):
        """Mock user repository"""
        repo = AsyncMock()
        test_user = User(
            id="test-user-id",
            username="testuser",
            email="test@example.com",
            password_hash="$argon2id$v=19$m=65536,t=3,p=2$hash",
            roles=["user"],
            is_active=True
        )
        repo.get_by_username.return_value = test_user
        repo.get_by_id.return_value = test_user
        return repo
    
    @pytest.fixture
    def mock_token_repository(self):
        """Mock token repository"""
        repo = AsyncMock()
        repo.save_token.return_value = True
        repo.get_by_token.return_value = None
        repo.cleanup_expired_tokens.return_value = 0
        return repo
    
    @pytest.fixture
    async def auth_service(self, redis_client, mock_user_repository, mock_token_repository):
        """Create unified auth service instance"""
        return UnifiedAuthService(
            user_repository=mock_user_repository,
            token_repository=mock_token_repository,
            redis_client=redis_client,
            secret_key="test-secret-key",
            algorithm="HS256",
            access_token_expire_minutes=30
        )
    
    @pytest.mark.asyncio
    async def test_password_security_integration(self, auth_service):
        """Test password hashing and verification integration"""
        # Test password hashing
        password = "TestPassword123!"
        hashed = await auth_service.hash_password(password)
        
        assert hashed is not None
        assert hashed != password
        assert hashed.startswith("$argon2id$")
        
        # Test password verification
        is_valid = await auth_service.verify_password(password, hashed)
        assert is_valid is True
        
        # Test wrong password
        is_invalid = await auth_service.verify_password("WrongPassword", hashed)
        assert is_invalid is False
    
    @pytest.mark.asyncio
    async def test_jwt_token_lifecycle(self, auth_service, mock_user_repository):
        """Test complete JWT token lifecycle"""
        user = User(
            id="test-user",
            username="testuser",
            email="test@example.com",
            roles=["user", "admin"]
        )
        
        # Create access token
        access_token, expires_at = auth_service.create_access_token(user)
        assert access_token is not None
        assert expires_at > datetime.utcnow()
        
        # Verify token
        payload = auth_service.verify_token(access_token)
        assert payload is not None
        assert payload["sub"] == "test-user"
        assert payload["username"] == "testuser"
        assert "user" in payload["roles"]
        assert "admin" in payload["roles"]
        
        # Create refresh token
        refresh_token, refresh_expires = auth_service.create_refresh_token(user)
        assert refresh_token is not None
        assert refresh_expires > expires_at  # Refresh token should last longer
        
        # Verify refresh token
        refresh_payload = auth_service.verify_token(refresh_token)
        assert refresh_payload["type"] == "refresh"
    
    @pytest.mark.asyncio
    async def test_account_lockout_integration(self, auth_service, redis_client):
        """Test account lockout mechanism"""
        user_id = "test-user-lockout"
        client_ip = "192.168.1.100"
        
        # Simulate multiple failed attempts
        for i in range(5):
            await auth_service.record_failed_attempt(user_id, client_ip)
        
        # Check if account is locked
        is_locked = await auth_service.check_account_lockout(user_id, client_ip)
        assert is_locked is True
        
        # Verify lockout data in Redis
        lock_key = f"account_lock:{user_id}"
        lockout_data = await redis_client.get(lock_key)
        assert lockout_data is not None
        
        lockout_info = json.loads(lockout_data)
        assert lockout_info["reason"] == "too_many_failed_attempts"
        assert lockout_info["client_ip"] == client_ip
    
    @pytest.mark.asyncio
    async def test_api_key_management(self, auth_service, redis_client):
        """Test API key creation and validation"""
        user_id = "test-user-api"
        key_name = "test-api-key"
        scopes = ["read", "write"]
        
        # Create API key
        raw_key, key_hash = await auth_service.create_api_key(user_id, key_name, scopes)
        
        assert raw_key.startswith("xorb_")
        assert len(raw_key) > 40
        assert key_hash != raw_key
        
        # Validate API key
        key_data = await auth_service.validate_api_key(raw_key)
        assert key_data is not None
        assert key_data["user_id"] == user_id
        assert key_data["name"] == key_name
        assert key_data["scopes"] == scopes
        assert key_data["usage_count"] == 1
        
        # Second validation should increment usage
        key_data_2 = await auth_service.validate_api_key(raw_key)
        assert key_data_2["usage_count"] == 2
    
    @pytest.mark.asyncio
    async def test_permission_system_integration(self, auth_service):
        """Test role-based permission system"""
        # Test admin permissions
        admin_permissions = auth_service._get_user_permissions(["admin"])
        assert len(admin_permissions) > 0
        assert "agent:create" in admin_permissions
        assert "config:write" in admin_permissions
        
        # Test user permissions
        user_permissions = auth_service._get_user_permissions(["readonly"])
        assert "task:read" in user_permissions
        assert "agent:create" not in user_permissions
        
        # Test permission checking
        assert auth_service.has_permission(admin_permissions, "agent:create") is True
        assert auth_service.has_permission(user_permissions, "agent:create") is False
    
    @pytest.mark.asyncio
    async def test_full_authentication_flow(self, auth_service, mock_user_repository):
        """Test complete authentication flow"""
        # Setup test user
        mock_user_repository.get_by_username.return_value = User(
            id="flow-test-user",
            username="flowtest",
            email="flow@test.com",
            password_hash=await auth_service.hash_password("TestPassword123!"),
            roles=["user"],
            is_active=True
        )
        
        # Test successful authentication
        credentials = {
            "username": "flowtest",
            "password": "TestPassword123!",
            "client_ip": "192.168.1.1"
        }
        
        result = await auth_service.authenticate_user(credentials)
        
        assert result.success is True
        assert result.user is not None
        assert result.access_token is not None
        assert result.refresh_token is not None
        assert result.expires_at is not None
        
        # Test token validation
        validated_user = await auth_service.validate_token(result.access_token)
        assert validated_user is not None
        assert validated_user.username == "flowtest"
    
    @pytest.mark.asyncio
    async def test_token_blacklist_integration(self, auth_service, redis_client):
        """Test token blacklisting functionality"""
        user = User(id="blacklist-test", username="blacktest", email="black@test.com")
        
        # Create token
        access_token, _ = auth_service.create_access_token(user)
        
        # Verify token works
        payload = auth_service.verify_token(access_token)
        assert payload is not None
        
        # Revoke token
        revoked = await auth_service.revoke_token(access_token)
        assert revoked is True
        
        # Verify token is blacklisted
        blacklist_key = f"blacklist:{access_token}"
        blacklisted = await redis_client.exists(blacklist_key)
        assert blacklisted == 1
        
        # Validate token should now fail
        validated_user = await auth_service.validate_token(access_token)
        assert validated_user is None
    
    @pytest.mark.asyncio
    async def test_security_context_creation(self, auth_service):
        """Test security context creation"""
        user = User(
            id="context-test",
            username="contextuser",
            email="context@test.com",
            roles=["user", "analyst"]
        )
        
        # Mock request object
        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.50"
        mock_request.headers = {
            "X-Device-Fingerprint": "device123",
            "X-Session-ID": "session456"
        }
        
        context = auth_service.create_security_context(user, mock_request)
        
        assert context.user_id == "context-test"
        assert context.username == "contextuser"
        assert "user" in context.roles
        assert "analyst" in context.roles
        assert context.client_ip == "192.168.1.50"
        assert context.device_fingerprint == "device123"
        assert context.session_id == "session456"
        
        # Check permissions are populated
        assert len(context.permissions) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_authentication(self, auth_service, mock_user_repository):
        """Test concurrent authentication requests"""
        # Setup user
        mock_user_repository.get_by_username.return_value = User(
            id="concurrent-test",
            username="concurrent",
            email="concurrent@test.com",
            password_hash=await auth_service.hash_password("TestPassword123!"),
            roles=["user"],
            is_active=True
        )
        
        # Create multiple concurrent authentication requests
        async def authenticate():
            credentials = {
                "username": "concurrent",
                "password": "TestPassword123!",
                "client_ip": "192.168.1.200"
            }
            return await auth_service.authenticate_user(credentials)
        
        # Run 10 concurrent authentications
        tasks = [authenticate() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result.success is True
            assert result.access_token is not None
        
        # All tokens should be different
        tokens = [result.access_token for result in results]
        assert len(set(tokens)) == 10
    
    @pytest.mark.asyncio
    async def test_password_strength_validation(self, auth_service):
        """Test password strength validation"""
        # Test weak passwords
        weak_passwords = [
            "123456",           # Too short
            "password",         # No numbers, uppercase, special chars
            "Password123",      # No special characters
            "PASSWORD123!",     # No lowercase
            "password123!",     # No uppercase
        ]
        
        for weak_password in weak_passwords:
            with pytest.raises(ValidationError):
                await auth_service.hash_password(weak_password)
        
        # Test strong password
        strong_password = "SecurePassword123!"
        hashed = await auth_service.hash_password(strong_password)
        assert hashed is not None
    
    @pytest.mark.asyncio
    async def test_refresh_token_flow(self, auth_service, mock_token_repository):
        """Test refresh token flow"""
        user = User(id="refresh-test", username="refreshuser", email="refresh@test.com")
        
        # Create refresh token
        refresh_token, _ = auth_service.create_refresh_token(user)
        
        # Mock token repository to return the refresh token
        mock_auth_token = AuthToken(
            token=refresh_token,
            user_id=user.id,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        mock_token_repository.get_by_token.return_value = mock_auth_token
        
        # Use refresh token to get new access token
        new_access_token = await auth_service.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        
        # Verify new access token
        payload = auth_service.verify_token(new_access_token)
        assert payload is not None
        assert payload["sub"] == str(user.id)
        assert payload["type"] == "access"


class TestJWTManagerIntegration:
    """Integration tests for JWT Manager"""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance"""
        return JWTManager()
    
    @pytest.mark.asyncio
    async def test_jwt_manager_token_creation(self, jwt_manager):
        """Test JWT manager token creation"""
        payload = {"user_id": "test", "username": "testuser"}
        
        # Create token
        token = await jwt_manager.create_token(payload, expires_minutes=30)
        assert token is not None
        
        # Verify token
        verified_payload = await jwt_manager.verify_token(token)
        assert verified_payload["user_id"] == "test"
        assert verified_payload["username"] == "testuser"
        assert "exp" in verified_payload
        assert "iat" in verified_payload
    
    def test_jwt_manager_sync_operations(self, jwt_manager):
        """Test JWT manager synchronous operations"""
        payload = {"user_id": "sync-test", "roles": ["user"]}
        
        # Create token synchronously
        token = jwt_manager.create_token_sync(payload, expires_minutes=15)
        assert token is not None
        
        # Verify token synchronously
        verified_payload = jwt_manager.verify_token_sync(token)
        assert verified_payload["user_id"] == "sync-test"
        assert "user" in verified_payload["roles"]
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self, jwt_manager):
        """Test convenience functions"""
        from src.common.jwt_manager import create_access_token, verify_token
        
        # Create access token
        token = await create_access_token("user123", "testuser", ["admin"], 60)
        assert token is not None
        
        # Verify token
        payload = await verify_token(token)
        assert payload["sub"] == "user123"
        assert payload["username"] == "testuser"
        assert "admin" in payload["roles"]
        assert payload["type"] == "access"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])