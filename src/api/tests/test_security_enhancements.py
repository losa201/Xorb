"""
Comprehensive tests for security enhancements
Tests Argon2 hashing, audit logging, rate limiting, and MFA
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from ..app.services.auth_security_service import AuthSecurityService
from ..app.services.mfa_service import MFAService, TOTPGenerator
from ..app.middleware.audit_logging import AuditLoggingMiddleware, AuditLogger, AuditEvent
from ..app.middleware.advanced_rate_limiter import RateLimitingMiddleware, AdaptiveRateLimiter, RateLimitRule
from ..app.domain.entities import User
from ..app.domain.exceptions import (
    AccountLocked, SecurityViolation, ValidationError, 
    MFARequired, InvalidCredentials
)


@pytest.fixture
async def redis_client():
    """Mock Redis client for testing"""
    mock_redis = AsyncMock(spec=redis.Redis)
    
    # Mock Redis operations
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.incr.return_value = 1
    mock_redis.expire.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = 0
    mock_redis.hgetall.return_value = {}
    mock_redis.hset.return_value = True
    mock_redis.lpush.return_value = 1
    mock_redis.lrange.return_value = []
    mock_redis.zcard.return_value = 0
    mock_redis.zadd.return_value = True
    mock_redis.zremrangebyscore.return_value = 0
    
    # Mock pipeline
    pipeline_mock = AsyncMock()
    pipeline_mock.execute.return_value = [None, 0, True, True]
    mock_redis.pipeline.return_value.__aenter__.return_value = pipeline_mock
    
    return mock_redis


@pytest.fixture
def test_user():
    """Create test user"""
    return User(
        id=uuid4(),
        username="testuser",
        email="test@example.com",
        roles=["user"],
        is_active=True
    )


@pytest.fixture
def user_repository_mock():
    """Mock user repository"""
    mock_repo = AsyncMock()
    mock_repo.get_by_username.return_value = None
    mock_repo.get_by_id.return_value = None
    mock_repo.update.return_value = True
    return mock_repo


class TestAuthSecurityService:
    """Test enhanced authentication security service"""
    
    @pytest.fixture
    async def auth_security_service(self, user_repository_mock, redis_client):
        """Create auth security service for testing"""
        return AuthSecurityService(
            user_repository=user_repository_mock,
            redis_client=redis_client,
            max_failed_attempts=3,
            lockout_duration_minutes=5
        )
    
    @pytest.mark.asyncio
    async def test_password_hashing(self, auth_security_service):
        """Test Argon2 password hashing"""
        password = "SecurePassword123!"
        user_id = str(uuid4())
        
        # Hash password
        hashed = await auth_security_service.hash_password(password, user_id)
        
        # Verify properties
        assert hashed is not None
        assert len(hashed) > 50  # Argon2 hashes are long
        assert hashed.startswith("$argon2")
        assert hashed != password  # Not plain text
    
    @pytest.mark.asyncio
    async def test_password_verification(self, auth_security_service, test_user, redis_client):
        """Test password verification with security monitoring"""
        password = "SecurePassword123!"
        hashed = await auth_security_service.hash_password(password, str(test_user.id))
        
        # Mock account not locked
        redis_client.exists.return_value = 0
        
        # Test successful verification
        is_valid = await auth_security_service.verify_password(
            password, hashed, test_user, "127.0.0.1"
        )
        
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_password_verification_failure(self, auth_security_service, test_user, redis_client):
        """Test failed password verification handling"""
        password = "SecurePassword123!"
        wrong_password = "WrongPassword456!"
        hashed = await auth_security_service.hash_password(password, str(test_user.id))
        
        # Mock account not locked
        redis_client.exists.return_value = 0
        
        # Test failed verification
        is_valid = await auth_security_service.verify_password(
            wrong_password, hashed, test_user, "127.0.0.1"
        )
        
        assert is_valid is False
        # Verify failed attempt was logged
        redis_client.incr.assert_called()
    
    @pytest.mark.asyncio
    async def test_account_lockout(self, auth_security_service, test_user, redis_client):
        """Test account lockout after failed attempts"""
        # Mock account is locked
        redis_client.exists.return_value = 1
        
        with pytest.raises(AccountLocked):
            await auth_security_service.verify_password(
                "anypassword", "anyhash", test_user, "127.0.0.1"
            )
    
    @pytest.mark.asyncio
    async def test_password_strength_validation(self, auth_security_service):
        """Test password strength validation"""
        # Test weak password
        with pytest.raises(ValidationError, match="at least 12 characters"):
            await auth_security_service.hash_password("weak")
        
        # Test password without uppercase
        with pytest.raises(ValidationError, match="uppercase letter"):
            await auth_security_service.hash_password("nouppercase123!")
        
        # Test password without special characters
        with pytest.raises(ValidationError, match="special character"):
            await auth_security_service.hash_password("NoSpecialChars123")
    
    @pytest.mark.asyncio
    async def test_password_strength_checker(self, auth_security_service):
        """Test password strength analysis"""
        weak_password = "password123"
        strong_password = "SecureP@ssw0rd2024!"
        
        weak_result = await auth_security_service.check_password_strength(weak_password)
        strong_result = await auth_security_service.check_password_strength(strong_password)
        
        assert weak_result["is_strong"] is False
        assert strong_result["is_strong"] is True
        assert strong_result["has_uppercase"] is True
        assert strong_result["has_special"] is True
    
    @pytest.mark.asyncio
    async def test_secure_password_generation(self, auth_security_service):
        """Test secure password generation"""
        password = await auth_security_service.generate_secure_password(16)
        
        assert len(password) == 16
        assert any(c.isupper() for c in password)
        assert any(c.islower() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)


class TestAuditLogging:
    """Test comprehensive audit logging"""
    
    @pytest.fixture
    async def audit_logger(self, redis_client):
        """Create audit logger for testing"""
        return AuditLogger(redis_client)
    
    @pytest.mark.asyncio
    async def test_audit_event_creation(self):
        """Test audit event creation"""
        event = AuditEvent(
            event_id="test-event-1",
            event_type="authentication",
            user_id="user-123",
            client_ip="192.168.1.100",
            outcome="success"
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_id"] == "test-event-1"
        assert event_dict["event_type"] == "authentication"
        assert event_dict["user_id"] == "user-123"
        assert event_dict["outcome"] == "success"
        assert "timestamp" in event_dict
        assert "compliance_fields" in event_dict
    
    @pytest.mark.asyncio
    async def test_audit_event_logging(self, audit_logger, redis_client):
        """Test audit event logging to Redis"""
        event = AuditEvent(
            event_id="test-event-2",
            event_type="data_access",
            user_id="user-456",
            resource="/api/sensitive-data",
            risk_level="high"
        )
        
        await audit_logger.log_event(event)
        
        # Verify Redis operations were called
        redis_client.lpush.assert_called()
        redis_client.expire.assert_called()
    
    @pytest.mark.asyncio
    async def test_security_alert_generation(self, audit_logger, redis_client):
        """Test security alert generation"""
        # Mock multiple failed logins
        redis_client.get.return_value = b"5"  # 5 failed attempts
        
        event = AuditEvent(
            event_id="test-event-3",
            event_type="authentication",
            user_id="user-789",
            client_ip="192.168.1.200",
            outcome="failure"
        )
        
        await audit_logger.log_event(event)
        
        # Should trigger security alert logging
        assert redis_client.lpush.call_count >= 2  # Event + alert
    
    @pytest.mark.asyncio
    async def test_audit_trail_retrieval(self, audit_logger, redis_client):
        """Test audit trail retrieval"""
        # Mock Redis response
        mock_events = [
            json.dumps({
                "event_id": "event-1",
                "event_type": "authentication",
                "user_id": "user-123",
                "timestamp": "2024-01-01T12:00:00"
            })
        ]
        redis_client.lrange.return_value = [e.encode() for e in mock_events]
        
        events = await audit_logger.get_audit_trail(user_id="user-123", limit=10)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "authentication"
        assert events[0]["user_id"] == "user-123"


class TestRateLimiting:
    """Test advanced rate limiting"""
    
    @pytest.fixture
    async def rate_limiter(self, redis_client):
        """Create rate limiter for testing"""
        return AdaptiveRateLimiter(redis_client)
    
    @pytest.mark.asyncio
    async def test_sliding_window_rate_limit(self, rate_limiter, redis_client):
        """Test sliding window rate limiting"""
        # Mock Redis pipeline results
        redis_client.pipeline.return_value.__aenter__.return_value.execute.return_value = [
            None, 5, True, True  # [zremrangebyscore, zcard, zadd, expire]
        ]
        
        # Create mock request
        request = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers = {}
        
        # Test within limit
        is_allowed, info = await rate_limiter.check_rate_limit(
            "test-user", "per_ip", request
        )
        
        assert is_allowed is True
        assert "limit" in info
        assert "remaining" in info
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, rate_limiter, redis_client):
        """Test rate limit exceeded scenario"""
        # Mock high request count
        redis_client.pipeline.return_value.__aenter__.return_value.execute.return_value = [
            None, 100, True, True  # Exceed the default limit
        ]
        
        request = MagicMock()
        request.client.host = "192.168.1.200"
        request.headers = {}
        
        is_allowed, info = await rate_limiter.check_rate_limit(
            "heavy-user", "per_ip", request
        )
        
        assert is_allowed is False
        assert info["remaining"] == 0
    
    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting(self, rate_limiter, redis_client):
        """Test adaptive rate limiting based on threat level"""
        # Mock high threat level
        redis_client.get.side_effect = lambda key: {
            "threat:brute_force:192.168.1.300": b"15",  # Above threshold
            "global_threat_level": b"0.5"
        }.get(key, None)
        
        request = MagicMock()
        request.client.host = "192.168.1.300"
        request.headers = {}
        
        # Mock normal Redis pipeline for rate limiting
        redis_client.pipeline.return_value.__aenter__.return_value.execute.return_value = [
            None, 1, True, True
        ]
        
        is_allowed, info = await rate_limiter.check_rate_limit(
            "suspicious-user", "per_ip", request
        )
        
        # Should have reduced limits due to threat level
        assert "limit" in info
    
    @pytest.mark.asyncio
    async def test_rate_limit_stats(self, rate_limiter, redis_client):
        """Test rate limiting statistics"""
        # Mock violation data
        mock_violations = [
            json.dumps({
                "identifier": "192.168.1.100",
                "rule": "per_ip",
                "timestamp": "2024-01-01T12:00:00"
            })
        ]
        redis_client.lrange.return_value = [v.encode() for v in mock_violations]
        
        stats = await rate_limiter.get_rate_limit_stats()
        
        assert "violations_today" in stats
        assert "top_violators" in stats
        assert "rule_stats" in stats


class TestMFA:
    """Test Multi-Factor Authentication"""
    
    @pytest.fixture
    async def mfa_service(self, redis_client):
        """Create MFA service for testing"""
        return MFAService(redis_client)
    
    def test_totp_generation(self):
        """Test TOTP secret and token generation"""
        secret = TOTPGenerator.generate_secret()
        
        assert secret is not None
        assert len(secret) == 32  # Base32 encoded
        
        # Generate and verify token
        token = TOTPGenerator.get_current_token(secret)
        assert len(token) == 6
        assert token.isdigit()
        
        # Verify token
        is_valid = TOTPGenerator.verify_token(secret, token)
        assert is_valid is True
    
    def test_totp_qr_code_generation(self):
        """Test TOTP QR code generation"""
        secret = TOTPGenerator.generate_secret()
        qr_code = TOTPGenerator.generate_qr_code(secret, "test@example.com")
        
        assert qr_code is not None
        assert len(qr_code) > 0
        assert isinstance(qr_code, bytes)
    
    @pytest.mark.asyncio
    async def test_mfa_setup(self, mfa_service, test_user, redis_client):
        """Test MFA setup process"""
        setup_data = await mfa_service.setup_totp(test_user)
        
        assert "method_id" in setup_data
        assert "secret" in setup_data
        assert "qr_code" in setup_data
        assert "backup_codes" in setup_data
        
        # Verify temporary storage
        redis_client.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_mfa_verification_setup(self, mfa_service, redis_client):
        """Test MFA setup verification"""
        # Mock temporary setup data
        method_data = {
            "id": "method-123",
            "user_id": "user-456",
            "method_type": "totp",
            "name": "Authenticator App",
            "secret": "JBSWY3DPEHPK3PXP",
            "is_verified": False,
            "created_at": datetime.utcnow().isoformat()
        }
        redis_client.get.return_value = json.dumps(method_data).encode()
        
        # Generate valid token
        token = TOTPGenerator.get_current_token("JBSWY3DPEHPK3PXP")
        
        result = await mfa_service.verify_totp_setup("user-456", "method-123", token)
        
        assert result is True
        redis_client.hset.assert_called()  # Method should be saved
        redis_client.delete.assert_called()  # Temp data should be removed
    
    @pytest.mark.asyncio
    async def test_mfa_challenge_creation(self, mfa_service, redis_client):
        """Test MFA challenge creation"""
        # Mock existing MFA methods
        methods_data = {
            b"method-1": json.dumps({
                "id": "method-1",
                "user_id": "user-123",
                "method_type": "totp",
                "name": "Authenticator",
                "secret": "JBSWY3DPEHPK3PXP",
                "is_verified": True,
                "created_at": datetime.utcnow().isoformat()
            })
        }
        redis_client.hgetall.return_value = methods_data
        
        challenge = await mfa_service.create_mfa_challenge("user-123")
        
        assert challenge.user_id == "user-123"
        assert challenge.method_type == "totp"
        assert challenge.challenge_id is not None
        
        # Verify challenge was stored
        redis_client.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_mfa_challenge_verification(self, mfa_service, redis_client):
        """Test MFA challenge verification"""
        # Mock challenge data
        challenge_data = {
            "challenge_id": "challenge-123",
            "user_id": "user-456",
            "method_id": "method-1",
            "method_type": "totp",
            "challenge_data": {"method_type": "totp"},
            "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            "attempts": 0,
            "max_attempts": 3
        }
        redis_client.get.return_value = json.dumps(challenge_data).encode()
        
        # Mock method data for secret retrieval
        methods_data = {
            b"method-1": json.dumps({
                "id": "method-1",
                "user_id": "user-456",
                "method_type": "totp",
                "secret": "JBSWY3DPEHPK3PXP",
                "is_verified": True,
                "created_at": datetime.utcnow().isoformat()
            })
        }
        redis_client.hgetall.return_value = methods_data
        
        # Generate valid token
        token = TOTPGenerator.get_current_token("JBSWY3DPEHPK3PXP")
        
        result = await mfa_service.verify_mfa_challenge(
            "challenge-123",
            {"token": token}
        )
        
        assert result is True
        redis_client.delete.assert_called()  # Challenge should be removed
    
    @pytest.mark.asyncio
    async def test_backup_codes(self, mfa_service, redis_client):
        """Test backup codes generation and verification"""
        codes = await mfa_service.generate_backup_codes("user-789")
        
        assert len(codes) == 10  # Default backup codes count
        assert all(len(code) == 8 for code in codes)
        assert all(code.isalnum() for code in codes)
        
        # Verify codes were stored (hashed)
        redis_client.setex.assert_called()
        
        # Test backup code verification
        # Mock stored hashed codes
        import hashlib
        test_code = codes[0]
        hashed_codes = [hashlib.sha256(code.encode()).hexdigest() for code in codes]
        redis_client.get.return_value = json.dumps(hashed_codes).encode()
        
        is_valid = await mfa_service.verify_backup_code("user-789", test_code)
        assert is_valid is True


class TestIntegration:
    """Integration tests for security enhancements"""
    
    @pytest.mark.asyncio
    async def test_middleware_integration(self, redis_client):
        """Test middleware integration with FastAPI"""
        app = FastAPI()
        
        # Add security middleware
        app.add_middleware(RateLimitingMiddleware, redis_client=redis_client)
        app.add_middleware(AuditLoggingMiddleware, redis_client=redis_client)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        client = TestClient(app)
        
        # Test request goes through middleware
        response = client.get("/test")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_security_workflow(self, redis_client, test_user):
        """Test complete security workflow"""
        # Initialize services
        user_repo_mock = AsyncMock()
        auth_service = AuthSecurityService(user_repo_mock, redis_client)
        mfa_service = MFAService(redis_client)
        audit_logger = AuditLogger(redis_client)
        
        # 1. Hash password
        password = "SecurePassword123!"
        hashed = await auth_service.hash_password(password, str(test_user.id))
        
        # 2. Setup MFA
        setup_data = await mfa_service.setup_totp(test_user)
        
        # 3. Verify MFA setup
        redis_client.get.return_value = json.dumps({
            "id": setup_data["method_id"],
            "user_id": str(test_user.id),
            "method_type": "totp",
            "secret": setup_data["secret"],
            "is_verified": False,
            "created_at": datetime.utcnow().isoformat()
        }).encode()
        
        token = TOTPGenerator.get_current_token(setup_data["secret"])
        await mfa_service.verify_totp_setup(str(test_user.id), setup_data["method_id"], token)
        
        # 4. Log security event
        event = AuditEvent(
            event_id=str(uuid4()),
            event_type="mfa_enabled",
            user_id=str(test_user.id),
            outcome="success"
        )
        await audit_logger.log_event(event)
        
        # Verify all operations completed without errors
        assert hashed is not None
        assert setup_data["method_id"] is not None
        
        # Verify Redis interactions
        assert redis_client.setex.call_count > 0
        assert redis_client.lpush.call_count > 0


@pytest.mark.asyncio
async def test_performance_benchmarks(redis_client):
    """Performance benchmarks for security operations"""
    auth_service = AuthSecurityService(AsyncMock(), redis_client)
    
    # Benchmark password hashing
    start_time = time.time()
    for _ in range(10):
        await auth_service.hash_password("TestPassword123!", "user-123")
    hash_time = (time.time() - start_time) / 10
    
    # Should be under 100ms per hash on modern hardware
    assert hash_time < 0.1, f"Password hashing too slow: {hash_time:.3f}s"
    
    # Benchmark TOTP verification
    secret = TOTPGenerator.generate_secret()
    token = TOTPGenerator.get_current_token(secret)
    
    start_time = time.time()
    for _ in range(100):
        TOTPGenerator.verify_token(secret, token)
    totp_time = (time.time() - start_time) / 100
    
    # TOTP should be very fast
    assert totp_time < 0.001, f"TOTP verification too slow: {totp_time:.6f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])