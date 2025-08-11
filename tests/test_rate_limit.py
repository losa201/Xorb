"""
Comprehensive Test Suite for Adaptive Rate Limiting System
Tests for policies, limiters, middleware, and integration scenarios
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from uuid import uuid4

import redis.asyncio as redis
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

# Import rate limiting components
from src.api.app.rate_limit import (
    RateLimitPolicy,
    RateLimitWindow,
    RateLimitScope,
    RateLimitMode,
    BurstStrategy,
    AdaptiveConfig,
    PolicyResolver,
    AdaptiveRateLimiter,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    CircuitBreakerLimiter,
    RateLimitResult,
    RateLimitMiddleware,
    create_default_policies,
    create_testing_policies
)

from src.api.app.auth.models import UserClaims


class TestRateLimitPolicies:
    """Test rate limit policy creation and resolution"""
    
    def test_rate_limit_window_creation(self):
        """Test rate limit window creation and validation"""
        # Valid window
        window = RateLimitWindow(
            duration_seconds=60,
            max_requests=100,
            burst_allowance=20,
            burst_strategy=BurstStrategy.ADAPTIVE
        )
        assert window.duration_seconds == 60
        assert window.max_requests == 100
        assert window.burst_allowance == 20
        
        # Invalid duration
        with pytest.raises(ValueError, match="Window duration must be positive"):
            RateLimitWindow(duration_seconds=0, max_requests=100)
        
        # Invalid max requests
        with pytest.raises(ValueError, match="Max requests must be positive"):
            RateLimitWindow(duration_seconds=60, max_requests=0)
        
        # Invalid burst allowance
        with pytest.raises(ValueError, match="Burst allowance cannot be negative"):
            RateLimitWindow(duration_seconds=60, max_requests=100, burst_allowance=-1)
    
    def test_adaptive_config_creation(self):
        """Test adaptive configuration creation"""
        config = AdaptiveConfig(
            enable_reputation_scoring=True,
            violation_penalty_multiplier=2.0,
            reputation_decay_hours=24,
            escalation_thresholds={3: 0.5, 5: 0.25, 10: 0.1}
        )
        
        assert config.enable_reputation_scoring is True
        assert config.violation_penalty_multiplier == 2.0
        assert config.reputation_decay_hours == 24
        assert config.escalation_thresholds[3] == 0.5
    
    def test_rate_limit_policy_creation(self):
        """Test rate limit policy creation and validation"""
        policy = RateLimitPolicy(
            name="test_policy",
            scope=RateLimitScope.USER,
            scope_values={"user123"},
            windows=[
                RateLimitWindow(duration_seconds=60, max_requests=100),
                RateLimitWindow(duration_seconds=3600, max_requests=1000)
            ],
            mode=RateLimitMode.ENFORCE,
            priority=10
        )
        
        assert policy.name == "test_policy"
        assert policy.scope == RateLimitScope.USER
        assert "user123" in policy.scope_values
        assert len(policy.windows) == 2
        assert policy.mode == RateLimitMode.ENFORCE
        assert policy.priority == 10
        assert policy.adaptive_config is not None  # Auto-created
    
    def test_policy_effective_limits_with_reputation(self):
        """Test reputation-based policy adjustment"""
        policy = RateLimitPolicy(
            name="test_policy",
            scope=RateLimitScope.USER,
            windows=[RateLimitWindow(duration_seconds=60, max_requests=100, burst_allowance=20)]
        )
        
        # Full reputation (1.0) - no adjustment
        effective_windows = policy.get_effective_limits(reputation_score=1.0)
        assert effective_windows[0].max_requests == 100
        assert effective_windows[0].burst_allowance == 20
        
        # Half reputation (0.5) - 50% reduction
        effective_windows = policy.get_effective_limits(reputation_score=0.5)
        assert effective_windows[0].max_requests == 50
        assert effective_windows[0].burst_allowance == 10
        
        # Very low reputation (0.1) - minimum limits
        effective_windows = policy.get_effective_limits(reputation_score=0.1)
        assert effective_windows[0].max_requests == 10
        assert effective_windows[0].burst_allowance == 2
    
    def test_policy_escalation_logic(self):
        """Test violation-based escalation"""
        policy = RateLimitPolicy(
            name="test_policy",
            scope=RateLimitScope.USER,
            adaptive_config=AdaptiveConfig(
                escalation_thresholds={3: 0.5, 5: 0.25, 10: 0.1}
            )
        )
        
        assert not policy.should_escalate(2)  # Below threshold
        assert policy.should_escalate(3)      # At threshold
        assert policy.should_escalate(10)     # Above threshold
        
        assert policy.get_escalation_factor(2) == 1.0   # No reduction
        assert policy.get_escalation_factor(3) == 0.5   # 50% reduction
        assert policy.get_escalation_factor(5) == 0.25  # 75% reduction
        assert policy.get_escalation_factor(10) == 0.1  # 90% reduction
    
    def test_policy_serialization(self):
        """Test policy serialization and deserialization"""
        original_policy = RateLimitPolicy(
            name="test_policy",
            scope=RateLimitScope.ENDPOINT,
            scope_values={"/api/v1/test"},
            windows=[RateLimitWindow(duration_seconds=60, max_requests=100)],
            mode=RateLimitMode.SHADOW,
            priority=5,
            description="Test policy for serialization"
        )
        
        # Serialize to dictionary
        policy_dict = original_policy.to_dict()
        assert policy_dict["name"] == "test_policy"
        assert policy_dict["scope"] == "endpoint"
        assert "/api/v1/test" in policy_dict["scope_values"]
        
        # Deserialize from dictionary
        restored_policy = RateLimitPolicy.from_dict(policy_dict)
        assert restored_policy.name == original_policy.name
        assert restored_policy.scope == original_policy.scope
        assert restored_policy.scope_values == original_policy.scope_values
        assert restored_policy.mode == original_policy.mode


class TestPolicyResolver:
    """Test hierarchical policy resolution"""
    
    def setup_method(self):
        """Setup test policies"""
        self.resolver = PolicyResolver()
        
        # Add test policies with different scopes and priorities
        policies = [
            # Global policy (lowest priority)
            RateLimitPolicy(
                name="global_default",
                scope=RateLimitScope.GLOBAL,
                scope_values={"*"},
                windows=[RateLimitWindow(duration_seconds=60, max_requests=100)],
                priority=1
            ),
            
            # IP-based policy
            RateLimitPolicy(
                name="ip_limits",
                scope=RateLimitScope.IP,
                scope_values={"192.168.1.100"},
                windows=[RateLimitWindow(duration_seconds=60, max_requests=50)],
                priority=10
            ),
            
            # Tenant policy
            RateLimitPolicy(
                name="tenant_limits",
                scope=RateLimitScope.TENANT,
                scope_values={"tenant123"},
                windows=[RateLimitWindow(duration_seconds=60, max_requests=200)],
                priority=20
            ),
            
            # Role-based policy
            RateLimitPolicy(
                name="admin_limits",
                scope=RateLimitScope.ROLE,
                scope_values={"admin"},
                windows=[RateLimitWindow(duration_seconds=60, max_requests=500)],
                priority=30
            ),
            
            # User-specific policy
            RateLimitPolicy(
                name="user_limits",
                scope=RateLimitScope.USER,
                scope_values={"user123"},
                windows=[RateLimitWindow(duration_seconds=60, max_requests=300)],
                priority=40
            ),
            
            # Endpoint-specific policy (highest priority)
            RateLimitPolicy(
                name="auth_endpoint_limits",
                scope=RateLimitScope.ENDPOINT,
                scope_values={"/api/v1/auth/login"},
                windows=[RateLimitWindow(duration_seconds=60, max_requests=5)],
                priority=50
            )
        ]
        
        for policy in policies:
            self.resolver.add_policy(policy)
    
    def test_policy_resolution_hierarchy(self):
        """Test hierarchical policy resolution"""
        # Endpoint policy should have highest priority
        policy = self.resolver.resolve_policy(
            ip_address="192.168.1.100",
            user_id="user123",
            tenant_id="tenant123",
            roles={"admin"},
            endpoint="/api/v1/auth/login"
        )
        assert policy.name == "auth_endpoint_limits"
        assert policy.windows[0].max_requests == 5
        
        # User policy should override tenant/role when no endpoint match
        policy = self.resolver.resolve_policy(
            ip_address="192.168.1.100",
            user_id="user123",
            tenant_id="tenant123",
            roles={"admin"},
            endpoint="/api/v1/other"
        )
        assert policy.name == "user_limits"
        assert policy.windows[0].max_requests == 300
        
        # Role policy should override tenant when no user/endpoint match
        policy = self.resolver.resolve_policy(
            ip_address="192.168.1.100",
            user_id="other_user",
            tenant_id="tenant123",
            roles={"admin"},
            endpoint="/api/v1/other"
        )
        assert policy.name == "admin_limits"
        assert policy.windows[0].max_requests == 500
    
    def test_policy_fallback_to_global(self):
        """Test fallback to global policy when no specific match"""
        policy = self.resolver.resolve_policy(
            ip_address="10.0.0.1",
            user_id="unknown_user",
            tenant_id="unknown_tenant",
            roles={"user"},
            endpoint="/api/v1/unknown"
        )
        assert policy.name == "global_default"
        assert policy.windows[0].max_requests == 100
    
    def test_policy_caching(self):
        """Test policy resolution caching"""
        # First resolution
        start_time = time.time()
        policy1 = self.resolver.resolve_policy(
            user_id="user123",
            endpoint="/api/v1/test"
        )
        first_duration = time.time() - start_time
        
        # Second resolution (should be cached)
        start_time = time.time()
        policy2 = self.resolver.resolve_policy(
            user_id="user123",
            endpoint="/api/v1/test"
        )
        second_duration = time.time() - start_time
        
        assert policy1.name == policy2.name
        # Cached resolution should be faster
        assert second_duration < first_duration
    
    def test_policy_management(self):
        """Test adding and removing policies"""
        initial_count = len(self.resolver.policies)
        
        # Add new policy
        new_policy = RateLimitPolicy(
            name="test_new_policy",
            scope=RateLimitScope.USER,
            scope_values={"new_user"}
        )
        self.resolver.add_policy(new_policy)
        assert len(self.resolver.policies) == initial_count + 1
        
        # Remove policy
        removed = self.resolver.remove_policy(new_policy.policy_id)
        assert removed is True
        assert len(self.resolver.policies) == initial_count
        
        # Try to remove non-existent policy
        removed = self.resolver.remove_policy("non_existent_id")
        assert removed is False


@pytest.mark.asyncio
class TestTokenBucketLimiter:
    """Test token bucket rate limiting algorithm"""
    
    async def setup_method(self):
        """Setup Redis mock and token bucket limiter"""
        self.redis_mock = AsyncMock(spec=redis.Redis)
        self.limiter = TokenBucketLimiter(self.redis_mock)
        
        # Mock script loading
        self.redis_mock.script_load.return_value = "test_sha"
        await self.limiter.initialize()
    
    async def test_token_bucket_initialization(self):
        """Test token bucket limiter initialization"""
        assert self.limiter.script_sha == "test_sha"
        self.redis_mock.script_load.assert_called_once()
    
    async def test_successful_rate_limit_check(self):
        """Test successful rate limit check"""
        # Mock Redis response: [allowed, remaining, reset_time, retry_after, violations]
        self.redis_mock.evalsha.return_value = [1, 99, 1234567890.0, 0, 0]
        
        window = RateLimitWindow(duration_seconds=60, max_requests=100)
        policy = RateLimitPolicy(name="test_policy", scope=RateLimitScope.USER)
        
        result = await self.limiter.check_rate_limit(
            key="test_key",
            window=window,
            policy=policy
        )
        
        assert result.allowed is True
        assert result.remaining == 99
        assert result.reset_time == 1234567890.0
        assert result.retry_after is None
        assert result.algorithm == "token_bucket"
        assert result.policy_name == "test_policy"
    
    async def test_rate_limit_exceeded(self):
        """Test rate limit exceeded scenario"""
        # Mock Redis response: [denied, remaining, reset_time, retry_after, violations]
        self.redis_mock.evalsha.return_value = [0, 0, 1234567890.0, 30, 1]
        
        window = RateLimitWindow(duration_seconds=60, max_requests=100)
        policy = RateLimitPolicy(name="test_policy", scope=RateLimitScope.USER)
        
        result = await self.limiter.check_rate_limit(
            key="test_key",
            window=window,
            policy=policy
        )
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 30
        assert result.violation_count == 1
    
    async def test_redis_failure_fallback(self):
        """Test fallback behavior when Redis fails"""
        # Mock Redis failure
        self.redis_mock.evalsha.side_effect = Exception("Redis connection failed")
        
        window = RateLimitWindow(duration_seconds=60, max_requests=100)
        policy = RateLimitPolicy(name="test_policy", scope=RateLimitScope.USER)
        
        result = await self.limiter.check_rate_limit(
            key="test_key",
            window=window,
            policy=policy
        )
        
        # Should fail open
        assert result.allowed is True
        assert result.remaining == window.max_requests


@pytest.mark.asyncio
class TestSlidingWindowLimiter:
    """Test sliding window rate limiting algorithm"""
    
    async def setup_method(self):
        """Setup Redis mock and sliding window limiter"""
        self.redis_mock = AsyncMock(spec=redis.Redis)
        self.limiter = SlidingWindowLimiter(self.redis_mock)
        
        # Mock script loading
        self.redis_mock.script_load.return_value = "test_sha"
        await self.limiter.initialize()
    
    async def test_sliding_window_check(self):
        """Test sliding window rate limit check"""
        # Mock Redis response: [allowed, remaining, reset_time, retry_after, current_count]
        self.redis_mock.evalsha.return_value = [1, 49, 1234567890.0, 0, 50]
        
        window = RateLimitWindow(duration_seconds=3600, max_requests=100)
        policy = RateLimitPolicy(name="test_policy", scope=RateLimitScope.ENDPOINT)
        
        result = await self.limiter.check_rate_limit(
            key="test_key",
            window=window,
            policy=policy,
            request_id="req123"
        )
        
        assert result.allowed is True
        assert result.remaining == 49
        assert result.algorithm == "sliding_window"
    
    async def test_sliding_window_exceeded(self):
        """Test sliding window when limit exceeded"""
        # Mock Redis response: [denied, remaining, reset_time, retry_after, current_count]
        self.redis_mock.evalsha.return_value = [0, 0, 1234567890.0, 300, 100]
        
        window = RateLimitWindow(duration_seconds=3600, max_requests=100)
        policy = RateLimitPolicy(name="test_policy", scope=RateLimitScope.ENDPOINT)
        
        result = await self.limiter.check_rate_limit(
            key="test_key",
            window=window,
            policy=policy
        )
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 300


@pytest.mark.asyncio
class TestCircuitBreakerLimiter:
    """Test circuit breaker functionality"""
    
    async def setup_method(self):
        """Setup Redis mock and circuit breaker"""
        self.redis_mock = AsyncMock(spec=redis.Redis)
        self.circuit_breaker = CircuitBreakerLimiter(self.redis_mock)
        
        # Mock script loading
        self.redis_mock.script_load.return_value = "test_sha"
        await self.circuit_breaker.initialize()
    
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        # Mock Redis response: [allowed, state, request_count, threshold]
        self.redis_mock.evalsha.return_value = [1, "closed", 500, 1000]
        
        allowed, state = await self.circuit_breaker.check_circuit_breaker(
            threshold=1000,
            window_seconds=60
        )
        
        assert allowed is True
        assert state == "closed"
    
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state"""
        # Mock Redis response: [denied, state, request_count, threshold]
        self.redis_mock.evalsha.return_value = [0, "open", 1200, 1000]
        
        allowed, state = await self.circuit_breaker.check_circuit_breaker(
            threshold=1000,
            window_seconds=60
        )
        
        assert allowed is False
        assert state == "open"


@pytest.mark.asyncio
class TestAdaptiveRateLimiter:
    """Test adaptive rate limiter with reputation scoring"""
    
    async def setup_method(self):
        """Setup adaptive rate limiter with mocked components"""
        self.redis_mock = AsyncMock(spec=redis.Redis)
        self.limiter = AdaptiveRateLimiter(self.redis_mock)
        
        # Mock all component initializations
        with patch.object(self.limiter.token_bucket, 'initialize'), \
             patch.object(self.limiter.sliding_window, 'initialize'), \
             patch.object(self.limiter.circuit_breaker, 'initialize'):
            await self.limiter.initialize()
    
    async def test_adaptive_limiter_disabled_policy(self):
        """Test behavior with disabled policy"""
        policy = RateLimitPolicy(
            name="disabled_policy",
            scope=RateLimitScope.USER,
            mode=RateLimitMode.DISABLED,
            windows=[RateLimitWindow(duration_seconds=60, max_requests=100)]
        )
        
        results = await self.limiter.check_rate_limit("test_key", policy)
        
        assert len(results) == 1
        assert results[0].allowed is True
        assert results[0].algorithm == "disabled"
    
    async def test_adaptive_limiter_circuit_breaker_trip(self):
        """Test circuit breaker override"""
        # Mock circuit breaker in open state
        with patch.object(self.limiter.circuit_breaker, 'check_circuit_breaker', 
                         return_value=(False, "open")):
            
            policy = RateLimitPolicy(
                name="test_policy",
                scope=RateLimitScope.USER,
                windows=[RateLimitWindow(duration_seconds=60, max_requests=100)]
            )
            
            results = await self.limiter.check_rate_limit("test_key", policy)
            
            assert len(results) == 1
            assert results[0].allowed is False
            assert results[0].algorithm == "circuit_breaker"
    
    async def test_reputation_scoring(self):
        """Test reputation score management"""
        # Mock Redis for reputation storage
        self.redis_mock.hgetall.return_value = {
            b'score': b'0.8',
            b'violations': b'2',
            b'last_violation': str(time.time() - 3600).encode()
        }
        
        # Test getting reputation score
        score = await self.limiter._get_reputation_score("test_key")
        assert 0.7 <= score <= 0.9  # Should be around 0.8 with some decay
        
        # Test updating reputation after violation
        await self.limiter._update_reputation_score("test_key", violation=True)
        self.redis_mock.hset.assert_called()


@pytest.mark.asyncio
class TestRateLimitMiddleware:
    """Test rate limiting middleware integration"""
    
    async def setup_method(self):
        """Setup FastAPI app with rate limit middleware"""
        self.app = FastAPI()
        
        # Mock rate limiter and policy resolver
        self.rate_limiter_mock = AsyncMock(spec=AdaptiveRateLimiter)
        self.policy_resolver_mock = Mock(spec=PolicyResolver)
        
        # Setup default policy for testing
        self.test_policy = RateLimitPolicy(
            name="test_policy",
            scope=RateLimitScope.GLOBAL,
            windows=[RateLimitWindow(duration_seconds=60, max_requests=100)],
            mode=RateLimitMode.ENFORCE
        )
        self.policy_resolver_mock.resolve_policy.return_value = self.test_policy
        self.policy_resolver_mock.policies = {"test": self.test_policy}
        
        # Create middleware
        self.middleware = RateLimitMiddleware(
            app=self.app,
            rate_limiter=self.rate_limiter_mock,
            policy_resolver=self.policy_resolver_mock,
            enable_observability=False  # Disable for testing
        )
        
        # Add test endpoint
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        @self.app.get("/auth/login")
        async def auth_endpoint():
            return {"message": "auth"}
        
        # Add middleware to app
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=self.middleware.dispatch)
        self.client = TestClient(self.app)
    
    def test_bypass_paths(self):
        """Test that bypass paths are not rate limited"""
        # Mock to ensure rate limiter is not called
        response = self.client.get("/health")
        assert response.status_code == 404  # Endpoint doesn't exist, but should bypass rate limiting
        
        # Rate limiter should not have been called
        self.rate_limiter_mock.check_rate_limit.assert_not_called()
    
    async def test_successful_request(self):
        """Test successful rate limited request"""
        # Mock successful rate limit check
        self.rate_limiter_mock.check_rate_limit.return_value = [
            RateLimitResult(
                allowed=True,
                remaining=99,
                reset_time=time.time() + 60,
                retry_after=None,
                algorithm="token_bucket",
                policy_name="test_policy"
            )
        ]
        
        response = self.client.get("/test")
        assert response.status_code == 200
        assert "X-RateLimit-Policy" in response.headers
        assert response.headers["X-RateLimit-Policy"] == "test_policy"
        assert "X-RateLimit-Remaining" in response.headers
    
    async def test_rate_limit_exceeded(self):
        """Test rate limit exceeded response"""
        # Mock rate limit exceeded
        self.rate_limiter_mock.check_rate_limit.return_value = [
            RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=time.time() + 60,
                retry_after=30,
                algorithm="token_bucket",
                policy_name="test_policy"
            )
        ]
        
        response = self.client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "30"
        
        # Check response body
        data = response.json()
        assert data["error"] == "rate_limit_exceeded"
        assert "retry_after" in data
    
    async def test_shadow_mode(self):
        """Test shadow mode behavior"""
        # Set policy to shadow mode
        self.test_policy.mode = RateLimitMode.SHADOW
        
        # Mock rate limit exceeded
        self.rate_limiter_mock.check_rate_limit.return_value = [
            RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=time.time() + 60,
                retry_after=30,
                algorithm="token_bucket",
                policy_name="test_policy",
                violation_count=1
            )
        ]
        
        response = self.client.get("/test")
        # Should be allowed in shadow mode
        assert response.status_code == 200
    
    def test_client_ip_extraction(self):
        """Test client IP extraction with various headers"""
        # Test X-Forwarded-For header
        middleware = RateLimitMiddleware(app=None)
        
        # Mock request with X-Forwarded-For
        request_mock = Mock(spec=Request)
        request_mock.headers = {"X-Forwarded-For": "192.168.1.100, 10.0.0.1"}
        request_mock.client = None
        
        ip = middleware._get_client_ip(request_mock)
        assert ip == "192.168.1.100"
        
        # Test X-Real-IP header
        request_mock.headers = {"X-Real-IP": "192.168.1.200"}
        ip = middleware._get_client_ip(request_mock)
        assert ip == "192.168.1.200"
        
        # Test direct client IP
        request_mock.headers = {}
        request_mock.client = Mock()
        request_mock.client.host = "192.168.1.300"
        ip = middleware._get_client_ip(request_mock)
        assert ip == "192.168.1.300"
    
    def test_endpoint_normalization(self):
        """Test endpoint path normalization"""
        middleware = RateLimitMiddleware(app=None)
        
        # Test UUID replacement
        assert middleware._normalize_endpoint("/api/v1/users/12345678-1234-1234-1234-123456789012") == "/api/v1/users/{id}"
        
        # Test numeric ID replacement
        assert middleware._normalize_endpoint("/api/v1/posts/123") == "/api/v1/posts/{id}"
        
        # Test query parameter removal
        assert middleware._normalize_endpoint("/api/v1/search?q=test") == "/api/v1/search"
        
        # Test trailing slash removal
        assert middleware._normalize_endpoint("/api/v1/users/") == "/api/v1/users"


class TestDefaultPolicies:
    """Test default policy configurations"""
    
    def test_create_default_policies(self):
        """Test creation of default policies"""
        policies = create_default_policies()
        assert len(policies) > 0
        
        # Check that we have policies for different scopes
        scopes = {policy.scope for policy in policies}
        assert RateLimitScope.GLOBAL in scopes
        assert RateLimitScope.ENDPOINT in scopes
        assert RateLimitScope.ROLE in scopes
        
        # Check for critical endpoint policies
        endpoint_policies = [p for p in policies if p.scope == RateLimitScope.ENDPOINT]
        auth_policies = [p for p in endpoint_policies if "/api/v1/auth" in str(p.scope_values)]
        assert len(auth_policies) > 0
    
    def test_create_testing_policies(self):
        """Test creation of testing policies"""
        policies = create_testing_policies()
        assert len(policies) > 0
        
        # Testing policies should have shadow mode
        for policy in policies:
            assert policy.mode == RateLimitMode.SHADOW


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration tests for complex rate limiting scenarios"""
    
    async def setup_method(self):
        """Setup integration test environment"""
        # Use real Redis for integration tests (or mock if Redis not available)
        try:
            self.redis = redis.from_url("redis://localhost:6379/15")  # Test database
            await self.redis.ping()
            self.use_real_redis = True
        except:
            self.redis = AsyncMock(spec=redis.Redis)
            self.use_real_redis = False
        
        self.rate_limiter = AdaptiveRateLimiter(self.redis)
        
        if self.use_real_redis:
            await self.rate_limiter.initialize()
    
    async def teardown_method(self):
        """Cleanup after integration tests"""
        if self.use_real_redis:
            # Clear test data
            await self.redis.flushdb()
            await self.redis.close()
    
    @pytest.mark.skipif(True, reason="Requires Redis server")
    async def test_concurrent_requests(self):
        """Test concurrent request handling"""
        policy = RateLimitPolicy(
            name="concurrent_test",
            scope=RateLimitScope.USER,
            windows=[RateLimitWindow(duration_seconds=60, max_requests=10)]
        )
        
        # Send 15 concurrent requests
        async def make_request(i):
            return await self.rate_limiter.check_rate_limit(f"user:test", policy)
        
        tasks = [make_request(i) for i in range(15)]
        results = await asyncio.gather(*tasks)
        
        # Should have some allowed and some denied
        allowed_count = sum(1 for result_list in results for result in result_list if result.allowed)
        denied_count = sum(1 for result_list in results for result in result_list if not result.allowed)
        
        assert allowed_count <= 10  # Should not exceed limit
        assert denied_count >= 5    # Some should be denied
    
    @pytest.mark.skipif(True, reason="Requires Redis server")
    async def test_reputation_decay_over_time(self):
        """Test reputation score decay over time"""
        key = "reputation_test_user"
        
        # Create violations to lower reputation
        for _ in range(3):
            await self.rate_limiter._update_reputation_score(key, violation=True)
        
        # Check that reputation is lowered
        score = await self.rate_limiter._get_reputation_score(key)
        assert score < 1.0
        
        # Simulate time passing and good behavior
        for _ in range(10):
            await self.rate_limiter._update_reputation_score(key, violation=False)
        
        # Reputation should improve
        new_score = await self.rate_limiter._get_reputation_score(key)
        assert new_score > score


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])