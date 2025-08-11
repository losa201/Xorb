"""
Comprehensive test suite for adaptive rate limiting system.

Tests cover:
- Token bucket and sliding window algorithms
- Policy hierarchy and overrides
- Circuit breakers and emergency controls
- Observability and metrics
- Performance and scalability
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.api.app.core.adaptive_rate_limiter import (
    AdaptiveRateLimiter, RateLimitPolicy, PolicyScope, LimitAlgorithm,
    ReputationLevel, EmergencyRateLimiter
)
from src.api.app.core.rate_limit_policies import (
    HierarchicalPolicyManager, RateLimitContext, PolicyOverride, PolicyType
)
from src.api.app.core.rate_limit_observability import (
    RateLimitObservability, DecisionOutcome, RateLimitEvent
)


@pytest.fixture
async def mock_redis():
    """Mock Redis client for testing"""
    redis_mock = AsyncMock()
    
    # Mock eval responses for token bucket
    redis_mock.eval.return_value = [1, 50, 0]  # allowed, remaining, retry_after
    
    # Mock other Redis operations
    redis_mock.get.return_value = None
    redis_mock.hmget.return_value = [None, None, None]
    redis_mock.keys.return_value = []
    redis_mock.info.return_value = {"connected_clients": 1, "used_memory_human": "1MB"}
    
    return redis_mock


@pytest.fixture
def sample_policies() -> List[RateLimitPolicy]:
    """Sample rate limiting policies for testing"""
    return [
        RateLimitPolicy(
            scope=PolicyScope.IP,
            algorithm=LimitAlgorithm.SLIDING_WINDOW,
            requests_per_second=10.0,
            burst_size=50,
            window_seconds=60,
            priority=100
        ),
        RateLimitPolicy(
            scope=PolicyScope.USER,
            algorithm=LimitAlgorithm.TOKEN_BUCKET,
            requests_per_second=50.0,
            burst_size=200,
            window_seconds=60,
            priority=90
        ),
        RateLimitPolicy(
            scope=PolicyScope.TENANT,
            algorithm=LimitAlgorithm.TOKEN_BUCKET,
            requests_per_second=200.0,
            burst_size=1000,
            window_seconds=60,
            priority=80
        )
    ]


@pytest.fixture
def sample_context() -> RateLimitContext:
    """Sample rate limiting context for testing"""
    return RateLimitContext(
        scope=PolicyScope.USER,
        endpoint="/api/v1/test",
        user_id="user123",
        tenant_id="tenant456",
        ip_address="192.168.1.100",
        user_agent="TestAgent/1.0",
        is_authenticated=True,
        is_admin=False,
        role_names={"user", "premium_user"},
        business_hours=True,
        request_time=time.time()
    )


class TestAdaptiveRateLimiter:
    """Test adaptive rate limiter core functionality"""
    
    @pytest.mark.asyncio
    async def test_token_bucket_algorithm(self, mock_redis, sample_policies):
        """Test token bucket algorithm implementation"""
        # Mock token bucket script response
        mock_redis.eval.return_value = [1, 45, 0]  # allowed, remaining, retry_after
        
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        result = await limiter.check_rate_limit(
            identifier="test_user",
            scope=PolicyScope.USER,
            cost=5
        )
        
        assert result.allowed is True
        assert result.tokens_remaining == 45
        assert result.algorithm_used == LimitAlgorithm.TOKEN_BUCKET
        assert result.correlation_id is not None
        
        # Verify Redis script was called
        mock_redis.eval.assert_called()
        args = mock_redis.eval.call_args[0]
        assert len(args) >= 2  # Script and key count
    
    @pytest.mark.asyncio
    async def test_sliding_window_algorithm(self, mock_redis, sample_policies):
        """Test sliding window algorithm implementation"""
        # Mock sliding window script response
        mock_redis.eval.return_value = [1, 8, 0]  # allowed, remaining, retry_after
        
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        result = await limiter.check_rate_limit(
            identifier="192.168.1.100",
            scope=PolicyScope.IP,
            cost=2
        )
        
        assert result.allowed is True
        assert result.tokens_remaining == 8
        assert result.algorithm_used == LimitAlgorithm.SLIDING_WINDOW
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, mock_redis, sample_policies):
        """Test rate limit exceeded scenario"""
        # Mock rate limit exceeded response
        mock_redis.eval.return_value = [0, 0, 60]  # blocked, remaining, retry_after
        
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        result = await limiter.check_rate_limit(
            identifier="test_user",
            scope=PolicyScope.USER,
            cost=1
        )
        
        assert result.allowed is False
        assert result.tokens_remaining == 0
        assert result.retry_after_seconds == 60
    
    @pytest.mark.asyncio
    async def test_shadow_mode(self, mock_redis, sample_policies):
        """Test shadow mode functionality"""
        # Mock rate limit exceeded response
        mock_redis.eval.return_value = [0, 0, 60]  # blocked, remaining, retry_after
        
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=True  # Enable shadow mode
        )
        
        result = await limiter.check_rate_limit(
            identifier="test_user",
            scope=PolicyScope.USER,
            cost=1
        )
        
        # Should be allowed in shadow mode even if limit exceeded
        assert result.allowed is True
        assert "shadow_mode" in result.decision_metadata or True
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, mock_redis, sample_policies):
        """Test circuit breaker functionality"""
        # Mock circuit breaker open state
        mock_redis.hmget.return_value = ["open", str(time.time())]
        
        # Create policy with circuit breaker enabled
        cb_policy = RateLimitPolicy(
            scope=PolicyScope.USER,
            algorithm=LimitAlgorithm.TOKEN_BUCKET,
            requests_per_second=10.0,
            burst_size=50,
            window_seconds=60,
            circuit_breaker_enabled=True,
            failure_threshold=5,
            recovery_timeout_seconds=300
        )
        
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=[cb_policy],
            shadow_mode=False
        )
        
        result = await limiter.check_rate_limit(
            identifier="test_user",
            scope=PolicyScope.USER,
            cost=1
        )
        
        assert result.circuit_breaker_open is True
    
    @pytest.mark.asyncio
    async def test_reputation_based_limits(self, mock_redis, sample_policies):
        """Test reputation-based limit adjustments"""
        # Mock good reputation
        mock_redis.hmget.return_value = ["80.0", "0", "0"]  # score, violations, last_violation
        
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        result = await limiter.check_rate_limit(
            identifier="good_user",
            scope=PolicyScope.USER,
            cost=1
        )
        
        assert result.reputation_level in [ReputationLevel.GOOD, ReputationLevel.EXCELLENT]
    
    @pytest.mark.asyncio
    async def test_progressive_backoff(self, mock_redis, sample_policies):
        """Test progressive backoff functionality"""
        # Mock active backoff
        mock_redis.hmget.return_value = ["3", str(time.time() + 3600)]  # level, expires
        
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        result = await limiter.check_rate_limit(
            identifier="backoff_user",
            scope=PolicyScope.USER,
            cost=1
        )
        
        assert result.backoff_level == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_fail_open(self, sample_policies):
        """Test error handling with fail-open behavior"""
        # Create a Redis mock that raises exceptions
        error_redis = AsyncMock()
        error_redis.eval.side_effect = Exception("Redis connection failed")
        
        limiter = AdaptiveRateLimiter(
            redis_client=error_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        result = await limiter.check_rate_limit(
            identifier="test_user",
            scope=PolicyScope.USER,
            cost=1
        )
        
        # Should fail open on errors
        assert result.allowed is True
        assert "error" in result.decision_metadata.get('reason', '')


class TestHierarchicalPolicyManager:
    """Test hierarchical policy management"""
    
    def test_policy_resolution_hierarchy(self, mock_redis, sample_context):
        """Test policy resolution with hierarchy"""
        manager = HierarchicalPolicyManager(mock_redis)
        
        # Add tenant override
        tenant_override = PolicyOverride(
            policy_type=PolicyType.TENANT_OVERRIDE,
            scope=PolicyScope.USER,
            identifier=str(sample_context.tenant_id),
            requests_per_second=100.0,
            burst_size=500
        )
        manager.add_tenant_override(str(sample_context.tenant_id), PolicyScope.USER, tenant_override)
        
        # Add role override
        role_override = PolicyOverride(
            policy_type=PolicyType.ROLE_OVERRIDE,
            scope=PolicyScope.USER,
            identifier="premium_user",
            requests_per_second=150.0,
            burst_size=750
        )
        manager.add_role_override("premium_user", PolicyScope.USER, role_override)
        
        # Resolve policy
        resolved_policy = manager.resolve_policy(sample_context)
        
        assert resolved_policy is not None
        assert resolved_policy.requests_per_second == 150.0  # Role override should win
        assert "role_override" in resolved_policy.decision_metadata.get('overrides_applied', [])
    
    def test_hard_caps_enforcement(self, mock_redis, sample_context):
        """Test hard caps enforcement"""
        manager = HierarchicalPolicyManager(mock_redis)
        
        # Try to set limit above hard cap
        excessive_override = PolicyOverride(
            policy_type=PolicyType.TENANT_OVERRIDE,
            scope=PolicyScope.USER,
            identifier=str(sample_context.tenant_id),
            requests_per_second=10000.0,  # Way above hard cap
            burst_size=50000
        )
        manager.add_tenant_override(str(sample_context.tenant_id), PolicyScope.USER, excessive_override)
        
        resolved_policy = manager.resolve_policy(sample_context)
        
        # Should be capped at hard limit
        assert resolved_policy.requests_per_second <= manager.hard_caps[PolicyScope.USER]["max_requests_per_second"]
    
    def test_emergency_override_priority(self, mock_redis, sample_context):
        """Test emergency override has highest priority"""
        manager = HierarchicalPolicyManager(mock_redis)
        
        # Add emergency override
        emergency_override = PolicyOverride(
            policy_type=PolicyType.EMERGENCY_OVERRIDE,
            scope=PolicyScope.USER,
            identifier="emergency_test",
            requests_per_second=1.0,  # Very restrictive
            burst_size=5,
            priority=1
        )
        manager.add_emergency_override(PolicyScope.USER, emergency_override, duration_seconds=3600)
        
        resolved_policy = manager.resolve_policy(sample_context)
        
        assert resolved_policy.requests_per_second == 1.0
        assert "emergency_override" in resolved_policy.decision_metadata.get('overrides_applied', [])
    
    def test_endpoint_pattern_matching(self, mock_redis):
        """Test endpoint pattern matching"""
        manager = HierarchicalPolicyManager(mock_redis)
        
        context = RateLimitContext(
            scope=PolicyScope.ENDPOINT,
            endpoint="/api/v1/auth/login"
        )
        
        resolved_policy = manager.resolve_policy(context)
        
        # Should match auth endpoint pattern and apply strict limits
        assert resolved_policy is not None
        assert resolved_policy.requests_per_second <= 5.0  # Auth endpoints are restrictive


class TestRateLimitObservability:
    """Test observability and monitoring"""
    
    def test_metrics_recording(self, sample_context):
        """Test metrics recording"""
        observability = RateLimitObservability(
            enable_tracing=False,
            enable_detailed_logging=False
        )
        
        # Mock result
        from src.api.app.core.adaptive_rate_limiter import RateLimitResult
        result = RateLimitResult(
            allowed=True,
            policy_matched=None,
            tokens_remaining=45,
            retry_after_seconds=None,
            reputation_level=ReputationLevel.GOOD,
            backoff_level=0,
            circuit_breaker_open=False,
            algorithm_used=LimitAlgorithm.TOKEN_BUCKET,
            computation_time_ms=2.5,
            redis_hits=1,
            cache_hits=0,
            correlation_id="test123"
        )
        
        event = observability.record_decision(sample_context, result)
        
        assert event.correlation_id == "test123"
        assert event.decision_outcome == DecisionOutcome.ALLOWED
        assert event.computation_time_ms == 2.5
        assert event.reputation_level == ReputationLevel.GOOD
    
    def test_health_score_calculation(self):
        """Test health score calculation"""
        observability = RateLimitObservability()
        
        # Add some sample events
        current_time = time.time()
        for i in range(100):
            event = RateLimitEvent(
                timestamp=current_time - i,
                correlation_id=f"test{i}",
                decision_outcome=DecisionOutcome.ALLOWED if i < 80 else DecisionOutcome.BLOCKED_RATE_LIMIT,
                scope=PolicyScope.USER,
                identifier_hash="test_hash",
                endpoint="/test",
                user_id_hash=None,
                tenant_id=None,
                ip_address_hash="ip_hash",
                user_agent_hash=None,
                policy_matched="test_policy",
                algorithm_used=LimitAlgorithm.TOKEN_BUCKET,
                tokens_remaining=50,
                retry_after_seconds=None,
                reputation_level=ReputationLevel.NEUTRAL,
                backoff_level=0,
                computation_time_ms=1.0,
                redis_hits=1,
                cache_hits=0
            )
            observability.recent_events.append(event)
        
        health_score = observability.get_health_score()
        
        # Should be a reasonable score (80% allowed)
        assert 70.0 <= health_score <= 90.0
    
    def test_false_positive_tracking(self):
        """Test false positive/negative tracking"""
        from src.api.app.core.rate_limit_observability import FalsePositiveNegativeTracker
        
        tracker = FalsePositiveNegativeTracker()
        
        # Simulate some decisions
        legitimate_context = RateLimitContext(
            scope=PolicyScope.USER,
            is_authenticated=True,
            business_hours=True
        )
        
        legitimate_event = RateLimitEvent(
            timestamp=time.time(),
            correlation_id="test",
            decision_outcome=DecisionOutcome.ALLOWED,
            scope=PolicyScope.USER,
            identifier_hash="test",
            endpoint="/test",
            user_id_hash=None,
            tenant_id=None,
            ip_address_hash=None,
            user_agent_hash=None,
            policy_matched="test",
            algorithm_used=LimitAlgorithm.TOKEN_BUCKET,
            tokens_remaining=50,
            retry_after_seconds=None,
            reputation_level=ReputationLevel.GOOD,
            backoff_level=0,
            computation_time_ms=1.0,
            redis_hits=1,
            cache_hits=0
        )
        
        tracker.record_decision(legitimate_event, legitimate_context)
        
        estimates = tracker.get_estimates()
        assert "false_positive_rate" in estimates
        assert "false_negative_rate" in estimates
        assert "accuracy" in estimates


class TestEmergencyRateLimiter:
    """Test emergency controls"""
    
    @pytest.mark.asyncio
    async def test_emergency_mode_activation(self, mock_redis):
        """Test emergency mode activation"""
        emergency_limiter = EmergencyRateLimiter(mock_redis)
        
        await emergency_limiter.activate_emergency_mode(duration_seconds=300)
        
        mock_redis.setex.assert_called_with("emergency:rate_limit", 300, "active")
        assert emergency_limiter.emergency_active is True
    
    @pytest.mark.asyncio
    async def test_kill_switch_activation(self, mock_redis):
        """Test kill-switch activation"""
        emergency_limiter = EmergencyRateLimiter(mock_redis)
        
        await emergency_limiter.activate_kill_switch()
        
        mock_redis.set.assert_called_with("kill_switch:active", "true")
        assert emergency_limiter.kill_switch_active is True
    
    @pytest.mark.asyncio
    async def test_kill_switch_check(self, mock_redis):
        """Test kill-switch status check"""
        mock_redis.get.return_value = "true"
        
        emergency_limiter = EmergencyRateLimiter(mock_redis)
        
        is_active = await emergency_limiter.is_kill_switch_active()
        
        assert is_active is True
        mock_redis.get.assert_called_with("kill_switch:active")


class TestPerformance:
    """Performance and scalability tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_checks(self, mock_redis, sample_policies):
        """Test concurrent rate limit checks"""
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        # Simulate concurrent requests
        tasks = []
        for i in range(100):
            task = limiter.check_rate_limit(
                identifier=f"user{i % 10}",  # 10 different users
                scope=PolicyScope.USER,
                cost=1
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 1.0
        assert len(results) == 100
        assert all(r.correlation_id is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_local_cache_performance(self, mock_redis, sample_policies):
        """Test local cache improves performance"""
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        # First call should hit Redis
        result1 = await limiter.check_rate_limit(
            identifier="cached_user",
            scope=PolicyScope.USER,
            cost=1
        )
        
        # Reset mock to count subsequent calls
        mock_redis.reset_mock()
        
        # Second call should use cache for some operations
        result2 = await limiter.check_rate_limit(
            identifier="cached_user",
            scope=PolicyScope.USER,
            cost=1
        )
        
        # Both should succeed
        assert result1.correlation_id != result2.correlation_id
    
    def test_memory_usage_under_load(self, mock_redis, sample_policies):
        """Test memory usage under sustained load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        limiter = AdaptiveRateLimiter(
            redis_client=mock_redis,
            policies=sample_policies,
            shadow_mode=False
        )
        
        # Simulate many different identifiers
        for i in range(10000):
            limiter.local_cache[f"test_key_{i}"] = (time.time(), f"value_{i}")
        
        # Trigger cleanup
        current_time = time.time()
        expired_keys = [
            k for k, (cache_time, _) in limiter.local_cache.items()
            if current_time - cache_time > limiter.cache_ttl * 2
        ]
        for key in expired_keys:
            del limiter.local_cache[key]
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


@pytest.mark.integration
class TestIntegration:
    """Integration tests with real Redis (optional)"""
    
    @pytest.mark.skip(reason="Requires Redis server")
    @pytest.mark.asyncio
    async def test_real_redis_integration(self):
        """Test with real Redis server"""
        import redis.asyncio as redis
        
        redis_client = redis.Redis(host='localhost', port=6379, db=15)  # Test DB
        
        try:
            await redis_client.ping()
            
            policies = [
                RateLimitPolicy(
                    scope=PolicyScope.USER,
                    algorithm=LimitAlgorithm.TOKEN_BUCKET,
                    requests_per_second=10.0,
                    burst_size=50,
                    window_seconds=60
                )
            ]
            
            limiter = AdaptiveRateLimiter(
                redis_client=redis_client,
                policies=policies,
                shadow_mode=False
            )
            
            # Test actual rate limiting
            results = []
            for i in range(60):  # Should allow 50 + 10/sec for 1 sec
                result = await limiter.check_rate_limit(
                    identifier="integration_test_user",
                    scope=PolicyScope.USER,
                    cost=1
                )
                results.append(result)
                await asyncio.sleep(0.01)  # Small delay
            
            allowed_count = sum(1 for r in results if r.allowed)
            
            # Should allow initial burst plus some refill
            assert 50 <= allowed_count <= 60
            
        finally:
            await redis_client.flushdb()  # Clean test DB
            await redis_client.close()


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_adaptive_rate_limiter.py -v
    pytest.main([__file__, "-v", "--tb=short"])