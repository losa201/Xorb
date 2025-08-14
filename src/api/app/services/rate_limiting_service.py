"""
Advanced rate limiting service with Redis backend
Implements production-ready rate limiting with multiple algorithms and tenant isolation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

from ..domain.repositories import CacheRepository
from ..domain.value_objects import RateLimitInfo, UsageStats
from .interfaces import RateLimitingService
from .base_service import XORBService, ServiceType


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    name: str
    algorithm: RateLimitAlgorithm
    requests_per_window: int
    window_size_seconds: int
    burst_allowance: int = 0
    tenant_specific: bool = False
    endpoint_pattern: Optional[str] = None
    user_role_restrictions: Optional[Dict[str, int]] = None


@dataclass
class RateLimitState:
    """Current rate limit state for a key"""
    key: str
    current_count: int
    window_start: float
    last_request: float
    tokens_available: float
    blocked_until: Optional[float] = None


class ProductionRateLimitingService(RateLimitingService, XORBService):
    """Production-ready rate limiting service with Redis backend"""

    def __init__(self, cache_repository: CacheRepository):
        super().__init__(service_type=ServiceType.SECURITY)
        self.cache = cache_repository
        self.logger = logging.getLogger(__name__)

        # Default rate limit rules
        self._default_rules = {
            "api_global": RateLimitRule(
                name="Global API Rate Limit",
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                requests_per_window=1000,
                window_size_seconds=3600,  # 1 hour
                burst_allowance=100
            ),
            "authentication": RateLimitRule(
                name="Authentication Rate Limit",
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                requests_per_window=10,
                window_size_seconds=900,  # 15 minutes
                burst_allowance=3
            ),
            "ptaas_scan": RateLimitRule(
                name="PTaaS Scan Rate Limit",
                algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
                requests_per_window=5,
                window_size_seconds=300,  # 5 minutes
                tenant_specific=True
            ),
            "intelligence_analysis": RateLimitRule(
                name="Intelligence Analysis Rate Limit",
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                requests_per_window=50,
                window_size_seconds=3600,  # 1 hour
                tenant_specific=True
            ),
            "admin_operations": RateLimitRule(
                name="Admin Operations Rate Limit",
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                requests_per_window=20,
                window_size_seconds=3600,  # 1 hour
                user_role_restrictions={"admin": 100, "security_analyst": 20, "viewer": 5}
            )
        }

        # Custom rules storage
        self._custom_rules: Dict[str, RateLimitRule] = {}

        # Usage tracking
        self._usage_stats: Dict[str, Dict[str, int]] = {}

    async def check_rate_limit(
        self,
        key: str,
        rule_name: str = "api_global",
        tenant_id: Optional[UUID] = None,
        user_role: Optional[str] = None
    ) -> RateLimitInfo:
        """Check if request is within rate limits"""
        try:
            # Get rate limit rule
            rule = self._get_rule(rule_name)
            if not rule:
                self.logger.warning(f"Rate limit rule '{rule_name}' not found")
                return RateLimitInfo(
                    allowed=True,
                    remaining=1000,
                    reset_time=time.time() + 3600,
                    retry_after=None
                )

            # Construct cache key
            cache_key = self._build_cache_key(key, rule_name, tenant_id)

            # Apply role-specific limits if configured
            if rule.user_role_restrictions and user_role:
                rule = self._apply_role_restrictions(rule, user_role)

            # Check rate limit based on algorithm
            if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return await self._check_token_bucket(cache_key, rule)
            elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return await self._check_sliding_window(cache_key, rule)
            elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                return await self._check_fixed_window(cache_key, rule)
            elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                return await self._check_leaky_bucket(cache_key, rule)
            else:
                self.logger.error(f"Unknown rate limit algorithm: {rule.algorithm}")
                return RateLimitInfo(allowed=False, remaining=0, reset_time=time.time() + 300)

        except Exception as e:
            self.logger.error(f"Error checking rate limit for key '{key}': {str(e)}")
            # Fail open for availability
            return RateLimitInfo(
                allowed=True,
                remaining=100,
                reset_time=time.time() + 3600,
                retry_after=None
            )

    async def increment_usage(
        self,
        key: str,
        rule_name: str = "api_global",
        tenant_id: Optional[UUID] = None,
        cost: int = 1
    ) -> bool:
        """Increment usage counter for rate limiting"""
        try:
            # Get current rate limit state
            rate_limit_info = await self.check_rate_limit(key, rule_name, tenant_id)

            if not rate_limit_info.allowed:
                return False

            # Record usage
            cache_key = self._build_cache_key(key, rule_name, tenant_id)
            await self._record_usage(cache_key, cost)

            # Update usage stats
            await self._update_usage_stats(key, rule_name, tenant_id, cost)

            return True

        except Exception as e:
            self.logger.error(f"Error incrementing usage for key '{key}': {str(e)}")
            return True  # Fail open

    async def get_usage_stats(
        self,
        key: str,
        tenant_id: Optional[UUID] = None,
        time_range_hours: int = 24
    ) -> UsageStats:
        """Get usage statistics for a key"""
        try:
            stats_key = f"usage_stats:{key}"
            if tenant_id:
                stats_key += f":{tenant_id}"

            # Get usage data from cache
            usage_data = await self.cache.get(stats_key) or {}

            # Calculate statistics
            current_time = time.time()
            cutoff_time = current_time - (time_range_hours * 3600)

            total_requests = 0
            requests_by_hour = {}

            for timestamp_str, count in usage_data.items():
                try:
                    timestamp = float(timestamp_str)
                    if timestamp >= cutoff_time:
                        total_requests += count
                        hour_key = int(timestamp // 3600) * 3600
                        requests_by_hour[hour_key] = requests_by_hour.get(hour_key, 0) + count
                except (ValueError, TypeError):
                    continue

            return UsageStats(
                total_requests=total_requests,
                requests_per_hour=list(requests_by_hour.values()),
                average_rate=total_requests / time_range_hours if time_range_hours > 0 else 0,
                peak_hour_requests=max(requests_by_hour.values()) if requests_by_hour else 0,
                time_range_hours=time_range_hours
            )

        except Exception as e:
            self.logger.error(f"Error getting usage stats for key '{key}': {str(e)}")
            return UsageStats(
                total_requests=0,
                requests_per_hour=[],
                average_rate=0,
                peak_hour_requests=0,
                time_range_hours=time_range_hours
            )

    async def add_custom_rule(self, rule_name: str, rule: RateLimitRule) -> bool:
        """Add custom rate limiting rule"""
        try:
            self._custom_rules[rule_name] = rule

            # Persist to cache
            rules_key = "rate_limit_custom_rules"
            all_rules = await self.cache.get(rules_key) or {}
            all_rules[rule_name] = asdict(rule)
            await self.cache.set(rules_key, all_rules, ttl=86400)  # 24 hours

            self.logger.info(f"Added custom rate limit rule: {rule_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding custom rule '{rule_name}': {str(e)}")
            return False

    async def remove_custom_rule(self, rule_name: str) -> bool:
        """Remove custom rate limiting rule"""
        try:
            if rule_name in self._custom_rules:
                del self._custom_rules[rule_name]

            # Remove from cache
            rules_key = "rate_limit_custom_rules"
            all_rules = await self.cache.get(rules_key) or {}
            if rule_name in all_rules:
                del all_rules[rule_name]
                await self.cache.set(rules_key, all_rules, ttl=86400)

            self.logger.info(f"Removed custom rate limit rule: {rule_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error removing custom rule '{rule_name}': {str(e)}")
            return False

    async def get_all_rules(self) -> Dict[str, RateLimitRule]:
        """Get all available rate limiting rules"""
        all_rules = dict(self._default_rules)
        all_rules.update(self._custom_rules)
        return all_rules

    async def reset_rate_limit(self, key: str, rule_name: str, tenant_id: Optional[UUID] = None) -> bool:
        """Reset rate limit for a specific key"""
        try:
            cache_key = self._build_cache_key(key, rule_name, tenant_id)
            await self.cache.delete(cache_key)

            self.logger.info(f"Reset rate limit for key: {cache_key}")
            return True

        except Exception as e:
            self.logger.error(f"Error resetting rate limit for key '{key}': {str(e)}")
            return False

    async def _check_token_bucket(self, cache_key: str, rule: RateLimitRule) -> RateLimitInfo:
        """Implement token bucket algorithm"""
        current_time = time.time()

        # Get current state
        state_data = await self.cache.get(cache_key)
        if state_data:
            state = RateLimitState(**state_data)
        else:
            state = RateLimitState(
                key=cache_key,
                current_count=0,
                window_start=current_time,
                last_request=current_time,
                tokens_available=rule.requests_per_window
            )

        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - state.last_request
        tokens_to_add = (time_elapsed / rule.window_size_seconds) * rule.requests_per_window
        state.tokens_available = min(
            rule.requests_per_window + rule.burst_allowance,
            state.tokens_available + tokens_to_add
        )

        # Check if request is allowed
        allowed = state.tokens_available >= 1.0

        if allowed:
            state.tokens_available -= 1.0
            state.current_count += 1

        state.last_request = current_time

        # Save state
        await self.cache.set(cache_key, asdict(state), ttl=rule.window_size_seconds * 2)

        return RateLimitInfo(
            allowed=allowed,
            remaining=int(state.tokens_available),
            reset_time=current_time + rule.window_size_seconds,
            retry_after=1.0 / (rule.requests_per_window / rule.window_size_seconds) if not allowed else None
        )

    async def _check_sliding_window(self, cache_key: str, rule: RateLimitRule) -> RateLimitInfo:
        """Implement sliding window algorithm"""
        current_time = time.time()
        window_start = current_time - rule.window_size_seconds

        # Get request timestamps
        timestamps_key = f"{cache_key}:timestamps"
        timestamps = await self.cache.get(timestamps_key) or []

        # Filter out old timestamps
        recent_timestamps = [ts for ts in timestamps if ts >= window_start]

        # Check if request is allowed
        allowed = len(recent_timestamps) < rule.requests_per_window

        if allowed:
            recent_timestamps.append(current_time)

        # Save timestamps
        await self.cache.set(timestamps_key, recent_timestamps, ttl=rule.window_size_seconds)

        return RateLimitInfo(
            allowed=allowed,
            remaining=max(0, rule.requests_per_window - len(recent_timestamps)),
            reset_time=min(recent_timestamps) + rule.window_size_seconds if recent_timestamps else current_time,
            retry_after=None if allowed else (min(recent_timestamps) + rule.window_size_seconds - current_time)
        )

    async def _check_fixed_window(self, cache_key: str, rule: RateLimitRule) -> RateLimitInfo:
        """Implement fixed window algorithm"""
        current_time = time.time()
        window_number = int(current_time // rule.window_size_seconds)
        window_key = f"{cache_key}:window:{window_number}"

        # Get current count for this window
        current_count = await self.cache.get(window_key) or 0

        # Check if request is allowed
        allowed = current_count < rule.requests_per_window

        if allowed:
            # Increment counter
            await self.cache.set(window_key, current_count + 1, ttl=rule.window_size_seconds)

        window_end = (window_number + 1) * rule.window_size_seconds

        return RateLimitInfo(
            allowed=allowed,
            remaining=max(0, rule.requests_per_window - current_count - (1 if allowed else 0)),
            reset_time=window_end,
            retry_after=None if allowed else (window_end - current_time)
        )

    async def _check_leaky_bucket(self, cache_key: str, rule: RateLimitRule) -> RateLimitInfo:
        """Implement leaky bucket algorithm"""
        current_time = time.time()

        # Get current state
        state_data = await self.cache.get(cache_key)
        if state_data:
            state = RateLimitState(**state_data)
        else:
            state = RateLimitState(
                key=cache_key,
                current_count=0,
                window_start=current_time,
                last_request=current_time,
                tokens_available=0
            )

        # Calculate leak rate
        leak_rate = rule.requests_per_window / rule.window_size_seconds
        time_elapsed = current_time - state.last_request
        leaked_tokens = time_elapsed * leak_rate

        # Update bucket state
        state.tokens_available = max(0, state.tokens_available - leaked_tokens)

        # Check if request is allowed
        allowed = state.tokens_available < rule.requests_per_window

        if allowed:
            state.tokens_available += 1
            state.current_count += 1

        state.last_request = current_time

        # Save state
        await self.cache.set(cache_key, asdict(state), ttl=rule.window_size_seconds * 2)

        return RateLimitInfo(
            allowed=allowed,
            remaining=max(0, rule.requests_per_window - int(state.tokens_available)),
            reset_time=current_time + (state.tokens_available / leak_rate),
            retry_after=1.0 / leak_rate if not allowed else None
        )

    def _get_rule(self, rule_name: str) -> Optional[RateLimitRule]:
        """Get rate limit rule by name"""
        if rule_name in self._custom_rules:
            return self._custom_rules[rule_name]
        return self._default_rules.get(rule_name)

    def _build_cache_key(self, key: str, rule_name: str, tenant_id: Optional[UUID] = None) -> str:
        """Build cache key for rate limiting"""
        parts = ["rate_limit", rule_name, key]
        if tenant_id:
            parts.append(str(tenant_id))
        return ":".join(parts)

    def _apply_role_restrictions(self, rule: RateLimitRule, user_role: str) -> RateLimitRule:
        """Apply role-specific restrictions to rate limit rule"""
        if not rule.user_role_restrictions or user_role not in rule.user_role_restrictions:
            return rule

        # Create modified rule with role-specific limits
        role_limit = rule.user_role_restrictions[user_role]
        return RateLimitRule(
            name=f"{rule.name} (Role: {user_role})",
            algorithm=rule.algorithm,
            requests_per_window=min(rule.requests_per_window, role_limit),
            window_size_seconds=rule.window_size_seconds,
            burst_allowance=min(rule.burst_allowance, role_limit // 10),
            tenant_specific=rule.tenant_specific,
            endpoint_pattern=rule.endpoint_pattern,
            user_role_restrictions=rule.user_role_restrictions
        )

    async def _record_usage(self, cache_key: str, cost: int) -> None:
        """Record usage for analytics"""
        usage_key = f"{cache_key}:usage"
        current_usage = await self.cache.get(usage_key) or 0
        await self.cache.set(usage_key, current_usage + cost, ttl=86400)  # 24 hours

    async def _update_usage_stats(self, key: str, rule_name: str, tenant_id: Optional[UUID], cost: int) -> None:
        """Update usage statistics"""
        stats_key = f"usage_stats:{key}"
        if tenant_id:
            stats_key += f":{tenant_id}"

        current_time = time.time()
        hour_timestamp = str(int(current_time // 3600) * 3600)

        usage_data = await self.cache.get(stats_key) or {}
        usage_data[hour_timestamp] = usage_data.get(hour_timestamp, 0) + cost

        # Keep only last 7 days of data
        cutoff_time = current_time - (7 * 24 * 3600)
        usage_data = {ts: count for ts, count in usage_data.items()
                     if float(ts) >= cutoff_time}

        await self.cache.set(stats_key, usage_data, ttl=7 * 24 * 3600)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test cache connectivity
            test_key = "rate_limit_health_check"
            await self.cache.set(test_key, "ok", ttl=60)
            cache_result = await self.cache.get(test_key)
            await self.cache.delete(test_key)

            cache_healthy = cache_result == "ok"

            return {
                "status": "healthy" if cache_healthy else "degraded",
                "cache_connection": cache_healthy,
                "default_rules": len(self._default_rules),
                "custom_rules": len(self._custom_rules),
                "active_algorithms": len(RateLimitAlgorithm),
                "timestamp": str(current_time)
            }

        except Exception as e:
            self.logger.error(f"Rate limiting service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": str(time.time())
            }


def asdict(obj) -> Dict[str, Any]:
    """Convert dataclass to dictionary"""
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    return obj.__dict__ if hasattr(obj, '__dict__') else str(obj)
