"""
Rate limiting management utility for operations and administration.

This utility provides:
- Policy management and configuration
- Emergency controls and kill-switch
- Monitoring and health checks
- Performance tuning and optimization
- Staged rollout and A/B testing support
"""

import asyncio
import json
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
import argparse
import sys

import redis.asyncio as redis
import structlog

from ..core.adaptive_rate_limiter import (
    AdaptiveRateLimiter, RateLimitPolicy, PolicyScope, LimitAlgorithm,
    EmergencyRateLimiter, ReputationLevel
)
from ..core.rate_limit_policies import (
    HierarchicalPolicyManager, RateLimitContext, PolicyOverride, PolicyType
)
from ..core.rate_limit_observability import RateLimitObservability

logger = structlog.get_logger("rate_limit_manager")


class RateLimitManager:
    """
    Comprehensive rate limiting management utility.
    
    Provides administrative functions for:
    - Policy management
    - Emergency controls
    - Monitoring and diagnostics
    - Performance optimization
    - Operational tasks
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.policy_manager = None
        self.rate_limiter = None
        self.emergency_limiter = None
        self.observability = None
        
        logger.info("Rate limit manager initialized", redis_url=redis_url)
    
    async def connect(self):
        """Connect to Redis and initialize components"""
        self.redis_client = redis.from_url(self.redis_url)
        
        try:
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise
        
        # Initialize components
        self.policy_manager = HierarchicalPolicyManager(self.redis_client)
        self.emergency_limiter = EmergencyRateLimiter(self.redis_client)
        self.observability = RateLimitObservability()
        
        # Load existing policies
        await self.policy_manager.load_policies_from_redis()
        
        logger.info("Rate limit manager components initialized")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    # Policy Management Commands
    
    async def list_policies(self) -> Dict[str, Any]:
        """List all rate limiting policies"""
        if not self.policy_manager:
            raise RuntimeError("Manager not connected")
        
        stats = self.policy_manager.get_stats()
        
        # Get global policies
        global_policies = {}
        for scope, policy in self.policy_manager.global_policies.items():
            global_policies[scope.value] = {
                "requests_per_second": policy.requests_per_second,
                "burst_size": policy.burst_size,
                "window_seconds": policy.window_seconds,
                "algorithm": policy.algorithm.value,
                "enabled": policy.enabled,
                "circuit_breaker_enabled": policy.circuit_breaker_enabled
            }
        
        return {
            "global_policies": global_policies,
            "tenant_overrides": stats["tenant_overrides"],
            "role_overrides": stats["role_overrides"],
            "endpoint_overrides": len(stats["endpoint_overrides"]),
            "emergency_overrides": stats["emergency_overrides"],
            "cache_size": stats["cache_size"]
        }
    
    async def create_tenant_override(
        self,
        tenant_id: str,
        scope: str,
        requests_per_second: float,
        burst_size: int,
        description: str = None
    ) -> bool:
        """Create tenant-specific rate limit override"""
        if not self.policy_manager:
            raise RuntimeError("Manager not connected")
        
        try:
            scope_enum = PolicyScope(scope)
            
            override = PolicyOverride(
                policy_type=PolicyType.TENANT_OVERRIDE,
                scope=scope_enum,
                identifier=tenant_id,
                requests_per_second=requests_per_second,
                burst_size=burst_size,
                description=description or f"Tenant override for {tenant_id}"
            )
            
            self.policy_manager.add_tenant_override(tenant_id, scope_enum, override)
            await self.policy_manager.save_policies_to_redis()
            
            logger.info(
                "Tenant override created",
                tenant_id=tenant_id,
                scope=scope,
                requests_per_second=requests_per_second,
                burst_size=burst_size
            )
            return True
        
        except Exception as e:
            logger.error("Failed to create tenant override", error=str(e))
            return False
    
    async def create_role_override(
        self,
        role_name: str,
        scope: str,
        requests_per_second: float,
        burst_size: int,
        description: str = None
    ) -> bool:
        """Create role-specific rate limit override"""
        if not self.policy_manager:
            raise RuntimeError("Manager not connected")
        
        try:
            scope_enum = PolicyScope(scope)
            
            override = PolicyOverride(
                policy_type=PolicyType.ROLE_OVERRIDE,
                scope=scope_enum,
                identifier=role_name,
                requests_per_second=requests_per_second,
                burst_size=burst_size,
                description=description or f"Role override for {role_name}"
            )
            
            self.policy_manager.add_role_override(role_name, scope_enum, override)
            await self.policy_manager.save_policies_to_redis()
            
            logger.info(
                "Role override created",
                role_name=role_name,
                scope=scope,
                requests_per_second=requests_per_second,
                burst_size=burst_size
            )
            return True
        
        except Exception as e:
            logger.error("Failed to create role override", error=str(e))
            return False
    
    async def remove_tenant_override(self, tenant_id: str, scope: str = None) -> bool:
        """Remove tenant-specific overrides"""
        if not self.policy_manager:
            raise RuntimeError("Manager not connected")
        
        try:
            if tenant_id in self.policy_manager.tenant_overrides:
                if scope:
                    # Remove specific scope override
                    scope_enum = PolicyScope(scope)
                    overrides = self.policy_manager.tenant_overrides[tenant_id]
                    self.policy_manager.tenant_overrides[tenant_id] = [
                        o for o in overrides if o.scope != scope_enum
                    ]
                else:
                    # Remove all overrides for tenant
                    del self.policy_manager.tenant_overrides[tenant_id]
                
                await self.policy_manager.save_policies_to_redis()
                self.policy_manager._clear_cache()
                
                logger.info("Tenant override removed", tenant_id=tenant_id, scope=scope)
                return True
            
            return False
        
        except Exception as e:
            logger.error("Failed to remove tenant override", error=str(e))
            return False
    
    # Emergency Controls
    
    async def activate_emergency_mode(self, duration_seconds: int = 300) -> bool:
        """Activate emergency rate limiting"""
        if not self.emergency_limiter:
            raise RuntimeError("Manager not connected")
        
        try:
            await self.emergency_limiter.activate_emergency_mode(duration_seconds)
            logger.critical("Emergency mode activated", duration_seconds=duration_seconds)
            return True
        
        except Exception as e:
            logger.error("Failed to activate emergency mode", error=str(e))
            return False
    
    async def deactivate_emergency_mode(self) -> bool:
        """Deactivate emergency rate limiting"""
        if not self.redis_client:
            raise RuntimeError("Manager not connected")
        
        try:
            await self.redis_client.delete("emergency:rate_limit")
            logger.critical("Emergency mode deactivated")
            return True
        
        except Exception as e:
            logger.error("Failed to deactivate emergency mode", error=str(e))
            return False
    
    async def activate_kill_switch(self) -> bool:
        """Activate emergency kill-switch (blocks all requests)"""
        if not self.emergency_limiter:
            raise RuntimeError("Manager not connected")
        
        try:
            await self.emergency_limiter.activate_kill_switch()
            logger.critical("Kill-switch activated - ALL REQUESTS WILL BE BLOCKED")
            return True
        
        except Exception as e:
            logger.error("Failed to activate kill-switch", error=str(e))
            return False
    
    async def deactivate_kill_switch(self) -> bool:
        """Deactivate emergency kill-switch"""
        if not self.emergency_limiter:
            raise RuntimeError("Manager not connected")
        
        try:
            await self.emergency_limiter.deactivate_kill_switch()
            logger.critical("Kill-switch deactivated")
            return True
        
        except Exception as e:
            logger.error("Failed to deactivate kill-switch", error=str(e))
            return False
    
    async def check_emergency_status(self) -> Dict[str, Any]:
        """Check emergency control status"""
        if not self.emergency_limiter:
            raise RuntimeError("Manager not connected")
        
        try:
            emergency_mode = await self.emergency_limiter.check_emergency_mode()
            kill_switch = await self.emergency_limiter.is_kill_switch_active()
            
            return {
                "emergency_mode_active": emergency_mode,
                "kill_switch_active": kill_switch,
                "timestamp": time.time()
            }
        
        except Exception as e:
            logger.error("Failed to check emergency status", error=str(e))
            return {"error": str(e)}
    
    # Monitoring and Diagnostics
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        if not self.redis_client:
            raise RuntimeError("Manager not connected")
        
        try:
            # Redis health
            redis_info = await self.redis_client.info()
            redis_health = {
                "connected": True,
                "memory_usage": redis_info.get("used_memory_human", "unknown"),
                "connected_clients": redis_info.get("connected_clients", 0),
                "keyspace_hits": redis_info.get("keyspace_hits", 0),
                "keyspace_misses": redis_info.get("keyspace_misses", 0)
            }
            
            # Policy manager health
            policy_health = self.policy_manager.get_stats() if self.policy_manager else {}
            
            # Rate limiter health
            limiter_health = {}
            if self.rate_limiter:
                limiter_health = await self.rate_limiter.get_stats()
            
            # Observability health
            observability_health = {}
            if self.observability:
                observability_health = self.observability.get_stats()
                observability_health["health_score"] = self.observability.get_health_score()
            
            return {
                "overall_status": "healthy",
                "timestamp": time.time(),
                "redis": redis_health,
                "policies": policy_health,
                "rate_limiter": limiter_health,
                "observability": observability_health
            }
        
        except Exception as e:
            logger.error("Failed to get health status", error=str(e))
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def get_rate_limit_stats(
        self,
        scope: str = None,
        tenant_id: str = None,
        time_window_minutes: int = 5
    ) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        if not self.redis_client:
            raise RuntimeError("Manager not connected")
        
        try:
            # Get Redis keys for rate limiting
            if scope:
                pattern = f"rl:{scope}:*"
            else:
                pattern = "rl:*"
            
            keys = await self.redis_client.keys(pattern)
            
            # Analyze key patterns
            scope_counts = {}
            algorithm_counts = {}
            
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    key_scope = parts[1]
                    algorithm = parts[2]
                    
                    scope_counts[key_scope] = scope_counts.get(key_scope, 0) + 1
                    algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
            
            # Get recent decisions from observability
            recent_stats = {}
            if self.observability:
                recent_stats = self.observability.get_stats()
            
            return {
                "time_window_minutes": time_window_minutes,
                "total_active_limiters": len(keys),
                "scope_distribution": scope_counts,
                "algorithm_distribution": algorithm_counts,
                "recent_decisions": recent_stats,
                "timestamp": time.time()
            }
        
        except Exception as e:
            logger.error("Failed to get rate limit stats", error=str(e))
            return {"error": str(e)}
    
    async def get_top_limited_ips(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top rate-limited IP addresses"""
        if not self.redis_client:
            raise RuntimeError("Manager not connected")
        
        try:
            # Get IP-based rate limiting keys
            ip_keys = await self.redis_client.keys("rl:ip:*")
            
            ip_stats = []
            for key in ip_keys:
                # Get token count or window data
                try:
                    data = await self.redis_client.hmget(key, "tokens", "last_refill")
                    if data[0] is not None:
                        tokens = float(data[0])
                        last_refill = float(data[1] or 0)
                        
                        # Extract IP hash from key
                        ip_hash = key.split(":")[-1]
                        
                        ip_stats.append({
                            "ip_hash": ip_hash,
                            "tokens_remaining": tokens,
                            "last_activity": last_refill,
                            "utilization": max(0, 100 - tokens)  # Rough utilization
                        })
                except:
                    continue
            
            # Sort by utilization (most limited first)
            ip_stats.sort(key=lambda x: x["utilization"], reverse=True)
            
            return ip_stats[:limit]
        
        except Exception as e:
            logger.error("Failed to get top limited IPs", error=str(e))
            return []
    
    # Performance Optimization
    
    async def optimize_policies(self) -> Dict[str, Any]:
        """Analyze and suggest policy optimizations"""
        if not self.observability:
            raise RuntimeError("Manager not connected")
        
        try:
            stats = self.observability.get_stats()
            
            recommendations = []
            
            # Analyze allow/block ratio
            if stats.get("total_events", 0) > 100:
                allow_rate = stats.get("allow_rate", 1.0)
                
                if allow_rate < 0.5:
                    recommendations.append({
                        "type": "high_block_rate",
                        "severity": "warning",
                        "message": f"High block rate ({allow_rate:.1%}). Consider relaxing limits.",
                        "suggestion": "Review and potentially increase rate limits"
                    })
                elif allow_rate > 0.95:
                    recommendations.append({
                        "type": "low_block_rate",
                        "severity": "info",
                        "message": f"Very low block rate ({allow_rate:.1%}). Limits may be too generous.",
                        "suggestion": "Consider tightening rate limits for better protection"
                    })
            
            # Analyze computation time
            avg_time = stats.get("avg_computation_time_ms", 0)
            if avg_time > 10:
                recommendations.append({
                    "type": "high_latency",
                    "severity": "warning",
                    "message": f"High computation time ({avg_time:.1f}ms).",
                    "suggestion": "Consider enabling local caching or optimizing Redis"
                })
            
            # Analyze token remaining distribution
            avg_tokens = stats.get("avg_tokens_remaining", 0)
            if avg_tokens < 10:
                recommendations.append({
                    "type": "low_tokens",
                    "severity": "info",
                    "message": f"Low average tokens remaining ({avg_tokens:.0f}).",
                    "suggestion": "Consider increasing burst sizes for better user experience"
                })
            
            return {
                "analysis_timestamp": time.time(),
                "sample_size": stats.get("total_events", 0),
                "recommendations": recommendations,
                "current_health_score": self.observability.get_health_score()
            }
        
        except Exception as e:
            logger.error("Failed to optimize policies", error=str(e))
            return {"error": str(e)}
    
    async def cleanup_expired_data(self) -> Dict[str, Any]:
        """Clean up expired rate limiting data"""
        if not self.redis_client:
            raise RuntimeError("Manager not connected")
        
        try:
            cleanup_stats = {
                "expired_keys_removed": 0,
                "backoff_keys_cleaned": 0,
                "circuit_breaker_keys_cleaned": 0,
                "policy_cache_cleared": False
            }
            
            # Clean expired backoff entries
            backoff_keys = await self.redis_client.keys("backoff:*")
            for key in backoff_keys:
                expires = await self.redis_client.hget(key, 'expires')
                if expires and time.time() > float(expires):
                    await self.redis_client.delete(key)
                    cleanup_stats["backoff_keys_cleaned"] += 1
            
            # Clean expired circuit breaker data
            cb_keys = await self.redis_client.keys("cb:*")
            for key in cb_keys:
                # Check if circuit breaker data is stale
                last_update = await self.redis_client.hget(key, 'window_start')
                if last_update and time.time() - float(last_update) > 3600:  # 1 hour
                    await self.redis_client.delete(key)
                    cleanup_stats["circuit_breaker_keys_cleaned"] += 1
            
            # Clear policy cache
            if self.policy_manager:
                self.policy_manager._clear_cache()
                cleanup_stats["policy_cache_cleared"] = True
            
            # Remove expired policy overrides
            if self.policy_manager:
                self.policy_manager.remove_expired_overrides()
            
            logger.info("Cleanup completed", **cleanup_stats)
            return cleanup_stats
        
        except Exception as e:
            logger.error("Failed to cleanup expired data", error=str(e))
            return {"error": str(e)}
    
    # Shadow Mode and Staged Rollout
    
    async def enable_shadow_mode(self, percentage: float = 100.0) -> bool:
        """Enable shadow mode for testing"""
        try:
            await self.redis_client.hset(
                "rate_limiter:config",
                "shadow_mode_enabled", "true",
                "shadow_mode_percentage", str(percentage)
            )
            
            logger.info("Shadow mode enabled", percentage=percentage)
            return True
        
        except Exception as e:
            logger.error("Failed to enable shadow mode", error=str(e))
            return False
    
    async def disable_shadow_mode(self) -> bool:
        """Disable shadow mode"""
        try:
            await self.redis_client.hset(
                "rate_limiter:config",
                "shadow_mode_enabled", "false"
            )
            
            logger.info("Shadow mode disabled")
            return True
        
        except Exception as e:
            logger.error("Failed to disable shadow mode", error=str(e))
            return False
    
    async def get_shadow_mode_status(self) -> Dict[str, Any]:
        """Get shadow mode status"""
        try:
            config = await self.redis_client.hmget(
                "rate_limiter:config",
                "shadow_mode_enabled", "shadow_mode_percentage"
            )
            
            enabled = config[0] == "true" if config[0] else False
            percentage = float(config[1]) if config[1] else 0.0
            
            return {
                "shadow_mode_enabled": enabled,
                "shadow_mode_percentage": percentage,
                "timestamp": time.time()
            }
        
        except Exception as e:
            logger.error("Failed to get shadow mode status", error=str(e))
            return {"error": str(e)}


# CLI Interface

async def main():
    """Command-line interface for rate limit management"""
    parser = argparse.ArgumentParser(description="Rate Limiting Management Utility")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Policy commands
    policy_parser = subparsers.add_parser("policy", help="Policy management")
    policy_subparsers = policy_parser.add_subparsers(dest="policy_action")
    
    policy_subparsers.add_parser("list", help="List all policies")
    
    tenant_parser = policy_subparsers.add_parser("create-tenant-override", help="Create tenant override")
    tenant_parser.add_argument("--tenant-id", required=True, help="Tenant ID")
    tenant_parser.add_argument("--scope", required=True, choices=["ip", "user", "tenant", "endpoint"])
    tenant_parser.add_argument("--requests-per-second", type=float, required=True)
    tenant_parser.add_argument("--burst-size", type=int, required=True)
    tenant_parser.add_argument("--description", help="Override description")
    
    role_parser = policy_subparsers.add_parser("create-role-override", help="Create role override")
    role_parser.add_argument("--role-name", required=True, help="Role name")
    role_parser.add_argument("--scope", required=True, choices=["ip", "user", "tenant", "endpoint"])
    role_parser.add_argument("--requests-per-second", type=float, required=True)
    role_parser.add_argument("--burst-size", type=int, required=True)
    role_parser.add_argument("--description", help="Override description")
    
    # Emergency commands
    emergency_parser = subparsers.add_parser("emergency", help="Emergency controls")
    emergency_subparsers = emergency_parser.add_subparsers(dest="emergency_action")
    
    emergency_subparsers.add_parser("status", help="Check emergency status")
    
    activate_emergency = emergency_subparsers.add_parser("activate", help="Activate emergency mode")
    activate_emergency.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    
    emergency_subparsers.add_parser("deactivate", help="Deactivate emergency mode")
    emergency_subparsers.add_parser("kill-switch-on", help="Activate kill-switch")
    emergency_subparsers.add_parser("kill-switch-off", help="Deactivate kill-switch")
    
    # Monitoring commands
    monitoring_parser = subparsers.add_parser("monitor", help="Monitoring and diagnostics")
    monitoring_subparsers = monitoring_parser.add_subparsers(dest="monitor_action")
    
    monitoring_subparsers.add_parser("health", help="Get health status")
    monitoring_subparsers.add_parser("stats", help="Get rate limiting statistics")
    monitoring_subparsers.add_parser("top-ips", help="Get top rate-limited IPs")
    monitoring_subparsers.add_parser("optimize", help="Get optimization recommendations")
    
    # Maintenance commands
    maintenance_parser = subparsers.add_parser("maintain", help="Maintenance operations")
    maintenance_subparsers = maintenance_parser.add_subparsers(dest="maintain_action")
    
    maintenance_subparsers.add_parser("cleanup", help="Clean up expired data")
    
    # Shadow mode commands
    shadow_parser = subparsers.add_parser("shadow", help="Shadow mode controls")
    shadow_subparsers = shadow_parser.add_subparsers(dest="shadow_action")
    
    shadow_subparsers.add_parser("status", help="Get shadow mode status")
    shadow_subparsers.add_parser("enable", help="Enable shadow mode")
    shadow_subparsers.add_parser("disable", help="Disable shadow mode")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = RateLimitManager(args.redis_url)
    
    try:
        await manager.connect()
        
        # Execute command
        if args.command == "policy":
            if args.policy_action == "list":
                result = await manager.list_policies()
                print(json.dumps(result, indent=2))
            
            elif args.policy_action == "create-tenant-override":
                success = await manager.create_tenant_override(
                    args.tenant_id, args.scope, args.requests_per_second,
                    args.burst_size, args.description
                )
                print(f"Tenant override {'created' if success else 'failed'}")
            
            elif args.policy_action == "create-role-override":
                success = await manager.create_role_override(
                    args.role_name, args.scope, args.requests_per_second,
                    args.burst_size, args.description
                )
                print(f"Role override {'created' if success else 'failed'}")
        
        elif args.command == "emergency":
            if args.emergency_action == "status":
                result = await manager.check_emergency_status()
                print(json.dumps(result, indent=2))
            
            elif args.emergency_action == "activate":
                success = await manager.activate_emergency_mode(args.duration)
                print(f"Emergency mode {'activated' if success else 'failed'}")
            
            elif args.emergency_action == "deactivate":
                success = await manager.deactivate_emergency_mode()
                print(f"Emergency mode {'deactivated' if success else 'failed'}")
            
            elif args.emergency_action == "kill-switch-on":
                success = await manager.activate_kill_switch()
                print(f"Kill-switch {'activated' if success else 'failed'}")
            
            elif args.emergency_action == "kill-switch-off":
                success = await manager.deactivate_kill_switch()
                print(f"Kill-switch {'deactivated' if success else 'failed'}")
        
        elif args.command == "monitor":
            if args.monitor_action == "health":
                result = await manager.get_health_status()
                print(json.dumps(result, indent=2))
            
            elif args.monitor_action == "stats":
                result = await manager.get_rate_limit_stats()
                print(json.dumps(result, indent=2))
            
            elif args.monitor_action == "top-ips":
                result = await manager.get_top_limited_ips()
                print(json.dumps(result, indent=2))
            
            elif args.monitor_action == "optimize":
                result = await manager.optimize_policies()
                print(json.dumps(result, indent=2))
        
        elif args.command == "maintain":
            if args.maintain_action == "cleanup":
                result = await manager.cleanup_expired_data()
                print(json.dumps(result, indent=2))
        
        elif args.command == "shadow":
            if args.shadow_action == "status":
                result = await manager.get_shadow_mode_status()
                print(json.dumps(result, indent=2))
            
            elif args.shadow_action == "enable":
                success = await manager.enable_shadow_mode()
                print(f"Shadow mode {'enabled' if success else 'failed'}")
            
            elif args.shadow_action == "disable":
                success = await manager.disable_shadow_mode()
                print(f"Shadow mode {'disabled' if success else 'failed'}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())