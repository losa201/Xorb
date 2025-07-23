"""
Token Manager for LLM Usage Control
Manages token budgets and rate limiting for external LLM services
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

import redis.asyncio as redis

logger = logging.getLogger(__name__)

class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"

@dataclass
class TokenBudget:
    """Token budget configuration for different priorities"""
    daily_limit: int
    hourly_limit: int
    burst_limit: int
    current_usage: int = 0
    reset_time: float = field(default_factory=time.time)

@dataclass
class TokenUsage:
    """Token usage record"""
    timestamp: float
    tokens: int
    priority: Priority
    provider: str
    model: str
    cost_usd: float

class TokenManager:
    """Manages LLM token budgets and rate limiting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Redis connection for distributed state
        redis_host = config.get('redis_host', 'redis.xorb-prod.svc.cluster.local')
        redis_port = config.get('redis_port', 6379)
        redis_db = config.get('redis_db', 0)
        
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        # Budget configuration
        self.budgets = self._initialize_budgets(config.get('budgets', {}))
        
        # Rate limiting windows
        self.rate_limit_windows = {
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }
        
        # Emergency reserve
        self.emergency_reserve = config.get('emergency_reserve', 5000)
        self.emergency_threshold = config.get('emergency_threshold', 0.9)
        
    def _initialize_budgets(self, budget_config: Dict[str, Any]) -> Dict[Priority, TokenBudget]:
        """Initialize token budgets for different priorities"""
        default_budgets = {
            Priority.CRITICAL: TokenBudget(
                daily_limit=20000,
                hourly_limit=5000,
                burst_limit=1000
            ),
            Priority.HIGH: TokenBudget(
                daily_limit=30000,
                hourly_limit=3000,
                burst_limit=500
            ),
            Priority.STANDARD: TokenBudget(
                daily_limit=50000,
                hourly_limit=2000,
                burst_limit=300
            ),
            Priority.LOW: TokenBudget(
                daily_limit=10000,
                hourly_limit=500,
                burst_limit=100
            )
        }
        
        # Override with config values
        for priority, config_budget in budget_config.items():
            if priority in [p.value for p in Priority]:
                priority_enum = Priority(priority)
                if priority_enum in default_budgets:
                    budget = default_budgets[priority_enum]
                    budget.daily_limit = config_budget.get('daily_limit', budget.daily_limit)
                    budget.hourly_limit = config_budget.get('hourly_limit', budget.hourly_limit)
                    budget.burst_limit = config_budget.get('burst_limit', budget.burst_limit)
        
        return default_budgets
    
    async def can_allocate(self, priority: str, tokens: int) -> bool:
        """Check if tokens can be allocated for given priority"""
        try:
            priority_enum = Priority(priority)
            budget = self.budgets[priority_enum]
            
            # Check daily limit
            daily_usage = await self._get_usage('day', priority)
            if daily_usage + tokens > budget.daily_limit:
                logger.warning(f"Daily limit exceeded for {priority}: {daily_usage + tokens} > {budget.daily_limit}")
                
                # Check emergency reserve for critical requests
                if priority_enum == Priority.CRITICAL:
                    emergency_usage = await self._get_emergency_usage()
                    if emergency_usage + tokens <= self.emergency_reserve:
                        logger.info(f"Using emergency reserve for critical request: {tokens} tokens")
                        return True
                
                return False
            
            # Check hourly limit
            hourly_usage = await self._get_usage('hour', priority)
            if hourly_usage + tokens > budget.hourly_limit:
                logger.warning(f"Hourly limit exceeded for {priority}: {hourly_usage + tokens} > {budget.hourly_limit}")
                return False
            
            # Check burst limit (per minute)
            minute_usage = await self._get_usage('minute', priority)
            if minute_usage + tokens > budget.burst_limit:
                logger.warning(f"Burst limit exceeded for {priority}: {minute_usage + tokens} > {budget.burst_limit}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking token allocation: {e}")
            return False
    
    async def reserve_tokens(self, priority: str, tokens: int) -> str:
        """Reserve tokens for use"""
        try:
            if not await self.can_allocate(priority, tokens):
                raise ValueError(f"Cannot allocate {tokens} tokens for {priority}")
            
            # Generate reservation ID
            reservation_id = f"reservation_{int(time.time() * 1000)}_{priority}_{tokens}"
            
            # Store reservation in Redis with expiration
            reservation_data = {
                'priority': priority,
                'tokens': tokens,
                'timestamp': time.time(),
                'status': 'reserved'
            }
            
            await self.redis_client.setex(
                f"token_reservation:{reservation_id}",
                300,  # 5 minute expiration
                json.dumps(reservation_data)
            )
            
            logger.debug(f"Reserved {tokens} tokens for {priority}: {reservation_id}")
            return reservation_id
            
        except Exception as e:
            logger.error(f"Error reserving tokens: {e}")
            raise
    
    async def consume_tokens(self, priority: str, tokens: int, 
                           provider: str = "unknown", model: str = "unknown",
                           cost_usd: float = 0.0, reservation_id: Optional[str] = None):
        """Consume tokens from budget"""
        try:
            current_time = time.time()
            
            # Verify reservation if provided
            if reservation_id:
                reservation_key = f"token_reservation:{reservation_id}"
                reservation_data = await self.redis_client.get(reservation_key)
                
                if not reservation_data:
                    raise ValueError(f"Invalid or expired reservation: {reservation_id}")
                
                reservation = json.loads(reservation_data)
                if reservation['tokens'] != tokens or reservation['priority'] != priority:
                    raise ValueError("Reservation mismatch")
                
                # Mark reservation as consumed
                reservation['status'] = 'consumed'
                await self.redis_client.setex(reservation_key, 60, json.dumps(reservation))
            
            # Record usage in time-windowed keys
            for window, duration in self.rate_limit_windows.items():
                window_key = self._get_window_key(window, priority, current_time)
                
                # Use pipeline for atomic operations
                pipe = self.redis_client.pipeline()
                pipe.incrby(window_key, tokens)
                pipe.expire(window_key, duration)
                await pipe.execute()
            
            # Record detailed usage
            usage_record = TokenUsage(
                timestamp=current_time,
                tokens=tokens,
                priority=Priority(priority),
                provider=provider,
                model=model,
                cost_usd=cost_usd
            )
            
            await self._record_usage(usage_record)
            
            logger.debug(f"Consumed {tokens} tokens for {priority} from {provider}/{model}")
            
        except Exception as e:
            logger.error(f"Error consuming tokens: {e}")
            raise
    
    async def _get_usage(self, window: str, priority: str) -> int:
        """Get current usage for time window and priority"""
        try:
            window_key = self._get_window_key(window, priority, time.time())
            usage = await self.redis_client.get(window_key)
            return int(usage) if usage else 0
        except Exception as e:
            logger.error(f"Error getting usage: {e}")
            return 0
    
    async def _get_emergency_usage(self) -> int:
        """Get current emergency reserve usage"""
        try:
            emergency_key = f"emergency_usage:{datetime.now().strftime('%Y-%m-%d')}"
            usage = await self.redis_client.get(emergency_key) 
            return int(usage) if usage else 0
        except Exception as e:
            logger.error(f"Error getting emergency usage: {e}")
            return 0
    
    def _get_window_key(self, window: str, priority: str, timestamp: float) -> str:
        """Generate Redis key for time window"""
        dt = datetime.fromtimestamp(timestamp)
        
        if window == 'minute':
            time_str = dt.strftime('%Y-%m-%d:%H:%M')
        elif window == 'hour':
            time_str = dt.strftime('%Y-%m-%d:%H')
        else:  # day
            time_str = dt.strftime('%Y-%m-%d')
        
        return f"token_usage:{window}:{priority}:{time_str}"
    
    async def _record_usage(self, usage: TokenUsage):
        """Record detailed usage information"""
        try:
            # Store usage record
            usage_key = f"usage_detail:{int(usage.timestamp * 1000)}"
            usage_data = {
                'timestamp': usage.timestamp,
                'tokens': usage.tokens,
                'priority': usage.priority.value,
                'provider': usage.provider,
                'model': usage.model,
                'cost_usd': usage.cost_usd
            }
            
            await self.redis_client.setex(
                usage_key,
                86400 * 7,  # 7 days retention
                json.dumps(usage_data)
            )
            
            # Update aggregate statistics
            stats_key = f"token_stats:{datetime.now().strftime('%Y-%m-%d')}"
            pipe = self.redis_client.pipeline()
            pipe.hincrby(stats_key, f"total_tokens", usage.tokens)
            pipe.hincrby(stats_key, f"total_requests", 1)
            pipe.hincrby(stats_key, f"tokens_{usage.priority.value}", usage.tokens)
            pipe.hincrby(stats_key, f"tokens_{usage.provider}", usage.tokens)
            pipe.hincrbyfloat(stats_key, f"total_cost", usage.cost_usd)
            pipe.expire(stats_key, 86400 * 30)  # 30 days retention
            await pipe.execute()
            
        except Exception as e:
            logger.error(f"Error recording usage: {e}")
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        try:
            stats = {}
            current_time = time.time()
            
            # Get usage by priority and time window
            for priority in Priority:
                stats[priority.value] = {}
                for window in self.rate_limit_windows.keys():
                    usage = await self._get_usage(window, priority.value)
                    limit = getattr(self.budgets[priority], f"{window}ly_limit" if window != "minute" else "burst_limit")
                    
                    stats[priority.value][window] = {
                        'usage': usage,
                        'limit': limit,
                        'remaining': max(0, limit - usage),
                        'utilization': usage / limit if limit > 0 else 0
                    }
            
            # Get daily aggregate stats
            today = datetime.now().strftime('%Y-%m-%d')
            daily_stats_key = f"token_stats:{today}"
            daily_stats = await self.redis_client.hgetall(daily_stats_key)
            
            stats['daily_aggregates'] = {
                'total_tokens': int(daily_stats.get('total_tokens', 0)),
                'total_requests': int(daily_stats.get('total_requests', 0)),
                'total_cost': float(daily_stats.get('total_cost', 0.0)),
                'emergency_usage': await self._get_emergency_usage()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {}
    
    async def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get cost breakdown by provider and model"""
        try:
            cost_breakdown = {
                'by_provider': {},
                'by_model': {},
                'by_priority': {},
                'total_cost': 0.0
            }
            
            # Get recent usage records
            today = datetime.now().strftime('%Y-%m-%d')
            daily_stats_key = f"token_stats:{today}"
            daily_stats = await self.redis_client.hgetall(daily_stats_key)
            
            total_cost = float(daily_stats.get('total_cost', 0.0))
            cost_breakdown['total_cost'] = total_cost
            
            # Get cost by priority (approximated from token usage)
            for priority in Priority:
                tokens = int(daily_stats.get(f"tokens_{priority.value}", 0))
                # Rough cost estimation (actual costs vary by provider)
                estimated_cost = tokens * 0.0001  # $0.0001 per token estimate
                cost_breakdown['by_priority'][priority.value] = estimated_cost
            
            return cost_breakdown
            
        except Exception as e:
            logger.error(f"Error getting cost breakdown: {e}")
            return {}
    
    async def adjust_budget(self, priority: str, budget_changes: Dict[str, int]):
        """Dynamically adjust token budgets"""
        try:
            priority_enum = Priority(priority)
            budget = self.budgets[priority_enum]
            
            for limit_type, new_value in budget_changes.items():
                if hasattr(budget, f"{limit_type}_limit"):
                    setattr(budget, f"{limit_type}_limit", new_value)
                    logger.info(f"Adjusted {limit_type}_limit for {priority} to {new_value}")
            
            # Store updated budgets in Redis for persistence
            budget_key = f"token_budget:{priority}"
            budget_data = {
                'daily_limit': budget.daily_limit,
                'hourly_limit': budget.hourly_limit,
                'burst_limit': budget.burst_limit
            }
            await self.redis_client.set(budget_key, json.dumps(budget_data))
            
        except Exception as e:
            logger.error(f"Error adjusting budget: {e}")
            raise
    
    async def get_budget_recommendations(self) -> Dict[str, Any]:
        """Get budget adjustment recommendations based on usage patterns"""
        try:
            recommendations = {}
            
            for priority in Priority:
                stats = {}
                
                # Get usage patterns for last 7 days
                usage_history = []
                for i in range(7):
                    date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                    daily_key = f"token_stats:{date}"
                    daily_usage = await self.redis_client.hget(daily_key, f"tokens_{priority.value}")
                    usage_history.append(int(daily_usage) if daily_usage else 0)
                
                if usage_history:
                    avg_usage = sum(usage_history) / len(usage_history)
                    max_usage = max(usage_history)
                    current_limit = self.budgets[priority].daily_limit
                    
                    # Recommend adjustments
                    if avg_usage > current_limit * 0.8:
                        recommended_limit = int(max_usage * 1.2)  # 20% buffer
                        recommendations[priority.value] = {
                            'action': 'increase',
                            'current_limit': current_limit,
                            'recommended_limit': recommended_limit,
                            'reason': f'Average usage ({avg_usage:.0f}) approaching limit'
                        }
                    elif avg_usage < current_limit * 0.3:
                        recommended_limit = int(avg_usage * 2)  # 100% buffer
                        recommendations[priority.value] = {
                            'action': 'decrease',
                            'current_limit': current_limit,
                            'recommended_limit': recommended_limit,
                            'reason': f'Low usage ({avg_usage:.0f}) suggests over-allocation'
                        }
                    else:
                        recommendations[priority.value] = {
                            'action': 'maintain',
                            'current_limit': current_limit,
                            'reason': 'Usage within optimal range'
                        }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting budget recommendations: {e}")
            return {}
    
    async def cleanup_expired_data(self):
        """Clean up expired reservations and old usage data"""
        try:
            # Clean up expired reservations
            pattern = "token_reservation:*"
            async for key in self.redis_client.scan_iter(match=pattern):
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    await self.redis_client.delete(key)
            
            # Clean up old usage details (older than 7 days)
            cutoff_timestamp = time.time() - (7 * 86400)
            pattern = "usage_detail:*"
            deleted_count = 0
            
            async for key in self.redis_client.scan_iter(match=pattern):
                timestamp_str = key.split(':')[1]
                try:
                    timestamp = int(timestamp_str) / 1000
                    if timestamp < cutoff_timestamp:
                        await self.redis_client.delete(key)
                        deleted_count += 1
                except ValueError:
                    continue
            
            logger.info(f"Cleaned up {deleted_count} expired usage records")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired data: {e}")
    
    async def close(self):
        """Close Redis connections"""
        try:
            await self.redis_client.close()
            logger.info("Token manager connections closed")
        except Exception as e:
            logger.error(f"Error closing token manager: {e}")