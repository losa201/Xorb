"""
Circuit Breaker Pattern Implementation
Provides resilience against cascading failures in distributed systems
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    timeout_duration: int = 300  # seconds
    success_threshold: int = 3   # for half-open state
    monitor_window: int = 60     # seconds

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN to HALF_OPEN"""
        return (
            self.state == CircuitState.OPEN and 
            time.time() - self.last_failure_time >= self.config.timeout_duration
        )
    
    def _record_success(self):
        """Record a successful call"""
        with self._lock:
            self.total_calls += 1
            self.successful_calls += 1
            self.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def _record_failure(self):
        """Record a failed call"""
        with self._lock:
            self.total_calls += 1
            self.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker closed - service recovered")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info("Circuit breaker half-open - testing service")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            # Check if we should attempt reset
            if self._should_attempt_reset():
                self._transition_to_half_open()
            
            # Reject calls if circuit is open
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open. Last failure: {self.last_failure_time}"
                )
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            logger.error(f"Circuit breaker recorded failure: {e}")
            raise
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self.state
    
    def get_statistics(self) -> dict:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                'state': self.state.value,
                'total_calls': self.total_calls,
                'successful_calls': self.successful_calls,
                'failed_calls': self.failed_calls,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'failure_rate': self.failed_calls / max(self.total_calls, 1),
                'success_rate': self.successful_calls / max(self.total_calls, 1)
            }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info("Circuit breaker manually reset")
    
    def force_open(self):
        """Manually force circuit breaker to OPEN state"""
        with self._lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.warning("Circuit breaker manually opened")

class AsyncCircuitBreaker:
    """Async-optimized circuit breaker for high-concurrency scenarios"""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
        # Async synchronization
        self._lock = asyncio.Lock()
        
        # Monitoring
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        # Rate limiting for half-open state
        self._half_open_calls = 0
        self._max_half_open_calls = 1
    
    async def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN to HALF_OPEN"""
        return (
            self.state == CircuitState.OPEN and 
            time.time() - self.last_failure_time >= self.config.timeout_duration
        )
    
    async def _record_success(self):
        """Record a successful call"""
        async with self._lock:
            self.total_calls += 1
            self.successful_calls += 1
            self.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    async def _record_failure(self):
        """Record a failed call"""
        async with self._lock:
            self.total_calls += 1
            self.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    await self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()
    
    async def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self._half_open_calls = 0
        logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    async def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._half_open_calls = 0
        logger.info("Circuit breaker closed - service recovered")
    
    async def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self._half_open_calls = 0
        logger.info("Circuit breaker half-open - testing service")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            # Check if we should attempt reset
            if await self._should_attempt_reset():
                await self._transition_to_half_open()
            
            # Reject calls if circuit is open
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open. Last failure: {self.last_failure_time}"
                )
            
            # Limit concurrent calls in half-open state
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._max_half_open_calls:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is half-open and at capacity"
                    )
                self._half_open_calls += 1
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure()
            logger.error(f"Circuit breaker recorded failure: {e}")
            raise
        finally:
            # Decrement half-open call count
            if self.state == CircuitState.HALF_OPEN:
                async with self._lock:
                    self._half_open_calls = max(0, self._half_open_calls - 1)
    
    async def get_statistics(self) -> dict:
        """Get circuit breaker statistics"""
        async with self._lock:
            return {
                'state': self.state.value,
                'total_calls': self.total_calls,
                'successful_calls': self.successful_calls,
                'failed_calls': self.failed_calls,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'failure_rate': self.failed_calls / max(self.total_calls, 1),
                'success_rate': self.successful_calls / max(self.total_calls, 1),
                'half_open_calls': self._half_open_calls
            }

# Decorator for easy circuit breaker application
def circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker to functions"""
    def decorator(func: Callable):
        cb = CircuitBreaker(config)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await cb.call(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(cb.call(func, *args, **kwargs))
            return sync_wrapper
    
    return decorator

# Example usage:
# @circuit_breaker(CircuitBreakerConfig(failure_threshold=3, timeout_duration=60))
# async def external_api_call():
#     # Your API call logic here
#     pass