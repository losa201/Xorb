"""
Comprehensive error handling with circuit breakers and resilience patterns
"""

import asyncio
import time
import traceback
from typing import Any, Dict, List, Optional, Callable, TypeVar, Type
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import defaultdict, deque
import json

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import structlog

from .logging import get_logger, security_logger
from .metrics import get_metrics_service

logger = get_logger(__name__)
T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    SECURITY = "security"
    RATE_LIMIT = "rate_limit"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorDetails:
    """Detailed error information"""
    error_id: str
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    request_path: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    retry_after: Optional[int] = None
    user_message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response format"""
    error: bool = True
    error_id: str
    error_type: str
    message: str
    user_message: Optional[str] = None
    timestamp: str
    correlation_id: Optional[str] = None
    retry_after: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    monitoring_window_seconds: int = 300
    max_failures_in_window: int = 10


class CircuitBreaker:
    """Circuit breaker implementation for external service calls"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0
        
        # Sliding window for monitoring
        self.failure_times: deque = deque(maxlen=config.max_failures_in_window)
        
        self.metrics_service = get_metrics_service()
        logger.info("Circuit breaker initialized", name=name, config=config.__dict__)
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            if time.time() < self.next_attempt_time:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. Next attempt at {self.next_attempt_time}"
                )
            else:
                # Transition to half-open
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to half-open", name=self.name)
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                # Reset circuit breaker
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker reset to closed", name=self.name)
        
        # Record metrics
        if self.metrics_service:
            self.metrics_service.custom_metrics.increment_counter(
                "circuit_breaker_success",
                1,
                {"circuit_breaker": self.name, "state": self.state.value}
            )
    
    async def _on_failure(self, error: Exception):
        """Handle failed call"""
        current_time = time.time()
        self.failure_count += 1
        self.last_failure_time = current_time
        self.failure_times.append(current_time)
        
        # Check if we should open the circuit
        if self.state == CircuitBreakerState.CLOSED:
            # Check failure threshold
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
            
            # Check sliding window
            elif self._should_open_based_on_window():
                self._open_circuit()
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Single failure in half-open state opens the circuit
            self._open_circuit()
        
        # Record metrics
        if self.metrics_service:
            self.metrics_service.custom_metrics.increment_counter(
                "circuit_breaker_failure",
                1,
                {"circuit_breaker": self.name, "state": self.state.value, "error_type": type(error).__name__}
            )
        
        logger.warning("Circuit breaker failure recorded",
                      name=self.name,
                      failure_count=self.failure_count,
                      state=self.state.value,
                      error=str(error))
    
    def _should_open_based_on_window(self) -> bool:
        """Check if circuit should open based on sliding window"""
        current_time = time.time()
        window_start = current_time - self.config.monitoring_window_seconds
        
        # Count failures in the window
        failures_in_window = sum(
            1 for failure_time in self.failure_times
            if failure_time >= window_start
        )
        
        return failures_in_window >= self.config.max_failures_in_window
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state = CircuitBreakerState.OPEN
        self.next_attempt_time = time.time() + self.config.timeout_seconds
        
        logger.error("Circuit breaker opened",
                    name=self.name,
                    failure_count=self.failure_count,
                    next_attempt=self.next_attempt_time)
        
        # Record metrics
        if self.metrics_service:
            self.metrics_service.custom_metrics.increment_counter(
                "circuit_breaker_opened",
                1,
                {"circuit_breaker": self.name}
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": self.next_attempt_time,
            "failures_in_window": len(self.failure_times)
        }


class RetryConfig:
    """Configuration for retry mechanisms"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError, TimeoutError, HTTPException
        ]


class RetryMechanism:
    """Intelligent retry mechanism with exponential backoff"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.metrics_service = get_metrics_service()
    
    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        operation_name: str = "unknown",
        **kwargs
    ) -> T:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Record successful retry metrics
                if attempt > 0 and self.metrics_service:
                    self.metrics_service.custom_metrics.increment_counter(
                        "retry_success",
                        1,
                        {"operation": operation_name, "attempt": str(attempt + 1)}
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.warning("Non-retryable exception", 
                                 operation=operation_name,
                                 attempt=attempt + 1,
                                 error=str(e))
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning("Retrying operation",
                             operation=operation_name,
                             attempt=attempt + 1,
                             max_attempts=self.config.max_attempts,
                             delay=delay,
                             error=str(e))
                
                # Record retry metrics
                if self.metrics_service:
                    self.metrics_service.custom_metrics.increment_counter(
                        "retry_attempt",
                        1,
                        {"operation": operation_name, "attempt": str(attempt + 1)}
                    )
                
                await asyncio.sleep(delay)
        
        # All retries failed
        if self.metrics_service:
            self.metrics_service.custom_metrics.increment_counter(
                "retry_exhausted",
                1,
                {"operation": operation_name}
            )
        
        logger.error("All retry attempts failed",
                    operation=operation_name,
                    max_attempts=self.config.max_attempts,
                    final_error=str(last_exception))
        
        raise last_exception
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        if self.config.exponential_backoff:
            delay = self.config.base_delay * (2 ** attempt)
        else:
            delay = self.config.base_delay
        
        # Apply max delay
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter to avoid thundering herd
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


class RateLimitError(HTTPException):
    """Rate limit exceeded error"""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )


class CircuitBreakerOpenError(Exception):
    """Circuit breaker is open"""
    pass


class ValidationError(HTTPException):
    """Validation error"""
    
    def __init__(self, message: str, field: Optional[str] = None):
        detail = {"message": message}
        if field:
            detail["field"] = field
        
        super().__init__(status_code=422, detail=detail)


class BusinessLogicError(HTTPException):
    """Business logic error"""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        detail = {"message": message}
        if error_code:
            detail["error_code"] = error_code
        
        super().__init__(status_code=400, detail=detail)


class ExternalServiceError(Exception):
    """External service error"""
    
    def __init__(self, service: str, message: str, status_code: Optional[int] = None):
        self.service = service
        self.status_code = status_code
        super().__init__(message)


class ErrorHandler:
    """Central error handling and tracking"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recent_errors: deque = deque(maxlen=1000)
        self.metrics_service = get_metrics_service()
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker"""
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def register_retry_config(self, operation: str, config: RetryConfig):
        """Register retry configuration for operation"""
        self.retry_configs[operation] = config
    
    def get_retry_mechanism(self, operation: str) -> RetryMechanism:
        """Get retry mechanism for operation"""
        config = self.retry_configs.get(operation, RetryConfig())
        return RetryMechanism(config)
    
    async def handle_error(
        self,
        error: Exception,
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorDetails:
        """Handle and categorize error"""
        import uuid
        
        # Generate error ID
        error_id = str(uuid.uuid4())
        
        # Categorize error
        category = self._categorize_error(error)
        severity = self._assess_severity(error, category)
        
        # Extract correlation context
        correlation_id = getattr(request.state, 'correlation_id', None) if request else None
        user_id = getattr(request.state, 'user_id', None) if request else None
        request_path = request.url.path if request else None
        
        # Create error details
        error_details = ErrorDetails(
            error_id=error_id,
            error_type=type(error).__name__,
            message=str(error),
            category=category,
            severity=severity,
            timestamp=time.time(),
            correlation_id=correlation_id,
            user_id=user_id,
            request_path=request_path,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # Add user-friendly message
        error_details.user_message = self._get_user_message(error, category)
        
        # Handle retry-after for rate limits
        if isinstance(error, RateLimitError):
            error_details.retry_after = int(error.headers.get("Retry-After", 60))
        
        # Track error
        self._track_error(error_details)
        
        # Log error based on severity
        self._log_error(error_details)
        
        return error_details
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error by type"""
        if isinstance(error, (ValidationError, ValueError)):
            return ErrorCategory.VALIDATION
        elif isinstance(error, RateLimitError):
            return ErrorCategory.RATE_LIMIT
        elif isinstance(error, BusinessLogicError):
            return ErrorCategory.BUSINESS_LOGIC
        elif isinstance(error, ExternalServiceError):
            return ErrorCategory.EXTERNAL_SERVICE
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, PermissionError):
            return ErrorCategory.AUTHORIZATION
        elif "auth" in str(error).lower():
            return ErrorCategory.AUTHENTICATION
        elif "database" in str(error).lower() or "sql" in str(error).lower():
            return ErrorCategory.DATABASE
        elif "security" in str(error).lower():
            return ErrorCategory.SECURITY
        else:
            return ErrorCategory.SYSTEM
    
    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity"""
        if category == ErrorCategory.SECURITY:
            return ErrorSeverity.CRITICAL
        elif category in [ErrorCategory.DATABASE, ErrorCategory.EXTERNAL_SERVICE]:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.BUSINESS_LOGIC, ErrorCategory.NETWORK]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_user_message(self, error: Exception, category: ErrorCategory) -> str:
        """Get user-friendly error message"""
        if category == ErrorCategory.VALIDATION:
            return "Please check your input and try again."
        elif category == ErrorCategory.RATE_LIMIT:
            return "Too many requests. Please try again later."
        elif category == ErrorCategory.AUTHENTICATION:
            return "Authentication failed. Please check your credentials."
        elif category == ErrorCategory.AUTHORIZATION:
            return "You don't have permission to perform this action."
        elif category == ErrorCategory.EXTERNAL_SERVICE:
            return "External service is temporarily unavailable. Please try again later."
        elif category == ErrorCategory.DATABASE:
            return "Database error occurred. Please try again later."
        else:
            return "An unexpected error occurred. Please try again later."
    
    def _track_error(self, error_details: ErrorDetails):
        """Track error for monitoring and alerting"""
        # Count errors by type
        self.error_counts[error_details.error_type] += 1
        
        # Add to recent errors
        self.recent_errors.append({
            "error_id": error_details.error_id,
            "error_type": error_details.error_type,
            "category": error_details.category.value,
            "severity": error_details.severity.value,
            "timestamp": error_details.timestamp,
            "message": error_details.message[:200]  # Truncate long messages
        })
        
        # Record metrics
        if self.metrics_service:
            self.metrics_service.custom_metrics.increment_counter(
                "errors_total",
                1,
                {
                    "error_type": error_details.error_type,
                    "category": error_details.category.value,
                    "severity": error_details.severity.value
                }
            )
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error based on severity"""
        log_data = {
            "error_id": error_details.error_id,
            "error_type": error_details.error_type,
            "category": error_details.category.value,
            "message": error_details.message,
            "correlation_id": error_details.correlation_id,
            "user_id": error_details.user_id,
            "request_path": error_details.request_path
        }
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error", **log_data)
            security_logger.critical("Critical error detected", **log_data)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error("High severity error", **log_data)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error", **log_data)
        else:
            logger.info("Low severity error", **log_data)
    
    def create_error_response(self, error_details: ErrorDetails) -> ErrorResponse:
        """Create standardized error response"""
        return ErrorResponse(
            error_id=error_details.error_id,
            error_type=error_details.error_type,
            message=error_details.message,
            user_message=error_details.user_message,
            timestamp=str(error_details.timestamp),
            correlation_id=error_details.correlation_id,
            retry_after=error_details.retry_after
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        recent_count = len(self.recent_errors)
        error_rate = recent_count / 1000.0 if recent_count > 0 else 0.0
        
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "recent_errors_count": recent_count,
            "error_rate": error_rate,
            "circuit_breakers": {
                name: cb.get_status()
                for name, cb in self.circuit_breakers.items()
            }
        }


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if not _error_handler:
        _error_handler = ErrorHandler()
    return _error_handler


# Decorators for error handling
def with_circuit_breaker(circuit_breaker_name: str):
    """Decorator to add circuit breaker protection"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            error_handler = get_error_handler()
            circuit_breaker = error_handler.get_circuit_breaker(circuit_breaker_name)
            
            if not circuit_breaker:
                # Create default circuit breaker
                config = CircuitBreakerConfig()
                circuit_breaker = error_handler.register_circuit_breaker(circuit_breaker_name, config)
            
            return await circuit_breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # For sync functions, create async wrapper
            async def async_func():
                return func(*args, **kwargs)
            
            return asyncio.run(async_wrapper())
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def with_retry(operation_name: str, config: Optional[RetryConfig] = None):
    """Decorator to add retry logic"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            error_handler = get_error_handler()
            
            if config:
                error_handler.register_retry_config(operation_name, config)
            
            retry_mechanism = error_handler.get_retry_mechanism(operation_name)
            return await retry_mechanism.execute_with_retry(func, *args, operation_name=operation_name, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # For sync functions, use basic retry without async
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for FastAPI"""
    error_handler = get_error_handler()
    error_details = await error_handler.handle_error(exc, request)
    error_response = error_handler.create_error_response(error_details)
    
    # Determine HTTP status code
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
    elif error_details.category == ErrorCategory.VALIDATION:
        status_code = 422
    elif error_details.category == ErrorCategory.AUTHENTICATION:
        status_code = 401
    elif error_details.category == ErrorCategory.AUTHORIZATION:
        status_code = 403
    elif error_details.category == ErrorCategory.RATE_LIMIT:
        status_code = 429
    else:
        status_code = 500
    
    headers = {}
    if error_details.retry_after:
        headers["Retry-After"] = str(error_details.retry_after)
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.dict(),
        headers=headers
    )