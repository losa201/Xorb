#!/usr/bin/env python3
"""
XORB Enhanced Error Handling & Recovery Framework
Comprehensive error management, recovery, and resilience system
"""

import asyncio
import json
import logging
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import hashlib

# Enhanced logging configuration
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM_RESOURCE = "system_resource"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"

@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    error_id: str
    timestamp: datetime
    service_name: str
    function_name: str
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    stack_trace: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    system_context: Dict[str, Any] = field(default_factory=dict)
    business_context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class RecoveryAction:
    """Recovery action definition"""
    action_id: str
    name: str
    strategy: RecoveryStrategy
    handler: Callable
    max_attempts: int = 3
    backoff_strategy: str = "exponential"
    timeout_seconds: int = 30
    conditions: Dict[str, Any] = field(default_factory=dict)
    fallback_action: Optional[str] = None

@dataclass
class CircuitBreakerState:
    """Enhanced circuit breaker state"""
    service_name: str
    state: str  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    half_open_max_calls: int = 3
    current_half_open_calls: int = 0

class XORBErrorHandler:
    """Central error handling and recovery system"""
    
    def __init__(self, service_name: str, config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.config = config or {}
        self.error_history: List[ErrorContext] = []
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.error_patterns: Dict[str, int] = {}
        self.active_degradations: Dict[str, datetime] = {}
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize metrics
        self.metrics = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "circuit_breaker_trips": 0,
            "degradation_events": 0
        }
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info(f"XORB Error Handler initialized for service: {service_name}")

    def _setup_logging(self):
        """Setup enhanced logging configuration"""
        log_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[%(service_name)s] - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Create logger
        self.logger = logging.getLogger(f"xorb.{self.service_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        
        # File handler for errors
        error_handler = logging.FileHandler(f'logs/xorb_errors_{self.service_name}.log')
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(log_format)
        error_handler.setFormatter(error_formatter)
        
        # Add service name to log context
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.service_name = self.service_name
            return record
        logging.setLogRecordFactory(record_factory)
        
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(error_handler)

    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks"""
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"xorb-error-{self.service_name}")
        
        # Start error pattern analysis
        self.executor.submit(self._pattern_analysis_loop)
        
        # Start circuit breaker monitoring
        self.executor.submit(self._circuit_breaker_monitor_loop)

    def register_recovery_action(self, action: RecoveryAction):
        """Register a recovery action"""
        self.recovery_actions[action.action_id] = action
        self.logger.info(f"Registered recovery action: {action.name}")

    def handle_error(self, 
                    error: Exception, 
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    user_id: Optional[str] = None,
                    request_id: Optional[str] = None) -> ErrorContext:
        """Handle an error with comprehensive logging and recovery"""
        
        # Create error context
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            service_name=self.service_name,
            function_name=self._get_calling_function(),
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id,
            system_context=self._collect_system_context(),
            business_context=context or {}
        )
        
        # Store error history
        self.error_history.append(error_context)
        self._update_metrics(error_context)
        
        # Log error with appropriate level
        log_level = self._map_severity_to_log_level(severity)
        self.logger.log(
            log_level,
            f"Error {error_context.error_id}: {error_context.error_message}",
            extra={
                "error_id": error_context.error_id,
                "category": category.value,
                "severity": severity.value,
                "context": context
            }
        )
        
        # Attempt recovery
        self._attempt_recovery(error_context)
        
        # Check for circuit breaker activation
        self._check_circuit_breaker(error_context)
        
        # Analyze error patterns
        self._analyze_error_pattern(error_context)
        
        return error_context

    def _get_calling_function(self) -> str:
        """Get the name of the function that called handle_error"""
        try:
            frame = sys._getframe(2)  # Go up 2 frames to get the actual caller
            return frame.f_code.co_name
        except:
            return "unknown"

    def _collect_system_context(self) -> Dict[str, Any]:
        """Collect current system context"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "process_id": os.getpid(),
                "thread_count": threading.active_count(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception:
            return {"collection_error": True}

    def _map_severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Map error severity to logging level"""
        mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.ERROR)

    def _update_metrics(self, error_context: ErrorContext):
        """Update error metrics"""
        self.metrics["total_errors"] += 1
        
        # Update category metrics
        category = error_context.category.value
        self.metrics["errors_by_category"][category] = (
            self.metrics["errors_by_category"].get(category, 0) + 1
        )
        
        # Update severity metrics
        severity = error_context.severity.value
        self.metrics["errors_by_severity"][severity] = (
            self.metrics["errors_by_severity"].get(severity, 0) + 1
        )

    def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt to recover from the error"""
        # Find matching recovery actions
        matching_actions = self._find_matching_recovery_actions(error_context)
        
        for action in matching_actions:
            try:
                self.logger.info(f"Attempting recovery action: {action.name}")
                
                # Execute recovery action
                success = self._execute_recovery_action(action, error_context)
                
                if success:
                    error_context.resolved = True
                    error_context.resolution_time = datetime.now()
                    error_context.recovery_strategy = action.strategy
                    self.logger.info(f"Recovery successful: {action.name}")
                    break
                else:
                    error_context.recovery_attempts += 1
                    self.logger.warning(f"Recovery failed: {action.name}")
                    
            except Exception as recovery_error:
                self.logger.error(f"Recovery action failed: {action.name} - {recovery_error}")
                error_context.recovery_attempts += 1

    def _find_matching_recovery_actions(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Find recovery actions that match the error context"""
        matching_actions = []
        
        for action in self.recovery_actions.values():
            if self._action_matches_error(action, error_context):
                matching_actions.append(action)
        
        # Sort by priority (could be based on strategy, conditions, etc.)
        return sorted(matching_actions, key=lambda x: x.max_attempts, reverse=True)

    def _action_matches_error(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Check if a recovery action matches the error context"""
        conditions = action.conditions
        
        # Check error category
        if "categories" in conditions:
            if error_context.category.value not in conditions["categories"]:
                return False
        
        # Check error type
        if "error_types" in conditions:
            if error_context.error_type not in conditions["error_types"]:
                return False
        
        # Check severity
        if "min_severity" in conditions:
            severity_order = {
                ErrorSeverity.LOW: 1,
                ErrorSeverity.MEDIUM: 2,
                ErrorSeverity.HIGH: 3,
                ErrorSeverity.CRITICAL: 4
            }
            if severity_order[error_context.severity] < severity_order[ErrorSeverity[conditions["min_severity"]]]:
                return False
        
        return True

    def _execute_recovery_action(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute a recovery action"""
        try:
            if action.strategy == RecoveryStrategy.RETRY:
                return self._retry_operation(action, error_context)
            elif action.strategy == RecoveryStrategy.FALLBACK:
                return self._execute_fallback(action, error_context)
            elif action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._handle_circuit_breaker(action, error_context)
            elif action.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._enable_graceful_degradation(action, error_context)
            else:
                return action.handler(error_context)
        except Exception:
            return False

    def _retry_operation(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Implement retry logic with backoff"""
        for attempt in range(action.max_attempts):
            try:
                # Calculate backoff delay
                if action.backoff_strategy == "exponential":
                    delay = (2 ** attempt) * 0.1  # 0.1, 0.2, 0.4, 0.8 seconds
                else:  # linear
                    delay = attempt * 0.5  # 0, 0.5, 1.0, 1.5 seconds
                
                if attempt > 0:
                    time.sleep(delay)
                
                # Execute retry
                result = action.handler(error_context)
                if result:
                    self.logger.info(f"Retry successful on attempt {attempt + 1}")
                    return True
                    
            except Exception as retry_error:
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {retry_error}")
        
        return False

    def _execute_fallback(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute fallback logic"""
        try:
            result = action.handler(error_context)
            if result:
                self.logger.info(f"Fallback executed successfully: {action.name}")
                return True
        except Exception as fallback_error:
            self.logger.error(f"Fallback failed: {fallback_error}")
        
        return False

    def _handle_circuit_breaker(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Handle circuit breaker logic"""
        service_key = f"{error_context.service_name}:{error_context.function_name}"
        
        if service_key not in self.circuit_breakers:
            self.circuit_breakers[service_key] = CircuitBreakerState(
                service_name=service_key,
                state="CLOSED"
            )
        
        cb = self.circuit_breakers[service_key]
        
        if cb.state == "OPEN":
            # Check if timeout has passed
            if (datetime.now() - cb.last_failure_time).seconds > cb.timeout_seconds:
                cb.state = "HALF_OPEN"
                cb.current_half_open_calls = 0
                self.logger.info(f"Circuit breaker transitioning to HALF_OPEN: {service_key}")
            else:
                # Circuit is open, fail fast
                return False
        
        if cb.state == "HALF_OPEN":
            if cb.current_half_open_calls >= cb.half_open_max_calls:
                return False
            cb.current_half_open_calls += 1
        
        # Try the operation
        try:
            result = action.handler(error_context)
            if result:
                cb.success_count += 1
                cb.last_success_time = datetime.now()
                
                if cb.state == "HALF_OPEN" and cb.success_count >= cb.success_threshold:
                    cb.state = "CLOSED"
                    cb.failure_count = 0
                    self.logger.info(f"Circuit breaker closed: {service_key}")
                
                return True
            else:
                self._handle_circuit_breaker_failure(cb, service_key)
                return False
                
        except Exception:
            self._handle_circuit_breaker_failure(cb, service_key)
            return False

    def _handle_circuit_breaker_failure(self, cb: CircuitBreakerState, service_key: str):
        """Handle circuit breaker failure"""
        cb.failure_count += 1
        cb.last_failure_time = datetime.now()
        
        if cb.failure_count >= cb.failure_threshold:
            cb.state = "OPEN"
            self.metrics["circuit_breaker_trips"] += 1
            self.logger.warning(f"Circuit breaker opened: {service_key}")

    def _enable_graceful_degradation(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Enable graceful degradation"""
        degradation_key = f"{error_context.service_name}:{error_context.function_name}"
        self.active_degradations[degradation_key] = datetime.now()
        self.metrics["degradation_events"] += 1
        
        try:
            result = action.handler(error_context)
            self.logger.info(f"Graceful degradation enabled: {degradation_key}")
            return result
        except Exception as degradation_error:
            self.logger.error(f"Graceful degradation failed: {degradation_error}")
            return False

    def _check_circuit_breaker(self, error_context: ErrorContext):
        """Check if circuit breaker should be activated"""
        service_key = f"{error_context.service_name}:{error_context.function_name}"
        
        if service_key in self.circuit_breakers:
            cb = self.circuit_breakers[service_key]
            self._handle_circuit_breaker_failure(cb, service_key)

    def _analyze_error_pattern(self, error_context: ErrorContext):
        """Analyze error patterns for predictive handling"""
        # Create pattern key
        pattern_key = f"{error_context.category.value}:{error_context.error_type}"
        
        # Update pattern count
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
        
        # Check for recurring patterns
        if self.error_patterns[pattern_key] > 5:  # Threshold for pattern detection
            self.logger.warning(f"Recurring error pattern detected: {pattern_key}")
            # Could trigger additional monitoring, alerts, or preemptive actions

    def _pattern_analysis_loop(self):
        """Background task for pattern analysis"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._analyze_patterns()
            except Exception as e:
                self.logger.error(f"Pattern analysis error: {e}")

    def _analyze_patterns(self):
        """Analyze error patterns and trends"""
        if len(self.error_history) < 10:
            return
        
        # Get recent errors (last hour)
        recent_errors = [
            error for error in self.error_history
            if (datetime.now() - error.timestamp).seconds < 3600
        ]
        
        if len(recent_errors) > 10:  # High error rate
            self.logger.warning(f"High error rate detected: {len(recent_errors)} errors in last hour")

    def _circuit_breaker_monitor_loop(self):
        """Background task for circuit breaker monitoring"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._monitor_circuit_breakers()
            except Exception as e:
                self.logger.error(f"Circuit breaker monitor error: {e}")

    def _monitor_circuit_breakers(self):
        """Monitor and manage circuit breakers"""
        current_time = datetime.now()
        
        for service_key, cb in self.circuit_breakers.items():
            if cb.state == "OPEN":
                # Check if it's time to transition to HALF_OPEN
                if cb.last_failure_time and (current_time - cb.last_failure_time).seconds > cb.timeout_seconds:
                    cb.state = "HALF_OPEN"
                    cb.current_half_open_calls = 0
                    self.logger.info(f"Circuit breaker auto-transitioning to HALF_OPEN: {service_key}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        return {
            "service_name": self.service_name,
            "metrics": self.metrics,
            "recent_errors": len([
                e for e in self.error_history
                if (datetime.now() - e.timestamp).seconds < 3600
            ]),
            "circuit_breakers": {
                service: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count
                }
                for service, cb in self.circuit_breakers.items()
            },
            "active_degradations": len(self.active_degradations),
            "error_patterns": dict(list(self.error_patterns.items())[:10])  # Top 10 patterns
        }

    def is_service_degraded(self, function_name: str) -> bool:
        """Check if a service function is currently degraded"""
        degradation_key = f"{self.service_name}:{function_name}"
        return degradation_key in self.active_degradations

    def clear_degradation(self, function_name: str):
        """Clear degradation for a service function"""
        degradation_key = f"{self.service_name}:{function_name}"
        if degradation_key in self.active_degradations:
            del self.active_degradations[degradation_key]
            self.logger.info(f"Degradation cleared: {degradation_key}")

# Decorator for automatic error handling
def xorb_error_handler(category: ErrorCategory = ErrorCategory.UNKNOWN,
                      severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      retry_count: int = 0,
                      fallback_func: Optional[Callable] = None):
    """Decorator for automatic error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create error handler for this service
            service_name = getattr(args[0], '__class__', 'unknown').__name__ if args else 'unknown'
            
            if not hasattr(wrapper, '_error_handler'):
                wrapper._error_handler = XORBErrorHandler(service_name)
            
            error_handler = wrapper._error_handler
            
            # Try main function
            for attempt in range(max(1, retry_count + 1)):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_context = error_handler.handle_error(
                        e, category, severity,
                        context={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_attempts": retry_count + 1
                        }
                    )
                    
                    if attempt == retry_count:  # Last attempt
                        if fallback_func:
                            try:
                                return fallback_func(*args, **kwargs)
                            except Exception as fallback_error:
                                error_handler.handle_error(
                                    fallback_error, category, ErrorSeverity.HIGH,
                                    context={"fallback_function": fallback_func.__name__}
                                )
                        raise e
                    
                    # Wait before retry
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    
        return wrapper
    return decorator

# Async version of the decorator
def xorb_async_error_handler(category: ErrorCategory = ErrorCategory.UNKNOWN,
                            severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                            retry_count: int = 0,
                            fallback_func: Optional[Callable] = None):
    """Async decorator for automatic error handling"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get or create error handler for this service
            service_name = getattr(args[0], '__class__', 'unknown').__name__ if args else 'unknown'
            
            if not hasattr(wrapper, '_error_handler'):
                wrapper._error_handler = XORBErrorHandler(service_name)
            
            error_handler = wrapper._error_handler
            
            # Try main function
            for attempt in range(max(1, retry_count + 1)):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_context = error_handler.handle_error(
                        e, category, severity,
                        context={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_attempts": retry_count + 1
                        }
                    )
                    
                    if attempt == retry_count:  # Last attempt
                        if fallback_func:
                            try:
                                if asyncio.iscoroutinefunction(fallback_func):
                                    return await fallback_func(*args, **kwargs)
                                else:
                                    return fallback_func(*args, **kwargs)
                            except Exception as fallback_error:
                                error_handler.handle_error(
                                    fallback_error, category, ErrorSeverity.HIGH,
                                    context={"fallback_function": fallback_func.__name__}
                                )
                        raise e
                    
                    # Wait before retry
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    
        return wrapper
    return decorator

# Global error handler registry
_global_error_handlers: Dict[str, XORBErrorHandler] = {}

def get_error_handler(service_name: str) -> XORBErrorHandler:
    """Get or create error handler for a service"""
    if service_name not in _global_error_handlers:
        _global_error_handlers[service_name] = XORBErrorHandler(service_name)
    return _global_error_handlers[service_name]

def get_all_error_handlers() -> Dict[str, XORBErrorHandler]:
    """Get all registered error handlers"""
    return _global_error_handlers.copy()

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    error_handler = XORBErrorHandler("test_service")
    
    # Register a recovery action
    def retry_database_connection(error_context: ErrorContext) -> bool:
        print(f"Retrying database connection for error: {error_context.error_id}")
        # Simulate retry logic
        return True
    
    recovery_action = RecoveryAction(
        action_id="db_retry",
        name="Database Connection Retry",
        strategy=RecoveryStrategy.RETRY,
        handler=retry_database_connection,
        max_attempts=3,
        conditions={"categories": ["database"]}
    )
    
    error_handler.register_recovery_action(recovery_action)
    
    # Test error handling
    try:
        raise ConnectionError("Database connection failed")
    except Exception as e:
        error_context = error_handler.handle_error(
            e, 
            ErrorCategory.DATABASE, 
            ErrorSeverity.HIGH,
            context={"operation": "user_lookup", "table": "users"}
        )
        print(f"Error handled: {error_context.error_id}")
    
    # Get error summary
    summary = error_handler.get_error_summary()
    print(f"Error Summary: {json.dumps(summary, indent=2, default=str)}")