#!/usr/bin/env python3
"""
XORB Resilience & Scalability Layer - Circuit Breaker & Fault Tolerance
Advanced fault tolerance with circuit breakers, bulkheads, and self-healing
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque
import uuid
import threading
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit tripped, failing fast
    HALF_OPEN = "half_open" # Testing if service recovered

class FaultType(Enum):
    """Types of faults detected"""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"

class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    JITTERED_EXPONENTIAL = "jittered_exponential"

class BulkheadType(Enum):
    """Bulkhead isolation types"""
    THREAD_POOL = "thread_pool"
    SEMAPHORE = "semaphore"
    QUEUE_BASED = "queue_based"
    RESOURCE_POOL = "resource_pool"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5           # Number of failures to trip circuit
    recovery_timeout: int = 60           # Seconds to wait before trying half-open
    success_threshold: int = 3           # Successful calls to close circuit from half-open
    timeout: float = 30.0               # Request timeout in seconds
    slow_call_duration_threshold: float = 10.0  # Slow call threshold in seconds
    slow_call_rate_threshold: float = 0.5       # Percentage of slow calls to trip
    minimum_number_of_calls: int = 10           # Minimum calls before evaluating
    sliding_window_size: int = 100              # Size of sliding window for metrics
    failure_rate_threshold: float = 50.0       # Failure rate percentage to trip

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[str] = field(default_factory=lambda: [
        'TimeoutError', 'ConnectionError', 'aiohttp.ClientError'
    ])

@dataclass
class BulkheadConfig:
    """Bulkhead configuration"""
    type: BulkheadType = BulkheadType.SEMAPHORE
    max_concurrent_calls: int = 10
    max_queue_size: int = 100
    queue_timeout: float = 30.0
    thread_pool_size: int = 10

@dataclass
class FaultRecord:
    """Record of a fault occurrence"""
    fault_id: str
    service_name: str
    fault_type: FaultType
    error_message: str
    timestamp: datetime
    duration_ms: float = 0.0
    request_context: Dict[str, Any] = field(default_factory=dict)
    recovery_action: Optional[str] = None

@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    slow_calls: int = 0
    failure_rate: float = 0.0
    slow_call_rate: float = 0.0
    avg_response_time: float = 0.0
    circuit_trips: int = 0
    time_in_open_state: float = 0.0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None

class XORBCircuitBreaker:
    """Circuit breaker with advanced fault tolerance"""
    
    def __init__(self, service_name: str, config: Optional[CircuitBreakerConfig] = None):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.circuit_id = f"circuit_{service_name}_{int(time.time())}"
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_change_time = datetime.now()
        
        # Metrics tracking
        self.call_history: deque = deque(maxlen=self.config.sliding_window_size)
        self.metrics = CircuitBreakerMetrics()
        self.fault_records: List[FaultRecord] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Circuit breaker initialized for {service_name}: {self.circuit_id}")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        start_time = time.time()
        
        try:
            # Check if circuit allows execution
            if not self._can_execute():
                raise CircuitBreakerOpenException(
                    f"Circuit breaker is open for {self.service_name}"
                )
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Record success
            execution_time = (time.time() - start_time) * 1000  # ms
            await self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = (time.time() - start_time) * 1000  # ms
            await self._record_failure(e, execution_time)
            raise e
    
    def _can_execute(self) -> bool:
        """Check if the circuit allows execution"""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if datetime.now() - self.state_change_time >= timedelta(seconds=self.config.recovery_timeout):
                    self._transition_to_half_open()
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    async def _record_success(self, execution_time: float):
        """Record successful execution"""
        with self.lock:
            call_record = {
                'timestamp': datetime.now(),
                'success': True,
                'execution_time': execution_time,
                'slow_call': execution_time > self.config.slow_call_duration_threshold * 1000
            }
            
            self.call_history.append(call_record)
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = call_record['timestamp']
            
            # Update circuit state based on success
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.OPEN:
                # Should not happen, but reset if it does
                self._transition_to_closed()
            
            # Update metrics
            await self._update_metrics()
    
    async def _record_failure(self, exception: Exception, execution_time: float):
        """Record failed execution"""
        with self.lock:
            call_record = {
                'timestamp': datetime.now(),
                'success': False,
                'execution_time': execution_time,
                'exception': str(exception),
                'exception_type': type(exception).__name__
            }
            
            self.call_history.append(call_record)
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = call_record['timestamp']
            
            # Create fault record
            fault_record = FaultRecord(
                fault_id=str(uuid.uuid4()),
                service_name=self.service_name,
                fault_type=self._classify_fault(exception),
                error_message=str(exception),
                timestamp=call_record['timestamp'],
                duration_ms=execution_time
            )
            
            self.fault_records.append(fault_record)
            
            # Update circuit state based on failure
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                
                # Check if we should trip the circuit
                if await self._should_trip_circuit():
                    self._transition_to_open()
            
            # Update metrics
            await self._update_metrics()
    
    def _classify_fault(self, exception: Exception) -> FaultType:
        """Classify the type of fault based on exception"""
        exception_name = type(exception).__name__
        exception_message = str(exception).lower()
        
        if 'timeout' in exception_name.lower() or 'timeout' in exception_message:
            return FaultType.TIMEOUT
        elif 'connection' in exception_name.lower() or 'connection' in exception_message:
            return FaultType.CONNECTION_ERROR
        elif 'http' in exception_name.lower() or any(code in exception_message for code in ['400', '401', '403', '404', '500', '502', '503']):
            return FaultType.HTTP_ERROR
        elif 'unavailable' in exception_message or '503' in exception_message:
            return FaultType.SERVICE_UNAVAILABLE
        elif 'rate limit' in exception_message or '429' in exception_message:
            return FaultType.RATE_LIMIT
        elif 'resource' in exception_message or 'memory' in exception_message:
            return FaultType.RESOURCE_EXHAUSTION
        elif 'validation' in exception_message or '400' in exception_message:
            return FaultType.VALIDATION_ERROR
        else:
            return FaultType.UNKNOWN_ERROR
    
    async def _should_trip_circuit(self) -> bool:
        """Determine if circuit should be tripped"""
        if not self.call_history or len(self.call_history) < self.config.minimum_number_of_calls:
            return False
        
        # Calculate failure rate
        recent_calls = list(self.call_history)[-self.config.minimum_number_of_calls:]
        failures = len([call for call in recent_calls if not call['success']])
        failure_rate = (failures / len(recent_calls)) * 100
        
        # Calculate slow call rate
        slow_calls = len([call for call in recent_calls if call.get('slow_call', False)])
        slow_call_rate = (slow_calls / len(recent_calls)) * 100
        
        # Trip if failure rate or slow call rate exceeds threshold
        return (failure_rate >= self.config.failure_rate_threshold or 
                slow_call_rate >= self.config.slow_call_rate_threshold * 100)
    
    def _transition_to_open(self):
        """Transition circuit to open state"""
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker opening for {self.service_name}")
            self.state = CircuitState.OPEN
            self.state_change_time = datetime.now()
            self.failure_count = 0
            self.success_count = 0
            self.metrics.circuit_trips += 1
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        if self.state != CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker transitioning to half-open for {self.service_name}")
            self.state = CircuitState.HALF_OPEN
            self.state_change_time = datetime.now()
            self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition circuit to closed state"""
        if self.state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker closing for {self.service_name}")
            self.state = CircuitState.CLOSED
            self.state_change_time = datetime.now()
            self.failure_count = 0
            self.success_count = 0
    
    async def _update_metrics(self):
        """Update circuit breaker metrics"""
        try:
            if not self.call_history:
                return
            
            recent_calls = list(self.call_history)
            
            # Calculate rates
            if len(recent_calls) > 0:
                successful_calls = len([call for call in recent_calls if call['success']])
                failed_calls = len([call for call in recent_calls if not call['success']])
                slow_calls = len([call for call in recent_calls if call.get('slow_call', False)])
                
                self.metrics.failure_rate = (failed_calls / len(recent_calls)) * 100
                self.metrics.slow_call_rate = (slow_calls / len(recent_calls)) * 100
                
                # Calculate average response time
                execution_times = [call['execution_time'] for call in recent_calls]
                self.metrics.avg_response_time = statistics.mean(execution_times) if execution_times else 0.0
            
            # Calculate time in open state
            if self.state == CircuitState.OPEN:
                self.metrics.time_in_open_state = (datetime.now() - self.state_change_time).total_seconds()
            
        except Exception as e:
            logger.error(f"Failed to update circuit breaker metrics: {e}")
    
    def _acquire_circuit(self):
        """Context manager for circuit breaker execution"""
        return CircuitBreakerContext(self)
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self.lock:
            return {
                'circuit_id': self.circuit_id,
                'service_name': self.service_name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'metrics': asdict(self.metrics),
                'config': asdict(self.config),
                'state_change_time': self.state_change_time.isoformat(),
                'recent_faults': [asdict(fault) for fault in self.fault_records[-10:]],
                'call_history_size': len(self.call_history)
            }

class CircuitBreakerContext:
    """Context manager for circuit breaker execution"""
    
    def __init__(self, circuit_breaker: XORBCircuitBreaker):
        self.circuit_breaker = circuit_breaker
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class XORBRetryMechanism:
    """Advanced retry mechanism with multiple strategies"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.retry_id = str(uuid.uuid4())
        
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                
                # Log successful retry if not first attempt
                if attempt > 1:
                    execution_time = (time.time() - start_time) * 1000
                    logger.info(f"Retry successful on attempt {attempt} after {execution_time:.2f}ms")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.warning(f"Exception not retryable: {type(e).__name__}")
                    raise e
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts:
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                
                logger.warning(f"Retry attempt {attempt} failed: {str(e)}, retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(f"All {self.config.max_attempts} retry attempts failed")
        raise RetryExhaustedException(f"Max retry attempts ({self.config.max_attempts}) exceeded") from last_exception
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        exception_name = type(exception).__name__
        return any(retryable in exception_name for retryable in self.config.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.config.base_delay * self._fibonacci(attempt)
        elif self.config.strategy == RetryStrategy.JITTERED_EXPONENTIAL:
            base_delay = self.config.base_delay * (self.config.multiplier ** (attempt - 1))
            jitter = base_delay * 0.1 * (2 * time.time() % 1 - 1)  # ±10% jitter
            delay = base_delay + jitter
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        return min(delay, self.config.max_delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number"""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)

class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted"""
    pass

class XORBBulkhead:
    """Bulkhead pattern implementation for resource isolation"""
    
    def __init__(self, name: str, config: Optional[BulkheadConfig] = None):
        self.name = name
        self.config = config or BulkheadConfig()
        self.bulkhead_id = str(uuid.uuid4())
        
        # Initialize based on bulkhead type
        if self.config.type == BulkheadType.SEMAPHORE:
            self.semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        elif self.config.type == BulkheadType.QUEUE_BASED:
            self.queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Metrics
        self.active_calls = 0
        self.queued_calls = 0
        self.rejected_calls = 0
        self.total_calls = 0
        
        logger.info(f"Bulkhead initialized: {name} ({self.config.type.value})")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead protection"""
        self.total_calls += 1
        
        try:
            if self.config.type == BulkheadType.SEMAPHORE:
                return await self._execute_with_semaphore(func, *args, **kwargs)
            elif self.config.type == BulkheadType.QUEUE_BASED:
                return await self._execute_with_queue(func, *args, **kwargs)
            else:
                # Direct execution as fallback
                return await func(*args, **kwargs)
                
        except asyncio.TimeoutError:
            self.rejected_calls += 1
            raise BulkheadRejectedException(f"Bulkhead {self.name} rejected call due to timeout")
        except Exception as e:
            raise e
    
    async def _execute_with_semaphore(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with semaphore-based bulkhead"""
        try:
            # Acquire semaphore with timeout
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.queue_timeout
            )
            
            self.active_calls += 1
            
            try:
                return await func(*args, **kwargs)
            finally:
                self.active_calls -= 1
                self.semaphore.release()
                
        except asyncio.TimeoutError:
            raise BulkheadRejectedException(f"Bulkhead {self.name} semaphore acquisition timeout")
    
    async def _execute_with_queue(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with queue-based bulkhead"""
        # Create execution task
        task_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        try:
            # Add to queue with timeout
            await asyncio.wait_for(
                self.queue.put((task_id, func, args, kwargs, future)),
                timeout=self.config.queue_timeout
            )
            
            self.queued_calls += 1
            
            # Wait for execution result
            return await future
            
        except asyncio.TimeoutError:
            raise BulkheadRejectedException(f"Bulkhead {self.name} queue full")
        finally:
            self.queued_calls = max(0, self.queued_calls - 1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get bulkhead status"""
        return {
            'bulkhead_id': self.bulkhead_id,
            'name': self.name,
            'type': self.config.type.value,
            'config': asdict(self.config),
            'metrics': {
                'active_calls': self.active_calls,
                'queued_calls': self.queued_calls,
                'rejected_calls': self.rejected_calls,
                'total_calls': self.total_calls,
                'rejection_rate': (self.rejected_calls / max(1, self.total_calls)) * 100
            }
        }

class BulkheadRejectedException(Exception):
    """Exception raised when bulkhead rejects a call"""
    pass

class XORBFaultToleranceManager:
    """Central fault tolerance manager"""
    
    def __init__(self):
        self.manager_id = str(uuid.uuid4())
        self.circuit_breakers: Dict[str, XORBCircuitBreaker] = {}
        self.retry_mechanisms: Dict[str, XORBRetryMechanism] = {}
        self.bulkheads: Dict[str, XORBBulkhead] = {}
        
        # Global fault tracking
        self.global_fault_history: List[FaultRecord] = []
        self.fault_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info(f"Fault tolerance manager initialized: {self.manager_id}")
    
    def create_circuit_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> XORBCircuitBreaker:
        """Create and register a circuit breaker"""
        circuit_breaker = XORBCircuitBreaker(service_name, config)
        self.circuit_breakers[service_name] = circuit_breaker
        return circuit_breaker
    
    def create_retry_mechanism(self, service_name: str, config: Optional[RetryConfig] = None) -> XORBRetryMechanism:
        """Create and register a retry mechanism"""
        retry_mechanism = XORBRetryMechanism(config)
        self.retry_mechanisms[service_name] = retry_mechanism
        return retry_mechanism
    
    def create_bulkhead(self, name: str, config: Optional[BulkheadConfig] = None) -> XORBBulkhead:
        """Create and register a bulkhead"""
        bulkhead = XORBBulkhead(name, config)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    async def execute_with_protection(self, service_name: str, func: Callable, 
                                    enable_circuit_breaker: bool = True,
                                    enable_retry: bool = True,
                                    enable_bulkhead: bool = False,
                                    bulkhead_name: Optional[str] = None,
                                    *args, **kwargs) -> Any:
        """Execute function with full fault tolerance protection"""
        
        # Prepare protection layers
        execution_func = func
        
        # Layer 3: Bulkhead (outermost)
        if enable_bulkhead and bulkhead_name and bulkhead_name in self.bulkheads:
            bulkhead = self.bulkheads[bulkhead_name]
            original_func = execution_func
            execution_func = lambda *a, **kw: bulkhead.execute(original_func, *a, **kw)
        
        # Layer 2: Retry mechanism
        if enable_retry and service_name in self.retry_mechanisms:
            retry_mechanism = self.retry_mechanisms[service_name]
            original_func = execution_func
            execution_func = lambda *a, **kw: retry_mechanism.execute_with_retry(original_func, *a, **kw)
        
        # Layer 1: Circuit breaker (innermost)
        if enable_circuit_breaker and service_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[service_name]
            original_func = execution_func
            execution_func = lambda *a, **kw: circuit_breaker.execute(original_func, *a, **kw)
        
        # Execute with all protection layers
        try:
            return await execution_func(*args, **kwargs)
        except Exception as e:
            # Record fault globally
            await self._record_global_fault(service_name, e)
            raise e
    
    async def _record_global_fault(self, service_name: str, exception: Exception):
        """Record fault in global tracking system"""
        try:
            fault_record = FaultRecord(
                fault_id=str(uuid.uuid4()),
                service_name=service_name,
                fault_type=self._classify_fault_type(exception),
                error_message=str(exception),
                timestamp=datetime.now()
            )
            
            self.global_fault_history.append(fault_record)
            
            # Keep only recent faults (last 1000)
            if len(self.global_fault_history) > 1000:
                self.global_fault_history = self.global_fault_history[-1000:]
            
            # Update fault patterns
            self._update_fault_patterns(fault_record)
            
        except Exception as e:
            logger.error(f"Failed to record global fault: {e}")
    
    def _classify_fault_type(self, exception: Exception) -> FaultType:
        """Classify fault type from exception"""
        exception_name = type(exception).__name__
        exception_message = str(exception).lower()
        
        if 'timeout' in exception_name.lower():
            return FaultType.TIMEOUT
        elif 'connection' in exception_name.lower():
            return FaultType.CONNECTION_ERROR
        elif 'circuitbreaker' in exception_name.lower():
            return FaultType.SERVICE_UNAVAILABLE
        elif 'bulkhead' in exception_name.lower():
            return FaultType.RESOURCE_EXHAUSTION
        else:
            return FaultType.UNKNOWN_ERROR
    
    def _update_fault_patterns(self, fault_record: FaultRecord):
        """Update fault pattern analysis"""
        try:
            pattern_key = f"{fault_record.service_name}_{fault_record.fault_type.value}"
            
            pattern_entry = {
                'timestamp': fault_record.timestamp.isoformat(),
                'fault_type': fault_record.fault_type.value,
                'error_message': fault_record.error_message
            }
            
            self.fault_patterns[pattern_key].append(pattern_entry)
            
            # Keep only recent patterns (last 50 per pattern)
            if len(self.fault_patterns[pattern_key]) > 50:
                self.fault_patterns[pattern_key] = self.fault_patterns[pattern_key][-50:]
                
        except Exception as e:
            logger.error(f"Failed to update fault patterns: {e}")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance status"""
        try:
            circuit_breaker_status = {}
            for name, cb in self.circuit_breakers.items():
                circuit_breaker_status[name] = cb.get_status()
            
            bulkhead_status = {}
            for name, bh in self.bulkheads.items():
                bulkhead_status[name] = bh.get_status()
            
            # Global fault statistics
            recent_faults = [fault for fault in self.global_fault_history 
                           if (datetime.now() - fault.timestamp).total_seconds() < 3600]
            
            fault_type_counts = defaultdict(int)
            for fault in recent_faults:
                fault_type_counts[fault.fault_type.value] += 1
            
            return {
                'manager_id': self.manager_id,
                'circuit_breakers': circuit_breaker_status,
                'bulkheads': bulkhead_status,
                'global_statistics': {
                    'total_faults_24h': len([f for f in self.global_fault_history 
                                           if (datetime.now() - f.timestamp).total_seconds() < 86400]),
                    'recent_faults_1h': len(recent_faults),
                    'fault_type_distribution': dict(fault_type_counts),
                    'fault_patterns_detected': len(self.fault_patterns),
                    'most_problematic_services': self._get_most_problematic_services()
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {'error': str(e)}
    
    def _get_most_problematic_services(self) -> List[Dict[str, Any]]:
        """Get services with most faults"""
        try:
            service_fault_counts = defaultdict(int)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            
            for fault in self.global_fault_history:
                if fault.timestamp > recent_cutoff:
                    service_fault_counts[fault.service_name] += 1
            
            # Sort by fault count and return top 5
            sorted_services = sorted(service_fault_counts.items(), key=lambda x: x[1], reverse=True)
            
            return [
                {'service_name': service, 'fault_count': count}
                for service, count in sorted_services[:5]
            ]
            
        except Exception as e:
            logger.error(f"Failed to get problematic services: {e}")
            return []

# Example usage and testing
async def main():
    """Example usage of XORB Fault Tolerance components"""
    try:
        print("🛡️ XORB Fault Tolerance System initializing...")
        
        # Initialize fault tolerance manager
        ft_manager = XORBFaultToleranceManager()
        
        # Create circuit breakers for services
        neural_cb = ft_manager.create_circuit_breaker("neural_orchestrator", CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            timeout=10.0
        ))
        
        # Create retry mechanisms
        retry_config = RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0
        )
        ft_manager.create_retry_mechanism("neural_orchestrator", retry_config)
        
        # Create bulkheads
        ft_manager.create_bulkhead("neural_processing", BulkheadConfig(
            type=BulkheadType.SEMAPHORE,
            max_concurrent_calls=5
        ))
        
        print("✅ Fault tolerance components initialized")
        
        # Test circuit breaker
        async def test_service_call():
            """Simulate a service call"""
            await asyncio.sleep(0.1)  # Simulate processing time
            if time.time() % 3 < 1:  # Simulate intermittent failures
                raise Exception("Simulated service failure")
            return {"status": "success", "data": "test_data"}
        
        print("\n🧪 Testing fault tolerance mechanisms...")
        
        # Execute several calls with protection
        for i in range(10):
            try:
                result = await ft_manager.execute_with_protection(
                    service_name="neural_orchestrator",
                    func=test_service_call,
                    enable_circuit_breaker=True,
                    enable_retry=True,
                    enable_bulkhead=True,
                    bulkhead_name="neural_processing"
                )
                print(f"✅ Call {i+1}: Success")
            except Exception as e:
                print(f"❌ Call {i+1}: Failed - {str(e)}")
            
            await asyncio.sleep(0.5)
        
        # Get status
        status = ft_manager.get_comprehensive_status()
        print(f"\n📊 Fault Tolerance Status:")
        print(f"- Circuit Breakers: {len(status['circuit_breakers'])}")
        print(f"- Bulkheads: {len(status['bulkheads'])}")
        print(f"- Recent Faults (1h): {status['global_statistics']['recent_faults_1h']}")
        print(f"- Total Faults (24h): {status['global_statistics']['total_faults_24h']}")
        
        for service_name, cb_status in status['circuit_breakers'].items():
            print(f"\n🔌 Circuit Breaker - {service_name}:")
            print(f"  - State: {cb_status['state']}")
            print(f"  - Total Calls: {cb_status['metrics']['total_calls']}")
            print(f"  - Success Rate: {(cb_status['metrics']['successful_calls']/max(1, cb_status['metrics']['total_calls']))*100:.1f}%")
            print(f"  - Circuit Trips: {cb_status['metrics']['circuit_trips']}")
        
        print(f"\n✅ XORB Fault Tolerance System demonstration completed!")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())