#!/usr/bin/env python3
"""
XORB Advanced Service Mesh Implementation
Intelligent routing, load balancing, and EPYC-optimized traffic management
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import hashlib

import aiohttp
import aioredis
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    EPYC_NUMA_AWARE = "epyc_numa_aware"
    AI_WORKLOAD_OPTIMIZED = "ai_workload_optimized"

class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing recovery

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceEndpoint:
    """Enhanced service endpoint with EPYC optimization metadata."""
    host: str
    port: int
    weight: int = 100
    numa_node: Optional[int] = None
    ccx_affinity: Optional[int] = None
    current_connections: int = 0
    response_time_avg: float = 0.0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    thermal_load: float = 0.0  # CPU temperature impact

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def efficiency_score(self) -> float:
        """Calculate endpoint efficiency considering EPYC factors."""
        base_score = 1.0

        # Health penalty
        if self.health_status == HealthStatus.UNHEALTHY:
            return 0.0
        elif self.health_status == HealthStatus.DEGRADED:
            base_score *= 0.5

        # Response time penalty (lower is better)
        if self.response_time_avg > 0:
            base_score *= max(0.1, 1.0 - (self.response_time_avg / 1000))  # 1s baseline

        # Connection load penalty
        if self.current_connections > 100:
            base_score *= max(0.1, 1.0 - (self.current_connections / 1000))

        # Thermal penalty for EPYC optimization
        if self.thermal_load > 0.8:
            base_score *= 0.5
        elif self.thermal_load > 0.6:
            base_score *= 0.8

        return base_score

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_duration: int = 30  # seconds
    half_open_max_calls: int = 3
    reset_timeout: int = 60  # seconds

@dataclass
class RetryConfig:
    """Retry policy configuration."""
    max_attempts: int = 3
    base_delay: float = 0.1  # seconds
    max_delay: float = 2.0   # seconds
    backoff_multiplier: float = 2.0
    retryable_status_codes: List[int] = field(default_factory=lambda: [502, 503, 504])

class CircuitBreaker:
    """Advanced circuit breaker with EPYC thermal awareness."""

    def __init__(self, config: CircuitBreakerConfig, service_name: str):
        self.config = config
        self.service_name = service_name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

        # Metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state',
            ['service']
        )
        self.circuit_breaker_failures = Counter(
            'circuit_breaker_failures_total',
            'Circuit breaker failures',
            ['service']
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit breaker {self.service_name} transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.service_name} is OPEN")

        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise Exception(f"Circuit breaker {self.service_name} half-open limit exceeded")
            self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_duration

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker {self.service_name} transitioning to CLOSED")

        self.circuit_breaker_state.labels(service=self.service_name).set(0)  # CLOSED

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        self.circuit_breaker_failures.labels(service=self.service_name).inc()

        if (self.state == CircuitBreakerState.CLOSED and
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.service_name} transitioning to OPEN")
            self.circuit_breaker_state.labels(service=self.service_name).set(1)  # OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.service_name} transitioning to OPEN from HALF_OPEN")
            self.circuit_breaker_state.labels(service=self.service_name).set(1)  # OPEN

class EPYCLoadBalancer:
    """EPYC-optimized intelligent load balancer."""

    def __init__(self, service_name: str, strategy: LoadBalancingStrategy = LoadBalancingStrategy.EPYC_NUMA_AWARE):
        self.service_name = service_name
        self.strategy = strategy
        self.endpoints: List[ServiceEndpoint] = []
        self.current_index = 0
        self.redis: Optional[aioredis.Redis] = None

        # Metrics
        self.requests_total = Counter(
            'load_balancer_requests_total',
            'Total load balancer requests',
            ['service', 'endpoint', 'status']
        )
        self.response_time = Histogram(
            'load_balancer_response_time_seconds',
            'Load balancer response time',
            ['service', 'endpoint']
        )
        self.active_connections = Gauge(
            'load_balancer_active_connections',
            'Active connections per endpoint',
            ['service', 'endpoint']
        )

    async def initialize(self, redis_url: str = "redis://redis:6379"):
        """Initialize load balancer with Redis backend."""
        self.redis = aioredis.from_url(redis_url)
        await self._load_endpoints_from_discovery()

    def add_endpoint(self, endpoint: ServiceEndpoint):
        """Add service endpoint."""
        self.endpoints.append(endpoint)
        logger.info(f"Added endpoint {endpoint.endpoint_url} for service {self.service_name}")

    def remove_endpoint(self, host: str, port: int):
        """Remove service endpoint."""
        self.endpoints = [ep for ep in self.endpoints if not (ep.host == host and ep.port == port)]
        logger.info(f"Removed endpoint {host}:{port} for service {self.service_name}")

    async def select_endpoint(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceEndpoint]:
        """Select optimal endpoint using configured strategy."""
        healthy_endpoints = [ep for ep in self.endpoints if ep.health_status != HealthStatus.UNHEALTHY]

        if not healthy_endpoints:
            logger.error(f"No healthy endpoints available for service {self.service_name}")
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._select_least_response_time(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.EPYC_NUMA_AWARE:
            return self._select_epyc_numa_aware(healthy_endpoints, request_context)
        elif self.strategy == LoadBalancingStrategy.AI_WORKLOAD_OPTIMIZED:
            return self._select_ai_workload_optimized(healthy_endpoints, request_context)
        else:
            return self._select_round_robin(healthy_endpoints)

    def _select_round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Simple round-robin selection."""
        if not endpoints:
            return None

        selected = endpoints[self.current_index % len(endpoints)]
        self.current_index = (self.current_index + 1) % len(endpoints)
        return selected

    def _select_least_connections(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with least active connections."""
        return min(endpoints, key=lambda ep: ep.current_connections)

    def _select_weighted_round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round-robin based on endpoint weights."""
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return self._select_round_robin(endpoints)

        # Simple weighted selection (could be optimized with proper algorithm)
        import random
        threshold = random.randint(1, total_weight)
        current_weight = 0

        for endpoint in endpoints:
            current_weight += endpoint.weight
            if current_weight >= threshold:
                return endpoint

        return endpoints[-1]  # Fallback

    def _select_least_response_time(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with lowest average response time."""
        return min(endpoints, key=lambda ep: ep.response_time_avg if ep.response_time_avg > 0 else float('inf'))

    def _select_epyc_numa_aware(self, endpoints: List[ServiceEndpoint], request_context: Optional[Dict[str, Any]]) -> ServiceEndpoint:
        """EPYC NUMA-aware endpoint selection."""
        if not request_context:
            return self._select_least_response_time(endpoints)

        preferred_numa_node = request_context.get('numa_preference')
        workload_type = request_context.get('workload_type', 'balanced')

        # Filter by NUMA preference if specified
        numa_preferred = endpoints
        if preferred_numa_node is not None:
            numa_preferred = [ep for ep in endpoints if ep.numa_node == preferred_numa_node]
            if not numa_preferred:
                numa_preferred = endpoints  # Fallback

        # Score endpoints based on EPYC optimization factors
        scored_endpoints = []
        for endpoint in numa_preferred:
            score = endpoint.efficiency_score

            # Workload-specific adjustments
            if workload_type == 'cpu_intensive' and endpoint.ccx_affinity is not None:
                score *= 1.2  # Prefer CCX-optimized endpoints
            elif workload_type == 'memory_intensive' and endpoint.numa_node == 0:
                score *= 1.1  # Prefer memory controller proximity

            scored_endpoints.append((endpoint, score))

        # Select highest scoring endpoint
        best_endpoint = max(scored_endpoints, key=lambda x: x[1])[0]
        return best_endpoint

    def _select_ai_workload_optimized(self, endpoints: List[ServiceEndpoint], request_context: Optional[Dict[str, Any]]) -> ServiceEndpoint:
        """AI workload optimized selection."""
        if not request_context:
            return self._select_epyc_numa_aware(endpoints, request_context)

        model_type = request_context.get('model_type', 'unknown')
        complexity = request_context.get('complexity', 'standard')

        # Score endpoints for AI workload characteristics
        scored_endpoints = []
        for endpoint in endpoints:
            score = endpoint.efficiency_score

            # AI-specific optimizations
            if model_type in ['llm', 'transformer'] and endpoint.ccx_affinity is not None:
                score *= 1.3  # High cache locality for transformers

            if complexity == 'cognitive' and endpoint.thermal_load < 0.5:
                score *= 1.2  # Prefer cooler endpoints for intensive work

            # Prefer endpoints with lower connection count for AI workloads
            if endpoint.current_connections < 10:
                score *= 1.1

            scored_endpoints.append((endpoint, score))

        best_endpoint = max(scored_endpoints, key=lambda x: x[1])[0]
        return best_endpoint

    async def update_endpoint_stats(self, endpoint: ServiceEndpoint, response_time: float, success: bool):
        """Update endpoint statistics after request."""
        if success:
            endpoint.success_count += 1
            endpoint.failure_count = max(0, endpoint.failure_count - 1)  # Decay failures
        else:
            endpoint.failure_count += 1

        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if endpoint.response_time_avg == 0:
            endpoint.response_time_avg = response_time
        else:
            endpoint.response_time_avg = alpha * response_time + (1 - alpha) * endpoint.response_time_avg

        # Update metrics
        status = "success" if success else "failure"
        self.requests_total.labels(
            service=self.service_name,
            endpoint=endpoint.endpoint_url,
            status=status
        ).inc()

        if success:
            self.response_time.labels(
                service=self.service_name,
                endpoint=endpoint.endpoint_url
            ).observe(response_time)

    async def _load_endpoints_from_discovery(self):
        """Load endpoints from service discovery (Redis-based)."""
        if not self.redis:
            return

        try:
            discovery_key = f"service_discovery:{self.service_name}"
            endpoints_data = await self.redis.get(discovery_key)

            if endpoints_data:
                endpoints = json.loads(endpoints_data)
                for ep_data in endpoints:
                    endpoint = ServiceEndpoint(
                        host=ep_data['host'],
                        port=ep_data['port'],
                        weight=ep_data.get('weight', 100),
                        numa_node=ep_data.get('numa_node'),
                        ccx_affinity=ep_data.get('ccx_affinity')
                    )
                    self.add_endpoint(endpoint)

        except Exception as e:
            logger.error(f"Failed to load endpoints from discovery: {e}")

class ServiceMeshProxy:
    """Advanced service mesh proxy with comprehensive traffic management."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.load_balancers: Dict[str, EPYCLoadBalancer] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.session: Optional[aiohttp.ClientSession] = None

        # Initialize default configurations
        self._initialize_configurations()

        # Metrics
        self.proxy_requests_total = Counter(
            'service_mesh_proxy_requests_total',
            'Total proxy requests',
            ['source_service', 'target_service', 'status']
        )
        self.proxy_response_time = Histogram(
            'service_mesh_proxy_response_time_seconds',
            'Proxy response time',
            ['source_service', 'target_service']
        )

    async def initialize(self):
        """Initialize service mesh proxy."""
        # Create HTTP session with optimized settings for EPYC
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        connector = aiohttp.TCPConnector(
            limit=1000,           # Total connection pool size
            limit_per_host=100,   # Per-host connection limit
            ttl_dns_cache=300,    # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': 'XORB-ServiceMesh/1.0'}
        )

        # Initialize load balancers for configured services
        for service_name in self.config.get('services', {}):
            await self._initialize_service_proxy(service_name)

        logger.info("Service mesh proxy initialized")

    async def close(self):
        """Clean shutdown of service mesh proxy."""
        if self.session:
            await self.session.close()

        for lb in self.load_balancers.values():
            if lb.redis:
                await lb.redis.close()

    async def proxy_request(
        self,
        source_service: str,
        target_service: str,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        request_context: Optional[Dict[str, Any]] = None
    ) -> aiohttp.ClientResponse:
        """Proxy request with intelligent routing and fault tolerance."""

        start_time = time.time()

        # Get load balancer for target service
        load_balancer = self.load_balancers.get(target_service)
        if not load_balancer:
            raise Exception(f"No load balancer configured for service {target_service}")

        # Select optimal endpoint
        endpoint = await load_balancer.select_endpoint(request_context)
        if not endpoint:
            raise Exception(f"No healthy endpoints available for service {target_service}")

        # Get circuit breaker
        circuit_breaker = self.circuit_breakers.get(target_service)
        retry_config = self.retry_configs.get(target_service, RetryConfig())

        # Prepare request
        url = f"{endpoint.endpoint_url}{path}"
        request_headers = headers or {}
        request_headers.update({
            'X-Source-Service': source_service,
            'X-Request-ID': self._generate_request_id(),
            'X-EPYC-Optimized': 'true'
        })

        # Add NUMA affinity hints if available
        if endpoint.numa_node is not None:
            request_headers['X-NUMA-Node'] = str(endpoint.numa_node)
        if endpoint.ccx_affinity is not None:
            request_headers['X-CCX-Affinity'] = str(endpoint.ccx_affinity)

        # Track active connection
        endpoint.current_connections += 1
        self.active_connections.labels(
            service=target_service,
            endpoint=endpoint.endpoint_url
        ).set(endpoint.current_connections)

        try:
            # Execute request with circuit breaker and retry logic
            async def make_request():
                async with self.session.request(
                    method=method,
                    url=url,
                    headers=request_headers,
                    data=data
                ) as response:
                    return response

            if circuit_breaker:
                response = await circuit_breaker.call(self._retry_request, make_request, retry_config)
            else:
                response = await self._retry_request(make_request, retry_config)

            # Update endpoint statistics
            response_time = time.time() - start_time
            await load_balancer.update_endpoint_stats(endpoint, response_time, True)

            # Update metrics
            self.proxy_requests_total.labels(
                source_service=source_service,
                target_service=target_service,
                status="success"
            ).inc()

            self.proxy_response_time.labels(
                source_service=source_service,
                target_service=target_service
            ).observe(response_time)

            return response

        except Exception as e:
            # Update failure statistics
            response_time = time.time() - start_time
            await load_balancer.update_endpoint_stats(endpoint, response_time, False)

            # Update metrics
            self.proxy_requests_total.labels(
                source_service=source_service,
                target_service=target_service,
                status="failure"
            ).inc()

            logger.error(f"Proxy request failed for {target_service}: {e}")
            raise e

        finally:
            # Release connection tracking
            endpoint.current_connections = max(0, endpoint.current_connections - 1)
            self.active_connections.labels(
                service=target_service,
                endpoint=endpoint.endpoint_url
            ).set(endpoint.current_connections)

    async def _retry_request(self, request_func: Callable, retry_config: RetryConfig) -> Any:
        """Execute request with retry logic."""
        last_exception = None

        for attempt in range(retry_config.max_attempts):
            try:
                return await request_func()
            except aiohttp.ClientError as e:
                last_exception = e

                # Check if error is retryable
                if hasattr(e, 'status') and e.status not in retry_config.retryable_status_codes:
                    raise e

                if attempt < retry_config.max_attempts - 1:
                    # Calculate backoff delay
                    delay = min(
                        retry_config.base_delay * (retry_config.backoff_multiplier ** attempt),
                        retry_config.max_delay
                    )
                    await asyncio.sleep(delay)
                    logger.warning(f"Retrying request (attempt {attempt + 1}/{retry_config.max_attempts}) after {delay}s delay")

        # All retries exhausted
        raise last_exception

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return hashlib.md5(f"{time.time()}{id(self)}".encode()).hexdigest()[:16]

    def _initialize_configurations(self):
        """Initialize default service configurations."""
        # Default circuit breaker config
        default_cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout_duration=30,
            half_open_max_calls=3
        )

        # Default retry config
        default_retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=2.0,
            backoff_multiplier=2.0
        )

        # Service-specific configurations
        service_configs = self.config.get('services', {})
        for service_name, service_config in service_configs.items():
            # Circuit breaker config
            cb_config = service_config.get('circuit_breaker', {})
            self.circuit_breakers[service_name] = CircuitBreaker(
                CircuitBreakerConfig(**cb_config) if cb_config else default_cb_config,
                service_name
            )

            # Retry config
            retry_config = service_config.get('retry', {})
            self.retry_configs[service_name] = RetryConfig(**retry_config) if retry_config else default_retry_config

    async def _initialize_service_proxy(self, service_name: str):
        """Initialize proxy components for a service."""
        service_config = self.config.get('services', {}).get(service_name, {})

        # Create load balancer with appropriate strategy
        strategy_name = service_config.get('load_balancing_strategy', 'epyc_numa_aware')
        strategy = LoadBalancingStrategy(strategy_name)

        load_balancer = EPYCLoadBalancer(service_name, strategy)
        await load_balancer.initialize()

        # Add static endpoints if configured
        static_endpoints = service_config.get('static_endpoints', [])
        for ep_config in static_endpoints:
            endpoint = ServiceEndpoint(
                host=ep_config['host'],
                port=ep_config['port'],
                weight=ep_config.get('weight', 100),
                numa_node=ep_config.get('numa_node'),
                ccx_affinity=ep_config.get('ccx_affinity')
            )
            load_balancer.add_endpoint(endpoint)

        self.load_balancers[service_name] = load_balancer

# Global service mesh proxy instance
service_mesh_proxy: Optional[ServiceMeshProxy] = None

async def initialize_service_mesh(config: Dict[str, Any]) -> ServiceMeshProxy:
    """Initialize global service mesh proxy."""
    global service_mesh_proxy
    service_mesh_proxy = ServiceMeshProxy(config)
    await service_mesh_proxy.initialize()
    return service_mesh_proxy

async def get_service_mesh() -> Optional[ServiceMeshProxy]:
    """Get global service mesh proxy instance."""
    return service_mesh_proxy
