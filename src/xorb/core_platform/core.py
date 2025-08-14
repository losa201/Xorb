import asyncio
import logging
import time
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any, Optional

import aioredis
from aiohttp import web, web_request
from aiohttp.web_middlewares import middleware
from aiohttp_cors import setup as cors_setup, ResourceOptions
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from xorb.shared.config import PlatformConfig
from xorb.shared.models import UnifiedUser, UnifiedSession, APIKeyModel
from xorb.database.database import AsyncSessionLocal
from xorb.database.repositories import UserRepository
from xorb.architecture.service_definitions import XORB_ARCHITECTURE
from xorb.architecture.service_mesh import initialize_service_mesh, get_service_mesh
from xorb.architecture.fault_tolerance import initialize_fault_tolerance, get_fault_tolerance
from xorb.architecture.observability import initialize_observability, get_observability, trace
from xorb.architecture.epyc_optimization import initialize_epyc_optimization, get_epyc_optimization, epyc_optimized, WorkloadType
from .auth import UnifiedAuthService
from .rate_limiter import UnifiedRateLimiter

# XORB Pristine Core Platform with Architectural Excellence
class XORBPristineCorePlatform:
    def __init__(self):
        self.app = web.Application(middlewares=[
            self.observability_middleware,
            self.auth_middleware,
            self.rate_limit_middleware,
            self.fault_tolerance_middleware,
            self.epyc_optimization_middleware,
            self.metrics_middleware,
            self.error_middleware
        ])
        self.redis = None
        self.auth_service = None
        self.rate_limiter = None
        self.service_mesh = None
        self.fault_tolerance = None
        self.observability = None
        self.epyc_optimization = None
        self.logger = logging.getLogger(__name__)

        # Enhanced metrics
        self.request_counter = Counter(
            'xorb_core_platform_requests_total',
            'Total requests to core platform',
            ['method', 'endpoint', 'status']
        )
        self.request_duration = Histogram(
            'xorb_core_platform_request_duration_seconds',
            'Request duration for core platform',
            ['method', 'endpoint']
        )
        self.active_users = Gauge(
            'xorb_core_platform_active_users',
            'Number of active users'
        )
        self.service_health = Gauge(
            'xorb_core_platform_service_health',
            'Health status of backend services',
            ['service']
        )

    async def init_services(self):
        """Initialize all pristine architecture services."""
        # Redis connection
        self.redis = aioredis.from_url("redis://redis:6379")

        # Initialize core services
        self.auth_service = UnifiedAuthService(db_session=AsyncSessionLocal(), redis_client=self.redis)
        self.rate_limiter = UnifiedRateLimiter(self.redis)

        # Initialize pristine architecture components
        self.service_mesh = await initialize_service_mesh({
            'services': {
                'intelligence-engine': {
                    'load_balancing_strategy': 'epyc_numa_aware',
                    'static_endpoints': [{'host': 'intelligence-engine', 'port': 8001}],
                    'circuit_breaker': {'failure_threshold': 5, 'timeout_duration': 60}
                },
                'execution-engine': {
                    'load_balancing_strategy': 'ai_workload_optimized',
                    'static_endpoints': [{'host': 'execution-engine', 'port': 8002}],
                    'circuit_breaker': {'failure_threshold': 3, 'timeout_duration': 30}
                }
            }
        })

        self.fault_tolerance = await initialize_fault_tolerance()

        self.observability = await initialize_observability('core-platform', {
            'jaeger_endpoint': 'http://jaeger:14268/api/traces',
            'metrics_port': 9090,
            'webhook_url': 'http://alertmanager:9093/api/v1/alerts'
        })

        self.epyc_optimization = await initialize_epyc_optimization()

        # Setup CORS
        cors = cors_setup(self.app, defaults={
            "*": ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })

        # Register routes
        self.setup_routes()

        self.logger.info("XORB Pristine Core Platform initialized with full architecture stack")

    def setup_routes(self):
        """Setup all pristine architecture routes."""
        # Authentication routes
        self.app.router.add_post('/auth/login', self.login)
        self.app.router.add_post('/auth/logout', self.logout)
        self.app.router.add_post('/auth/refresh', self.refresh_token)
        self.app.router.add_post('/auth/api-key', self.create_api_key)

        # Pristine architecture service proxy routes
        self.app.router.add_route('*', '/api/campaigns/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'campaign-orchestrator'))
        self.app.router.add_route('*', '/api/targets/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'target-registry'))
        self.app.router.add_route('*', '/api/agents/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'agent-lifecycle'))
        self.app.router.add_route('*', '/api/evidence/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'evidence-collector'))
        self.app.router.add_route('*', '/api/vulnerabilities/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'vulnerability-scanner'))
        self.app.router.add_route('*', '/api/exploits/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'exploitation-engine'))
        self.app.router.add_route('*', '/api/stealth/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'stealth-manager'))
        self.app.router.add_route('*', '/api/ai/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'ai-gateway'))
        self.app.router.add_route('*', '/api/threats/{path:.*}',
                                 lambda req: self.proxy_to_service(req, 'threat-intelligence'))

        # Platform management routes
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.prometheus_metrics)
        self.app.router.add_get('/status', self.get_status)
        self.app.router.add_get('/architecture', self.get_architecture_info)
        self.app.router.add_get('/fault-tolerance', self.get_fault_tolerance_status)
        self.app.router.add_get('/epyc-optimization', self.get_epyc_status)

    @middleware
    async def observability_middleware(self, request: web_request.Request, handler):
        """Distributed tracing and observability middleware."""
        if self.observability:
            async with self.observability.tracer.start_span(
                f"core_platform_{request.method}_{request.path}",
                tags={
                    'method': request.method,
                    'path': request.path,
                    'user_agent': request.headers.get('User-Agent', 'unknown')
                }
            ) as span:
                return await handler(request)
        else:
            return await handler(request)

    @middleware
    async def auth_middleware(self, request: web_request.Request, handler):
        """Authentication middleware."""
        # Skip auth for health checks
        if request.path in ['/health', '/metrics']:
            return await handler(request)

        # Try JWT first
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            payload = await self.auth_service.validate_jwt_token(token)
            if payload:
                request['user'] = payload
                return await handler(request)

        # Try API key
        api_key = request.headers.get('X-API-Key') or request.query.get('api_key')
        if api_key:
            key_model = await self.auth_service.validate_api_key(api_key)
            if key_model:
                request['user'] = {
                    'user_id': key_model.user_id,
                    'api_key_id': key_model.key_id,
                    'scopes': key_model.scopes,
                    'auth_type': 'api_key'
                }
                return await handler(request)

        # No valid auth found
        return web.json_response({'error': 'Authentication required'}, status=401)

    @middleware
    async def fault_tolerance_middleware(self, request: web_request.Request, handler):
        """Fault tolerance middleware with circuit breaker protection."""
        if self.fault_tolerance and request.path.startswith('/api/'):
            try:
                return await self.fault_tolerance.execute_with_circuit_breaker(
                    'api_gateway', handler, request
                )
            except Exception as e:
                if 'Circuit breaker' in str(e):
                    return web.json_response({
                        'error': 'Service temporarily unavailable',
                        'circuit_breaker': 'open',
                        'retry_after': 30
                    }, status=503)
                raise
        else:
            return await handler(request)

    @middleware
    async def epyc_optimization_middleware(self, request: web_request.Request, handler):
        """EPYC optimization middleware for workload-aware processing."""
        if self.epyc_optimization:
            # Determine workload type based on request
            workload_type = WorkloadType.IO_INTENSIVE  # Default for API requests

            if request.path.startswith('/api/ai/'):
                workload_type = WorkloadType.AI_INFERENCE
            elif request.path.startswith('/api/vulnerabilities/'):
                workload_type = WorkloadType.CPU_INTENSIVE
            elif request.path.startswith('/metrics'):
                workload_type = WorkloadType.LATENCY_SENSITIVE

            @epyc_optimized(workload_type)
            async def optimized_handler():
                return await handler(request)

            return await optimized_handler()
        else:
            return await handler(request)

    @middleware
    async def rate_limit_middleware(self, request: web_request.Request, handler):
        """Rate limiting middleware."""
        # Extract rate limit key
        if 'user' in request:
            key = f"user:{request['user'].get('user_id', 'unknown')}"
        else:
            key = f"ip:{request.remote}"

        # Check rate limit
        allowed, remaining = await self.rate_limiter.check_rate_limit(key)

        if not allowed:
            return web.json_response({'error': 'Rate limit exceeded'}, status=429)

        response = await handler(request)
        response.headers['X-RateLimit-Remaining'] = str(remaining)
        return response

    @middleware
    async def metrics_middleware(self, request: web_request.Request, handler):
        """Enhanced metrics collection middleware."""
        start_time = time.time()

        try:
            response = await handler(request)
            duration = time.time() - start_time

            # Record Prometheus metrics
            self.request_counter.labels(
                method=request.method,
                endpoint=request.path,
                status=str(response.status)
            ).inc()

            self.request_duration.labels(
                method=request.method,
                endpoint=request.path
            ).observe(duration)

            # Legacy Redis metrics for backward compatibility
            await self.redis.hincrby('metrics:requests', request.path, 1)
            await self.redis.hset('metrics:response_times', request.path, duration)

            return response
        except Exception as e:
            duration = time.time() - start_time

            self.request_counter.labels(
                method=request.method,
                endpoint=request.path,
                status='500'
            ).inc()

            await self.redis.hincrby('metrics:errors', request.path, 1)
            raise

    @middleware
    async def error_middleware(self, request: web_request.Request, handler):
        """Error handling middleware."""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            self.logger.exception(f"Unhandled error in {request.path}")
            return web.json_response({
                'error': 'Internal server error',
                'request_id': str(uuid4())
            }, status=500)

    @trace("proxy_to_service")
    async def proxy_to_service(self, request: web_request.Request, service_name: str):
        """Enhanced proxy with service mesh integration."""
        path = request.match_info.get('path', '')

        try:
            # Extract request data with enhanced headers
            headers = dict(request.headers)
            headers['X-User-ID'] = request.get('user', {}).get('user_id', '')
            headers['X-Request-ID'] = str(uuid4())
            headers['X-Service-Tier'] = self._get_service_tier(service_name)
            headers['X-EPYC-Optimized'] = 'true'

            # Get request context for intelligent routing
            request_context = {
                'numa_preference': 0 if service_name in ['campaign-orchestrator', 'target-registry'] else 1,
                'workload_type': self._determine_workload_type(service_name),
                'model_type': 'llm' if service_name == 'ai-gateway' else 'unknown',
                'complexity': 'cognitive' if 'ai' in service_name else 'standard'
            }

            # Proxy via service mesh with intelligent routing
            response = await self.service_mesh.proxy_request(
                source_service='core-platform',
                target_service=service_name,
                method=request.method,
                path=f"/{path}",
                headers=headers,
                data=await request.read() if request.body_exists else None,
                request_context=request_context
            )

            return web.Response(
                body=await response.read(),
                status=response.status,
                headers=dict(response.headers)
            )

        except Exception as e:
            self.logger.exception(f"Error proxying to {service_name}")

            # Update service health metrics
            self.service_health.labels(service=service_name).set(0)

            return web.json_response({
                'error': f'Service {service_name} unavailable',
                'service': service_name,
                'service_tier': self._get_service_tier(service_name),
                'retry_after': 30
            }, status=503)

    def _get_service_tier(self, service_name: str) -> str:
        """Get service tier for given service name."""
        service_def = XORB_ARCHITECTURE.services.get(service_name)
        return service_def.tier.value if service_def else 'unknown'

    def _determine_workload_type(self, service_name: str) -> str:
        """Determine workload type for intelligent routing."""
        workload_mapping = {
            'ai-gateway': 'ai_inference',
            'vulnerability-scanner': 'cpu_intensive',
            'exploitation-engine': 'cpu_intensive',
            'campaign-orchestrator': 'balanced',
            'target-registry': 'memory_intensive',
            'evidence-collector': 'io_intensive'
        }
        return workload_mapping.get(service_name, 'balanced')

    async def login(self, request: web_request.Request):
        """User login endpoint."""
        data = await request.json()
        username = data.get('username')
        password = data.get('password')

        user = await self.auth_service.user_repo.get_user_by_username(username)
        if not user or not self.auth_service.verify_password(password, user.password):
            return web.json_response({'error': 'Invalid credentials'}, status=401)

        token = await self.auth_service.create_jwt_token(user)

        return web.json_response({
            'token': token,
            'user': user.dict(),
            'expires_in': PlatformConfig.JWT_EXPIRY_HOURS * 3600
        })

    async def create_api_key(self, request: web_request.Request):
        """Create API key endpoint."""
        data = await request.json()
        user = request['user']

        api_key, key_model = await self.auth_service.create_api_key(
            user_id=user.id,
            name=data.get('name', 'Default API Key'),
            scopes=data.get('scopes', ['read', 'write'])
        )

        return web.json_response({
            'api_key': api_key,
            'key_id': key_model.key_id,
            'scopes': key_model.scopes,
            'rate_limit': key_model.rate_limit
        })

    async def health_check(self, request: web_request.Request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'services': await self.get_service_health()
        })

    async def get_service_health(self):
        """Get health status of all services."""
        health_status = {}
        for service_name in PlatformConfig.SERVICES:
            health_status[service_name] = await self.service_mesh.health_check(service_name)
        return health_status

    async def prometheus_metrics(self, request: web_request.Request):
        """Prometheus metrics endpoint."""
        metrics_data = generate_latest()
        return web.Response(text=metrics_data.decode('utf-8'), content_type='text/plain')

    async def get_metrics(self, request: web_request.Request):
        """Enhanced platform metrics with architecture insights."""
        base_metrics = {
            'requests': await self.redis.hgetall('metrics:requests'),
            'response_times': await self.redis.hgetall('metrics:response_times'),
            'errors': await self.redis.hgetall('metrics:errors'),
            'active_users': await self.redis.scard('active_users'),
        }

        # Add architecture-specific metrics
        if self.observability:
            observability_stats = await self.observability.get_usage_statistics() if hasattr(self.observability, 'get_usage_statistics') else {}
            base_metrics['observability'] = observability_stats

        if self.fault_tolerance:
            fault_tolerance_stats = self.fault_tolerance.get_usage_statistics() if hasattr(self.fault_tolerance, 'get_usage_statistics') else {}
            base_metrics['fault_tolerance'] = fault_tolerance_stats

        if self.epyc_optimization:
            epyc_stats = {
                'topology': self.epyc_optimization.topology.__dict__ if self.epyc_optimization.topology else {},
                'optimization_active': self.epyc_optimization.optimization_active
            }
            base_metrics['epyc_optimization'] = epyc_stats

        return web.json_response(base_metrics)

    async def get_status(self, request: web_request.Request):
        """Enhanced platform status with architecture information."""
        service_health = await self.get_service_health()

        # Calculate overall health score
        healthy_services = sum(1 for status in service_health.values() if status.get('healthy', False))
        total_services = len(service_health)
        health_percentage = (healthy_services / total_services * 100) if total_services > 0 else 0

        return web.json_response({
            'platform': 'XORB Pristine Core Platform',
            'version': '2.0.0',
            'architecture': 'Pristine Microservices with EPYC Optimization',
            'deployment': 'AMD EPYC Optimized with Service Mesh',
            'health_score': f"{health_percentage:.1f}%",
            'services': service_health,
            'architecture_components': {
                'service_mesh': self.service_mesh is not None,
                'fault_tolerance': self.fault_tolerance is not None,
                'observability': self.observability is not None,
                'epyc_optimization': self.epyc_optimization is not None
            },
            'service_tiers': {
                'core': len(XORB_ARCHITECTURE.get_services_by_tier('core')),
                'domain': len(XORB_ARCHITECTURE.get_services_by_tier('domain')),
                'platform': len(XORB_ARCHITECTURE.get_services_by_tier('platform')),
                'edge': len(XORB_ARCHITECTURE.get_services_by_tier('edge'))
            },
            'uptime': time.time(),
            'numa_nodes': self.epyc_optimization.topology.numa_nodes if self.epyc_optimization and self.epyc_optimization.topology else 2,
            'ccx_count': self.epyc_optimization.topology.ccx_count if self.epyc_optimization and self.epyc_optimization.topology else 8
        })

    async def get_architecture_info(self, request: web_request.Request):
        """Get detailed architecture information."""
        architecture_issues = XORB_ARCHITECTURE.validate_architecture()

        return web.json_response({
            'architecture_type': 'pristine_microservices',
            'total_services': len(XORB_ARCHITECTURE.services),
            'service_tiers': {
                tier.value: len(XORB_ARCHITECTURE.get_services_by_tier(tier))
                for tier in ['core', 'domain', 'platform', 'edge']
            },
            'service_mesh': {
                'enabled': self.service_mesh is not None,
                'provider': 'istio',
                'features': ['intelligent_routing', 'circuit_breakers', 'load_balancing']
            },
            'fault_tolerance': {
                'patterns': ['circuit_breaker', 'bulkhead', 'timeout', 'retry'],
                'enabled': self.fault_tolerance is not None
            },
            'epyc_optimization': {
                'numa_aware': True,
                'ccx_optimized': True,
                'thermal_management': True,
                'enabled': self.epyc_optimization is not None
            },
            'validation_issues': architecture_issues,
            'deployment_topology': XORB_ARCHITECTURE.deployment_topology
        })

    async def get_fault_tolerance_status(self, request: web_request.Request):
        """Get fault tolerance system status."""
        if not self.fault_tolerance:
            return web.json_response({'error': 'Fault tolerance not initialized'}, status=503)

        return web.json_response({
            'bulkheads': list(self.fault_tolerance.bulkheads.keys()),
            'circuit_breakers': list(self.fault_tolerance.circuit_breakers.keys()),
            'active': True,
            'patterns_available': ['circuit_breaker', 'bulkhead', 'timeout', 'retry', 'rate_limiter']
        })

    async def get_epyc_status(self, request: web_request.Request):
        """Get EPYC optimization status."""
        if not self.epyc_optimization:
            return web.json_response({'error': 'EPYC optimization not initialized'}, status=503)

        topology = self.epyc_optimization.topology

        return web.json_response({
            'optimization_active': self.epyc_optimization.optimization_active,
            'topology': {
                'total_cores': topology.total_cores,
                'numa_nodes': topology.numa_nodes,
                'ccx_count': topology.ccx_count,
                'cores_per_ccx': topology.cores_per_ccx,
                'base_frequency': topology.base_frequency,
                'boost_frequency': topology.boost_frequency
            },
            'workload_types': [wt.value for wt in WorkloadType],
            'thermal_management': True,
            'cache_optimization': True
        })
