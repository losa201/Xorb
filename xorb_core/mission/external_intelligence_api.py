#!/usr/bin/env python3
"""
XORB External Intelligence API v9.0 - Secure External Service Exposure

This module provides secure external-facing APIs for XORB intelligence services:
- RESTful APIs with comprehensive authentication and authorization
- Real-time intelligence streaming via WebSocket connections
- Secure data sharing with granular access controls
- API marketplace integration and subscription management
"""

import asyncio
import json
import logging
import uuid
import hashlib
import hmac
import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

import structlog
try:
    import aiohttp
    from aiohttp import web, WSMsgType
    from aiohttp_cors import setup as cors_setup, ResourceOptions
except ImportError:
    aiohttp = None
    web = None
    WSMsgType = None
    cors_setup = None
    ResourceOptions = None
from prometheus_client import Counter, Histogram, Gauge
try:
    import redis.asyncio as redis
except ImportError:
    redis = None

# Internal XORB imports
from ..autonomous.intelligent_orchestrator import IntelligentOrchestrator
from ..autonomous.episodic_memory_system import EpisodicMemorySystem
from .autonomous_bounty_engagement import AutonomousBountyEngagement
from .compliance_platform_integration import CompliancePlatformIntegration
from .adaptive_mission_engine import AdaptiveMissionEngine


class APIAccessLevel(Enum):
    """API access levels"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    INTERNAL = "internal"


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class APIEndpointType(Enum):
    """Types of API endpoints"""
    REST = "rest"
    WEBSOCKET = "websocket"
    GRAPHQL = "graphql"
    STREAMING = "streaming"


class SubscriptionTier(Enum):
    """API subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


@dataclass
class APICredentials:
    """API client credentials"""
    client_id: str
    client_secret: str
    api_key: str
    
    # Client information
    client_name: str
    organization: str
    contact_email: str
    
    # Access control
    access_level: APIAccessLevel
    subscription_tier: SubscriptionTier
    permitted_endpoints: List[str]
    rate_limits: Dict[str, int]
    
    # Security settings
    ip_whitelist: List[str] = None
    requires_mfa: bool = False
    token_lifetime: int = 3600  # seconds
    
    # Usage tracking
    created_at: datetime = None
    last_used: Optional[datetime] = None
    total_requests: int = 0
    
    # Status
    active: bool = True
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.ip_whitelist is None:
            self.ip_whitelist = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class APIEndpoint:
    """API endpoint definition"""
    endpoint_id: str
    path: str
    method: str
    endpoint_type: APIEndpointType
    
    # Endpoint configuration
    handler_function: str
    required_access_level: APIAccessLevel
    data_classification: DataClassification
    rate_limit: int  # requests per minute
    
    # Documentation
    name: str
    description: str
    parameters: Dict[str, Any]
    response_schema: Dict[str, Any]
    example_request: Dict[str, Any]
    example_response: Dict[str, Any]
    
    # Security and compliance
    requires_encryption: bool = True
    audit_logs: bool = True
    pii_data: bool = False
    
    # Performance
    cache_ttl: int = 0  # seconds, 0 = no cache
    timeout: int = 30   # seconds
    
    # Monitoring
    enabled: bool = True
    monitoring_level: str = "standard"  # minimal, standard, detailed
    
    # Usage statistics
    total_calls: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0


@dataclass
class APIRequest:
    """API request tracking"""
    request_id: str
    client_id: str
    endpoint_id: str
    
    # Request details
    method: str
    path: str
    query_params: Dict[str, Any]
    headers: Dict[str, str]
    body: Optional[Dict[str, Any]]
    
    # Request metadata
    timestamp: datetime
    client_ip: str
    user_agent: str
    
    # Processing details
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    response_status: Optional[int] = None
    response_size: Optional[int] = None
    
    # Security and compliance
    authenticated: bool = False
    authorized: bool = False
    data_classification: Optional[DataClassification] = None
    
    # Rate limiting
    rate_limit_applied: bool = False
    rate_limit_remaining: int = 0


@dataclass
class IntelligenceProduct:
    """Intelligence product for external consumption"""
    product_id: str
    name: str
    description: str
    
    # Product content
    data_type: str
    schema_version: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    
    # Classification and access
    classification: DataClassification
    access_level: APIAccessLevel
    retention_period: timedelta
    
    # Provenance and quality
    source_systems: List[str]
    confidence_score: float
    freshness: timedelta  # Age of data
    accuracy_estimate: float
    
    # Production metadata
    generated_at: datetime
    expires_at: Optional[datetime] = None
    version: str = "1.0"
    
    # Distribution tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class ExternalIntelligenceAPI:
    """
    External Intelligence API Service
    
    Provides secure external access to XORB intelligence capabilities:
    - RESTful API endpoints with comprehensive authentication
    - Real-time intelligence streaming
    - Secure data sharing with access controls
    - API marketplace and subscription management
    """
    
    def __init__(self, orchestrator: IntelligentOrchestrator):
        self.orchestrator = orchestrator
        self.logger = structlog.get_logger("xorb.external_api")
        
        # API management
        self.api_app: Optional[web.Application] = None
        self.api_endpoints: Dict[str, APIEndpoint] = {}
        self.client_credentials: Dict[str, APICredentials] = {}
        
        # Real-time connections
        self.websocket_connections: Dict[str, web.WebSocketResponse] = {}
        self.streaming_clients: Dict[str, Dict[str, Any]] = {}
        
        # Intelligence products
        self.intelligence_products: Dict[str, IntelligenceProduct] = {}
        self.product_catalog: Dict[str, Dict[str, Any]] = {}
        
        # Security and authentication
        self.jwt_secret: str = self._generate_jwt_secret()
        self.redis_client: Optional[redis.Redis] = None
        self.rate_limiters: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # API configuration
        self.api_host = "0.0.0.0"
        self.api_port = 8443
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.ssl_context = None  # Will be configured
        
        # Intelligence integration
        self.bounty_engagement: Optional[AutonomousBountyEngagement] = None
        self.compliance_integration: Optional[CompliancePlatformIntegration] = None
        self.mission_engine: Optional[AdaptiveMissionEngine] = None
        
        # Metrics and monitoring
        self.api_metrics = self._initialize_api_metrics()
        
        # Audit and compliance
        self.audit_trail: List[Dict[str, Any]] = []
        self.access_logs: List[APIRequest] = []
    
    def _initialize_api_metrics(self) -> Dict[str, Any]:
        """Initialize API metrics"""
        return {
            'api_requests_total': Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status']),
            'api_request_duration': Histogram('api_request_duration_seconds', 'API request duration', ['endpoint', 'method']),
            'active_connections': Gauge('api_active_connections', 'Active API connections', ['connection_type']),
            'authentication_attempts': Counter('api_authentication_attempts_total', 'Authentication attempts', ['result']),
            'rate_limit_exceeded': Counter('api_rate_limit_exceeded_total', 'Rate limit exceeded', ['client_id']),
            'intelligence_products_served': Counter('intelligence_products_served_total', 'Intelligence products served', ['product_type']),
            'data_volume_served': Counter('api_data_volume_bytes_total', 'Data volume served', ['endpoint']),
            'api_errors': Counter('api_errors_total', 'API errors', ['endpoint', 'error_type'])
        }
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT signing secret"""
        return hashlib.sha256(f"xorb_api_{uuid.uuid4()}_{time.time()}".encode()).hexdigest()
    
    async def start_external_api(self):
        """Start the external intelligence API service"""
        self.logger.info("🌐 Starting External Intelligence API")
        
        # Initialize Redis for session management
        await self._initialize_redis()
        
        # Set up API application
        await self._setup_api_application()
        
        # Initialize intelligence integrations
        await self._initialize_intelligence_integrations()
        
        # Configure security
        await self._configure_security()
        
        # Start API server
        await self._start_api_server()
        
        # Start background processes
        asyncio.create_task(self._intelligence_product_generator())
        asyncio.create_task(self._real_time_intelligence_streamer())
        asyncio.create_task(self._api_monitoring_loop())
        asyncio.create_task(self._cleanup_expired_sessions())
        
        self.logger.info(f"🚀 External API active on https://{self.api_host}:{self.api_port}")
    
    async def _setup_api_application(self):
        """Set up the API web application"""
        self.api_app = web.Application()
        
        # Configure CORS
        cors = cors_setup(self.api_app, defaults={
            "*": ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*",
            )
        })
        
        # Set up middleware
        self.api_app.middlewares.append(self._auth_middleware)
        self.api_app.middlewares.append(self._rate_limit_middleware)
        self.api_app.middlewares.append(self._audit_middleware)
        self.api_app.middlewares.append(self._error_handling_middleware)
        
        # Register API endpoints
        await self._register_api_endpoints()
        
        # Configure static documentation
        self.api_app.router.add_static('/docs', 'static/api_docs', name='api_docs')
    
    async def _register_api_endpoints(self):
        """Register all API endpoints"""
        # Intelligence endpoints
        self._register_endpoint('GET', '/api/v1/intelligence/threats', self._get_threat_intelligence)
        self._register_endpoint('GET', '/api/v1/intelligence/vulnerabilities', self._get_vulnerability_intelligence)
        self._register_endpoint('GET', '/api/v1/intelligence/assets', self._get_asset_intelligence)
        self._register_endpoint('GET', '/api/v1/intelligence/campaigns', self._get_campaign_intelligence)
        
        # Mission endpoints
        self._register_endpoint('POST', '/api/v1/missions', self._create_mission)
        self._register_endpoint('GET', '/api/v1/missions/{mission_id}', self._get_mission_status)
        self._register_endpoint('PUT', '/api/v1/missions/{mission_id}/control', self._control_mission)
        
        # Compliance endpoints
        self._register_endpoint('GET', '/api/v1/compliance/status', self._get_compliance_status)
        self._register_endpoint('POST', '/api/v1/compliance/assessment', self._initiate_compliance_assessment)
        self._register_endpoint('GET', '/api/v1/compliance/evidence/{evidence_id}', self._get_compliance_evidence)
        
        # Bounty platform endpoints
        self._register_endpoint('GET', '/api/v1/bounty/programs', self._get_bounty_programs)
        self._register_endpoint('GET', '/api/v1/bounty/submissions', self._get_bounty_submissions)
        self._register_endpoint('POST', '/api/v1/bounty/submit', self._submit_vulnerability)
        
        # Real-time endpoints
        self._register_endpoint('GET', '/api/v1/stream/intelligence', self._intelligence_websocket)
        self._register_endpoint('GET', '/api/v1/stream/missions', self._mission_websocket)
        
        # System endpoints
        self._register_endpoint('GET', '/api/v1/system/health', self._system_health)
        self._register_endpoint('GET', '/api/v1/system/metrics', self._system_metrics)
        
        # Authentication endpoints
        self._register_endpoint('POST', '/api/v1/auth/token', self._get_access_token)
        self._register_endpoint('POST', '/api/v1/auth/refresh', self._refresh_token)
        self._register_endpoint('DELETE', '/api/v1/auth/logout', self._logout)
    
    def _register_endpoint(self, method: str, path: str, handler: Callable):
        """Register an API endpoint"""
        endpoint_id = f"{method}:{path}"
        
        # Create endpoint definition
        endpoint = APIEndpoint(
            endpoint_id=endpoint_id,
            path=path,
            method=method,
            endpoint_type=APIEndpointType.REST,
            handler_function=handler.__name__,
            required_access_level=APIAccessLevel.AUTHENTICATED,
            data_classification=DataClassification.INTERNAL,
            rate_limit=100,  # Default rate limit
            name=handler.__name__.replace('_', ' ').title(),
            description=handler.__doc__ or f"{method} {path}",
            parameters={},
            response_schema={},
            example_request={},
            example_response={}
        )
        
        self.api_endpoints[endpoint_id] = endpoint
        
        # Add route to application
        if method == 'GET':
            self.api_app.router.add_get(path, handler)
        elif method == 'POST':
            self.api_app.router.add_post(path, handler)
        elif method == 'PUT':
            self.api_app.router.add_put(path, handler)
        elif method == 'DELETE':
            self.api_app.router.add_delete(path, handler)
    
    async def _intelligence_product_generator(self):
        """Generate intelligence products for external consumption"""
        while True:
            try:
                # Generate threat intelligence
                threat_intel = await self._generate_threat_intelligence()
                if threat_intel:
                    self.intelligence_products[threat_intel.product_id] = threat_intel
                
                # Generate vulnerability intelligence
                vuln_intel = await self._generate_vulnerability_intelligence()
                if vuln_intel:
                    self.intelligence_products[vuln_intel.product_id] = vuln_intel
                
                # Generate compliance intelligence
                compliance_intel = await self._generate_compliance_intelligence()
                if compliance_intel:
                    self.intelligence_products[compliance_intel.product_id] = compliance_intel
                
                # Clean up expired products
                await self._cleanup_expired_products()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error("Intelligence product generation error", error=str(e))
                await asyncio.sleep(600)
    
    async def _real_time_intelligence_streamer(self):
        """Stream real-time intelligence to connected clients"""
        while True:
            try:
                # Prepare intelligence updates
                updates = await self._prepare_intelligence_updates()
                
                # Stream to WebSocket clients
                for client_id, ws in list(self.websocket_connections.items()):
                    try:
                        if ws.closed:
                            del self.websocket_connections[client_id]
                            continue
                        
                        # Filter updates based on client access level
                        filtered_updates = await self._filter_updates_for_client(client_id, updates)
                        
                        if filtered_updates:
                            await ws.send_str(json.dumps(filtered_updates))
                    
                    except Exception as e:
                        self.logger.error(f"Failed to stream to client {client_id[:8]}", error=str(e))
                        try:
                            del self.websocket_connections[client_id]
                        except KeyError:
                            pass
                
                await asyncio.sleep(10)  # Every 10 seconds
                
            except Exception as e:
                self.logger.error("Real-time streaming error", error=str(e))
                await asyncio.sleep(30)
    
    # API Handler Methods
    async def _get_threat_intelligence(self, request: web.Request) -> web.Response:
        """Get threat intelligence data"""
        try:
            # Extract query parameters
            threat_types = request.query.getlist('type', [])
            severity_min = request.query.get('severity_min', 'low')
            limit = int(request.query.get('limit', 100))
            
            # Get threat intelligence
            threats = await self._fetch_threat_intelligence(threat_types, severity_min, limit)
            
            return web.json_response({
                'status': 'success',
                'data': threats,
                'metadata': {
                    'count': len(threats),
                    'generated_at': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            self.logger.error("Threat intelligence API error", error=str(e))
            return web.json_response({'status': 'error', 'message': str(e)}, status=500)
    
    async def _create_mission(self, request: web.Request) -> web.Response:
        """Create a new mission"""
        try:
            data = await request.json()
            
            # Validate mission request
            mission_type = data.get('mission_type')
            objectives = data.get('objectives', [])
            constraints = data.get('constraints', {})
            
            if not mission_type or not objectives:
                return web.json_response({
                    'status': 'error',
                    'message': 'mission_type and objectives are required'
                }, status=400)
            
            # Create mission through mission engine
            if self.mission_engine:
                from .adaptive_mission_engine import MissionType
                mission_plan = await self.mission_engine.plan_mission(
                    MissionType(mission_type),
                    objectives,
                    constraints
                )
                
                return web.json_response({
                    'status': 'success',
                    'data': {
                        'mission_id': mission_plan.mission_id,
                        'plan_id': mission_plan.plan_id,
                        'status': mission_plan.status,
                        'objectives_count': len(mission_plan.objectives)
                    }
                })
            else:
                return web.json_response({
                    'status': 'error',
                    'message': 'Mission engine not available'
                }, status=503)
                
        except Exception as e:
            self.logger.error("Create mission API error", error=str(e))
            return web.json_response({'status': 'error', 'message': str(e)}, status=500)
    
    async def _intelligence_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint for real-time intelligence"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_id = str(uuid.uuid4())
        self.websocket_connections[client_id] = ws
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        # Handle client subscription preferences
                        await self._handle_websocket_message(client_id, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({'error': 'Invalid JSON'}))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')
        
        except Exception as e:
            self.logger.error(f"WebSocket error for client {client_id[:8]}", error=str(e))
        
        finally:
            if client_id in self.websocket_connections:
                del self.websocket_connections[client_id]
        
        return ws
    
    async def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API status"""
        return {
            'external_api': {
                'server_status': 'running' if self.api_app else 'stopped',
                'endpoints_registered': len(self.api_endpoints),
                'active_clients': len(self.client_credentials),
                'active_connections': len(self.websocket_connections),
                'intelligence_products': len(self.intelligence_products)
            },
            'endpoint_statistics': {
                endpoint_id: {
                    'path': endpoint.path,
                    'method': endpoint.method,
                    'total_calls': endpoint.total_calls,
                    'average_response_time': endpoint.average_response_time,
                    'error_rate': endpoint.error_rate
                }
                for endpoint_id, endpoint in list(self.api_endpoints.items())[:10]
            },
            'client_summary': {
                tier.value: sum(1 for c in self.client_credentials.values() if c.subscription_tier == tier)
                for tier in SubscriptionTier
            },
            'intelligence_products_summary': {
                'total_products': len(self.intelligence_products),
                'products_by_classification': {
                    classification.value: sum(1 for p in self.intelligence_products.values() if p.classification == classification)
                    for classification in DataClassification
                },
                'average_confidence': np.mean([p.confidence_score for p in self.intelligence_products.values()]) if self.intelligence_products else 0.0
            },
            'api_performance': {
                'total_requests_served': sum(endpoint.total_calls for endpoint in self.api_endpoints.values()),
                'average_response_time': np.mean([endpoint.average_response_time for endpoint in self.api_endpoints.values()]) if self.api_endpoints else 0.0,
                'overall_error_rate': np.mean([endpoint.error_rate for endpoint in self.api_endpoints.values()]) if self.api_endpoints else 0.0
            }
        }
    
    # Placeholder implementations for complex methods
    async def _initialize_redis(self): pass
    async def _initialize_intelligence_integrations(self): pass
    async def _configure_security(self): pass
    async def _start_api_server(self): pass
    async def _api_monitoring_loop(self): pass
    async def _cleanup_expired_sessions(self): pass
    async def _auth_middleware(self, request, handler): return await handler(request)
    async def _rate_limit_middleware(self, request, handler): return await handler(request)
    async def _audit_middleware(self, request, handler): return await handler(request)
    async def _error_handling_middleware(self, request, handler): return await handler(request)
    async def _generate_threat_intelligence(self) -> Optional[IntelligenceProduct]: return None
    async def _generate_vulnerability_intelligence(self) -> Optional[IntelligenceProduct]: return None
    async def _generate_compliance_intelligence(self) -> Optional[IntelligenceProduct]: return None
    async def _cleanup_expired_products(self): pass
    async def _prepare_intelligence_updates(self) -> List[Dict[str, Any]]: return []
    async def _filter_updates_for_client(self, client_id: str, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]: return updates
    async def _fetch_threat_intelligence(self, threat_types: List[str], severity_min: str, limit: int) -> List[Dict[str, Any]]: return []
    async def _handle_websocket_message(self, client_id: str, data: Dict[str, Any]): pass
    async def _get_vulnerability_intelligence(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _get_asset_intelligence(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _get_campaign_intelligence(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _get_mission_status(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _control_mission(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _get_compliance_status(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _initiate_compliance_assessment(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _get_compliance_evidence(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _get_bounty_programs(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _get_bounty_submissions(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _submit_vulnerability(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _mission_websocket(self, request: web.Request) -> web.WebSocketResponse: return web.WebSocketResponse()
    async def _system_health(self, request: web.Request) -> web.Response: return web.json_response({'status': 'healthy'})
    async def _system_metrics(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _get_access_token(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _refresh_token(self, request: web.Request) -> web.Response: return web.json_response({})
    async def _logout(self, request: web.Request) -> web.Response: return web.json_response({})


# Global external API instance
external_intelligence_api = None