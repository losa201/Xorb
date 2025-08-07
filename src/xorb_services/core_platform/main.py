#!/usr/bin/env python3
"""
XORB Unified Core Platform
Consolidated API Gateway + Authentication + Service Mesh
Optimized for AMD EPYC deployment
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

import aioredis
import jwt
from aiohttp import web, web_request, web_response
from aiohttp.web_middlewares import middleware
from aiohttp_cors import setup as cors_setup, ResourceOptions
from pydantic import BaseModel, Field
import httpx

# Unified configuration
class PlatformConfig:
    # EPYC Optimization
    NUMA_NODES = 2
    CPU_CORES = 16
    MEMORY_GB = 32
    
    # Security
    JWT_SECRET = os.getenv("JWT_SECRET")
    if not JWT_SECRET:
        raise ValueError("JWT_SECRET environment variable must be set")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = 24
    
    # Rate limiting (per EPYC capabilities)
    RATE_LIMIT_REQUESTS = 1000  # per minute
    RATE_LIMIT_BURST = 50       # burst capacity
    
    # Service mesh
    SERVICES = {
        "intelligence-engine": "http://xorb-intelligence-engine:8001",
        "execution-engine": "http://xorb-execution-engine:8002",
        "ml-defense": "http://xorb-ml-defense:8003",
    }

# Unified data models
class UnifiedUser(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    email: str
    roles: List[str] = Field(default_factory=list)
    api_keys: List[str] = Field(default_factory=list)
    permissions: Dict[str, bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)

class UnifiedSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    ip_address: str
    user_agent: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    permissions: Dict[str, bool] = Field(default_factory=dict)

class APIKeyModel(BaseModel):
    key_id: str = Field(default_factory=lambda: str(uuid4()))
    key_hash: str
    user_id: str
    name: str
    scopes: List[str] = Field(default_factory=list)
    rate_limit: int = 1000
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None

# Unified Authentication and Authorization Service
class UnifiedAuthService:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
        
    async def create_api_key(self, user_id: str, name: str, scopes: List[str]) -> tuple[str, APIKeyModel]:
        """Create a new API key for a user."""
        # Generate secure API key
        raw_key = f"xorb_{uuid4().hex}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        api_key = APIKeyModel(
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            scopes=scopes,
            rate_limit=PlatformConfig.RATE_LIMIT_REQUESTS
        )
        
        # Store in Redis
        await self.redis.setex(
            f"api_key:{key_hash}",
            86400 * 30,  # 30 days
            api_key.json()
        )
        
        return raw_key, api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[APIKeyModel]:
        """Validate API key and return key model."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        key_data = await self.redis.get(f"api_key:{key_hash}")
        if not key_data:
            return None
        
        key_model = APIKeyModel.parse_raw(key_data)
        
        # Update last used
        key_model.last_used = datetime.utcnow()
        await self.redis.setex(
            f"api_key:{key_hash}",
            86400 * 30,
            key_model.json()
        )
        
        return key_model
    
    async def create_jwt_token(self, user: UnifiedUser) -> str:
        """Create JWT token for user."""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": datetime.utcnow() + timedelta(hours=PlatformConfig.JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow(),
            "iss": "xorb-platform"
        }
        
        return jwt.encode(payload, PlatformConfig.JWT_SECRET, algorithm=PlatformConfig.JWT_ALGORITHM)
    
    async def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(
                token, 
                PlatformConfig.JWT_SECRET, 
                algorithms=[PlatformConfig.JWT_ALGORITHM]
            )
            
            # Check if token is in blacklist
            is_blacklisted = await self.redis.get(f"blacklist:{token}")
            if is_blacklisted:
                return None
                
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None
    
    async def check_permissions(self, user_perms: Dict[str, bool], required_perms: List[str]) -> bool:
        """Check if user has required permissions."""
        return all(user_perms.get(perm, False) for perm in required_perms)

# Unified Rate Limiting Service
class UnifiedRateLimiter:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def check_rate_limit(self, key: str, limit: int = PlatformConfig.RATE_LIMIT_REQUESTS, window: int = 60) -> tuple[bool, int]:
        """Check rate limit for a key. Returns (allowed, remaining)."""
        current_time = int(time.time())
        window_start = current_time - window
        
        # Use sliding window counter
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
        pipe.zcard(f"rate_limit:{key}")
        pipe.zadd(f"rate_limit:{key}", {str(current_time): current_time})
        pipe.expire(f"rate_limit:{key}", window)
        
        results = await pipe.execute()
        current_requests = results[1]
        
        if current_requests >= limit:
            return False, 0
        
        return True, limit - current_requests - 1

# Unified Service Discovery and Health Check
class UnifiedServiceMesh:
    def __init__(self):
        self.services = PlatformConfig.SERVICES.copy()
        self.health_status = {}
        self.circuit_breakers = {}
        
    async def health_check(self, service_name: str) -> bool:
        """Check health of a service."""
        if service_name not in self.services:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.services[service_name]}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def proxy_request(self, service_name: str, path: str, method: str, **kwargs) -> httpx.Response:
        """Proxy request to service with circuit breaker."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        url = f"{self.services[service_name]}{path}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(method, url, **kwargs)
            return response

# Unified Core Platform Application
class UnifiedCorePlatform:
    def __init__(self):
        self.app = web.Application(middlewares=[
            self.auth_middleware,
            self.rate_limit_middleware,
            self.metrics_middleware,
            self.error_middleware
        ])
        self.redis = None
        self.auth_service = None
        self.rate_limiter = None
        self.service_mesh = None
        self.logger = logging.getLogger(__name__)
        
    async def init_services(self):
        """Initialize all services."""
        # Redis connection
        self.redis = aioredis.from_url("redis://redis:6379")
        
        # Initialize services
        self.auth_service = UnifiedAuthService(self.redis)
        self.rate_limiter = UnifiedRateLimiter(self.redis)
        self.service_mesh = UnifiedServiceMesh()
        
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
        
    def setup_routes(self):
        """Setup all routes."""
        # Authentication routes
        self.app.router.add_post('/auth/login', self.login)
        self.app.router.add_post('/auth/logout', self.logout)
        self.app.router.add_post('/auth/refresh', self.refresh_token)
        self.app.router.add_post('/auth/api-key', self.create_api_key)
        
        # Service proxy routes
        self.app.router.add_route('*', '/api/intelligence/{path:.*}', 
                                 lambda req: self.proxy_to_service(req, 'intelligence-engine'))
        self.app.router.add_route('*', '/api/execution/{path:.*}', 
                                 lambda req: self.proxy_to_service(req, 'execution-engine'))
        self.app.router.add_route('*', '/api/defense/{path:.*}', 
                                 lambda req: self.proxy_to_service(req, 'ml-defense'))
        
        # Platform routes
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.get_metrics)
        self.app.router.add_get('/status', self.get_status)
    
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
        """Metrics collection middleware."""
        start_time = time.time()
        
        try:
            response = await handler(request)
            duration = time.time() - start_time
            
            # Record metrics in Redis
            await self.redis.hincrby('metrics:requests', request.path, 1)
            await self.redis.hset('metrics:response_times', request.path, duration)
            
            return response
        except Exception as e:
            duration = time.time() - start_time
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
    
    async def proxy_to_service(self, request: web_request.Request, service_name: str):
        """Proxy request to backend service."""
        path = request.match_info.get('path', '')
        
        try:
            # Extract request data
            headers = dict(request.headers)
            headers['X-User-ID'] = request.get('user', {}).get('user_id', '')
            headers['X-Request-ID'] = str(uuid4())
            
            # Proxy the request
            response = await self.service_mesh.proxy_request(
                service_name=service_name,
                path=f"/{path}",
                method=request.method,
                headers=headers,
                params=dict(request.query),
                json=await request.json() if request.content_type == 'application/json' else None
            )
            
            # Return response
            return web.Response(
                text=response.text,
                status=response.status_code,
                headers=dict(response.headers)
            )
            
        except Exception as e:
            self.logger.exception(f"Error proxying to {service_name}")
            return web.json_response({
                'error': f'Service {service_name} unavailable',
                'service': service_name
            }, status=503)
    
    async def login(self, request: web_request.Request):
        """User login endpoint."""
        data = await request.json()
        username = data.get('username')
        password = data.get('password')
        
        # TODO: Implement actual user authentication
        # For now, create mock user
        user = UnifiedUser(
            username=username,
            email=f"{username}@xorb.local",
            roles=['user'],
            permissions={
                'campaigns:read': True,
                'campaigns:write': True,
                'agents:read': True,
                'agents:write': True
            }
        )
        
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
            user_id=user['user_id'],
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
    
    async def get_metrics(self, request: web_request.Request):
        """Get platform metrics."""
        metrics = {
            'requests': await self.redis.hgetall('metrics:requests'),
            'response_times': await self.redis.hgetall('metrics:response_times'),
            'errors': await self.redis.hgetall('metrics:errors'),
            'active_users': await self.redis.scard('active_users'),
        }
        return web.json_response(metrics)
    
    async def get_status(self, request: web_request.Request):
        """Get platform status."""
        return web.json_response({
            'platform': 'XORB Unified Core',
            'version': '1.0.0',
            'deployment': 'AMD EPYC Optimized',
            'services': await self.get_service_health(),
            'uptime': time.time(),  # TODO: Track actual uptime
        })

# Main application factory
async def create_app():
    """Create and configure the unified platform application."""
    platform = UnifiedCorePlatform()
    await platform.init_services()
    return platform.app

# Entry point for development
if __name__ == '__main__':
    import sys
    import os
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    async def main():
        app = await create_app()
        web.run_app(app, host='0.0.0.0', port=8000)
    
    asyncio.run(main())