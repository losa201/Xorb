#!/usr/bin/env python3
"""
XORB Enterprise API Gateway
Comprehensive API gateway with authentication, rate limiting, and security features
"""

import os
import sys
import json
import time
import asyncio
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import jwt
import bcrypt
import redis
from aiohttp import web, web_request, web_response, ClientSession
from aiohttp_cors import setup as cors_setup, ResourceOptions
import aiofiles
import ssl
from cryptography.fernet import Fernet
import ipaddress
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XORBAPIGateway:
    """Enterprise API Gateway with advanced security and authentication"""
    
    def __init__(self, config_path: str = "config/api_gateway_config.json"):
        self.config_path = config_path
        self.config = self.load_configuration()
        
        # Initialize components
        self.redis_client = None
        self.session_store = {}
        self.rate_limiters = defaultdict(list)
        self.blocked_ips = set()
        
        # Security components
        self.jwt_secret = self.config['jwt']['secret_key']
        self.fernet_key = Fernet.generate_key()
        self.cipher = Fernet(self.fernet_key)
        
        # API routes and services
        self.service_registry = {}
        self.api_keys = {}
        
        # Metrics and monitoring
        self.metrics = {
            'requests_total': 0,
            'requests_by_endpoint': defaultdict(int),
            'requests_by_user': defaultdict(int),
            'auth_failures': 0,
            'rate_limit_hits': 0,
            'blocked_requests': 0,
            'response_times': []
        }
        
        # Initialize web application
        self.app = web.Application(middlewares=[
            self.cors_middleware,
            self.security_middleware,
            self.auth_middleware,
            self.rate_limit_middleware,
            self.logging_middleware,
            self.metrics_middleware
        ])
        
        self.setup_routes()
        
    def load_configuration(self) -> Dict:
        """Load API gateway configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return self.get_default_configuration()
    
    def get_default_configuration(self) -> Dict:
        """Get default API gateway configuration"""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "ssl_enabled": True,
                "ssl_cert": "/etc/ssl/certs/api.xorb.security.crt",
                "ssl_key": "/etc/ssl/private/api.xorb.security.key"
            },
            "jwt": {
                "secret_key": secrets.token_urlsafe(32),
                "algorithm": "HS256",
                "expiration_hours": 24,
                "refresh_expiration_days": 30
            },
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "burst_limit": 10,
                "redis_url": "redis://localhost:6379/1"
            },
            "security": {
                "cors_origins": ["https://xorb.security", "https://verteidiq.com"],
                "api_key_header": "X-API-Key",
                "max_request_size": 10485760,  # 10MB
                "ip_whitelist": [],
                "ip_blacklist": [],
                "require_https": True,
                "csrf_protection": True
            },
            "authentication": {
                "providers": ["jwt", "api_key", "oauth2"],
                "password_min_length": 12,
                "password_complexity": True,
                "mfa_required": False,
                "session_timeout": 3600
            },
            "services": {
                "security_scanner": {
                    "url": "http://localhost:8001",
                    "health_endpoint": "/health",
                    "timeout": 30,
                    "retry_attempts": 3
                },
                "threat_intelligence": {
                    "url": "http://localhost:8002", 
                    "health_endpoint": "/health",
                    "timeout": 15,
                    "retry_attempts": 2
                },
                "network_monitor": {
                    "url": "http://localhost:8003",
                    "health_endpoint": "/health", 
                    "timeout": 10,
                    "retry_attempts": 2
                },
                "analytics_hub": {
                    "url": "http://localhost:8004",
                    "health_endpoint": "/health",
                    "timeout": 20,
                    "retry_attempts": 3
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics_endpoint": "/metrics",
                "health_endpoint": "/health",
                "prometheus_enabled": True
            }
        }
    
    async def init_redis(self):
        """Initialize Redis connection for rate limiting and caching"""
        if not self.config['rate_limiting']['enabled']:
            return
            
        try:
            import aioredis
            self.redis_client = await aioredis.from_url(
                self.config['rate_limiting']['redis_url'],
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for API gateway")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis_client = None
    
    def setup_routes(self):
        """Setup API gateway routes"""
        # Authentication endpoints
        self.app.router.add_post('/api/auth/login', self.login)
        self.app.router.add_post('/api/auth/logout', self.logout)
        self.app.router.add_post('/api/auth/refresh', self.refresh_token)
        self.app.router.add_post('/api/auth/register', self.register)
        self.app.router.add_get('/api/auth/user', self.get_user_info)
        
        # API key management
        self.app.router.add_post('/api/keys/generate', self.generate_api_key)
        self.app.router.add_get('/api/keys/list', self.list_api_keys)
        self.app.router.add_delete('/api/keys/{key_id}', self.revoke_api_key)
        
        # Gateway management
        self.app.router.add_get('/api/gateway/health', self.gateway_health)
        self.app.router.add_get('/api/gateway/metrics', self.gateway_metrics)
        self.app.router.add_get('/api/gateway/services', self.list_services)
        
        # Service proxy routes (dynamic routing)
        self.app.router.add_route('*', '/api/{service}/{path:.*}', self.proxy_request)
        
        # WebSocket proxy
        self.app.router.add_get('/ws/{service}/{path:.*}', self.proxy_websocket)
    
    @web.middleware
    async def cors_middleware(self, request: web_request.Request, handler):
        """CORS middleware"""
        response = await handler(request)
        
        origin = request.headers.get('Origin', '')
        allowed_origins = self.config['security']['cors_origins']
        
        if origin in allowed_origins or 'localhost' in origin:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-API-Key, X-Requested-With'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Max-Age'] = '86400'
        
        return response
    
    @web.middleware
    async def security_middleware(self, request: web_request.Request, handler):
        """Security middleware"""
        # HTTPS enforcement
        if self.config['security']['require_https'] and request.scheme != 'https':
            if not request.host.startswith('localhost') and not request.host.startswith('127.0.0.1'):
                raise web.HTTPPermanentRedirect(f"https://{request.host}{request.path_qs}")
        
        # Request size limit
        if request.content_length and request.content_length > self.config['security']['max_request_size']:
            raise web.HTTPRequestEntityTooLarge()
        
        # IP filtering
        client_ip = self.get_client_ip(request)
        
        # Check blacklist
        if client_ip in self.blocked_ips or client_ip in self.config['security']['ip_blacklist']:
            self.metrics['blocked_requests'] += 1
            raise web.HTTPForbidden(text="IP address blocked")
        
        # Check whitelist (if configured)
        whitelist = self.config['security']['ip_whitelist']
        if whitelist and client_ip not in whitelist:
            # Allow localhost for development
            if not self.is_local_ip(client_ip):
                self.metrics['blocked_requests'] += 1
                raise web.HTTPForbidden(text="IP address not whitelisted")
        
        response = await handler(request)
        
        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        return response
    
    @web.middleware
    async def auth_middleware(self, request: web_request.Request, handler):
        """Authentication middleware"""
        # Skip auth for public endpoints
        public_endpoints = ['/api/auth/login', '/api/auth/register', '/api/gateway/health']
        if any(request.path.startswith(endpoint) for endpoint in public_endpoints):
            return await handler(request)
        
        # Extract authentication credentials
        auth_result = await self.authenticate_request(request)
        
        if not auth_result['authenticated']:
            self.metrics['auth_failures'] += 1
            raise web.HTTPUnauthorized(
                text=json.dumps({'error': auth_result['error']}),
                content_type='application/json'
            )
        
        # Add user info to request
        request['user'] = auth_result['user']
        request['auth_method'] = auth_result['method']
        
        return await handler(request)
    
    @web.middleware
    async def rate_limit_middleware(self, request: web_request.Request, handler):
        """Rate limiting middleware"""
        if not self.config['rate_limiting']['enabled']:
            return await handler(request)
        
        client_ip = self.get_client_ip(request)
        user_id = getattr(request.get('user', {}), 'get', lambda x, y: None)('id', client_ip)
        
        # Check rate limits
        if await self.is_rate_limited(user_id, client_ip):
            self.metrics['rate_limit_hits'] += 1
            raise web.HTTPTooManyRequests(
                text=json.dumps({'error': 'Rate limit exceeded'}),
                content_type='application/json'
            )
        
        return await handler(request)
    
    @web.middleware
    async def logging_middleware(self, request: web_request.Request, handler):
        """Request logging middleware"""
        start_time = time.time()
        
        # Log request
        logger.info(f"API Request: {request.method} {request.path} from {self.get_client_ip(request)}")
        
        try:
            response = await handler(request)
            
            # Log response
            duration = (time.time() - start_time) * 1000
            logger.info(f"API Response: {response.status} in {duration:.2f}ms")
            
            return response
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"API Error: {str(e)} in {duration:.2f}ms")
            raise
    
    @web.middleware
    async def metrics_middleware(self, request: web_request.Request, handler):
        """Metrics collection middleware"""
        start_time = time.time()
        
        try:
            response = await handler(request)
            
            # Update metrics
            self.metrics['requests_total'] += 1
            self.metrics['requests_by_endpoint'][request.path] += 1
            
            if hasattr(request, 'user') and request.user:
                user_id = request.user.get('id', 'anonymous')
                self.metrics['requests_by_user'][user_id] += 1
            
            # Track response time
            duration = (time.time() - start_time) * 1000
            self.metrics['response_times'].append(duration)
            
            # Keep only last 1000 response times
            if len(self.metrics['response_times']) > 1000:
                self.metrics['response_times'] = self.metrics['response_times'][-1000:]
            
            response.headers['X-Response-Time'] = f"{duration:.2f}ms"
            
            return response
            
        except Exception as e:
            self.metrics['requests_total'] += 1
            raise
    
    async def authenticate_request(self, request: web_request.Request) -> Dict:
        """Authenticate incoming request"""
        # Try JWT authentication
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            jwt_result = await self.authenticate_jwt(token)
            if jwt_result['valid']:
                return {
                    'authenticated': True,
                    'user': jwt_result['user'],
                    'method': 'jwt'
                }
        
        # Try API key authentication
        api_key = request.headers.get(self.config['security']['api_key_header'])
        if api_key:
            api_key_result = await self.authenticate_api_key(api_key)
            if api_key_result['valid']:
                return {
                    'authenticated': True,
                    'user': api_key_result['user'],
                    'method': 'api_key'
                }
        
        return {
            'authenticated': False,
            'error': 'No valid authentication provided'
        }
    
    async def authenticate_jwt(self, token: str) -> Dict:
        """Authenticate JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.config['jwt']['algorithm']]
            )
            
            # Check expiration
            if payload.get('exp', 0) < time.time():
                return {'valid': False, 'error': 'Token expired'}
            
            # Check if token is revoked (Redis check)
            if self.redis_client:
                revoked = await self.redis_client.get(f"revoked_token:{token}")
                if revoked:
                    return {'valid': False, 'error': 'Token revoked'}
            
            return {
                'valid': True,
                'user': {
                    'id': payload.get('user_id'),
                    'email': payload.get('email'),
                    'role': payload.get('role', 'user'),
                    'permissions': payload.get('permissions', [])
                }
            }
            
        except jwt.InvalidTokenError as e:
            return {'valid': False, 'error': f'Invalid token: {str(e)}'}
    
    async def authenticate_api_key(self, api_key: str) -> Dict:
        """Authenticate API key"""
        # Hash the API key for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Check in memory store (in production, use database)
        if key_hash in self.api_keys:
            key_info = self.api_keys[key_hash]
            
            # Check if key is active
            if not key_info.get('active', True):
                return {'valid': False, 'error': 'API key inactive'}
            
            # Check expiration
            if key_info.get('expires_at') and key_info['expires_at'] < time.time():
                return {'valid': False, 'error': 'API key expired'}
            
            # Update last used
            key_info['last_used'] = time.time()
            
            return {
                'valid': True,
                'user': {
                    'id': key_info.get('user_id'),
                    'email': key_info.get('email'),
                    'role': key_info.get('role', 'api_user'),
                    'permissions': key_info.get('permissions', [])
                }
            }
        
        return {'valid': False, 'error': 'Invalid API key'}
    
    async def is_rate_limited(self, user_id: str, ip: str) -> bool:
        """Check if request should be rate limited"""
        current_time = time.time()
        
        # Use Redis for distributed rate limiting if available
        if self.redis_client:
            return await self.redis_rate_limit(user_id, ip, current_time)
        
        # Fallback to in-memory rate limiting
        return self.memory_rate_limit(user_id, current_time)
    
    async def redis_rate_limit(self, user_id: str, ip: str, current_time: float) -> bool:
        """Redis-based rate limiting"""
        pipe = self.redis_client.pipeline()
        
        # Check minute limit
        minute_key = f"rate_limit:minute:{user_id}:{int(current_time // 60)}"
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        
        # Check hour limit
        hour_key = f"rate_limit:hour:{user_id}:{int(current_time // 3600)}"
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)
        
        results = await pipe.execute()
        
        minute_count = results[0]
        hour_count = results[2]
        
        return (
            minute_count > self.config['rate_limiting']['requests_per_minute'] or
            hour_count > self.config['rate_limiting']['requests_per_hour']
        )
    
    def memory_rate_limit(self, user_id: str, current_time: float) -> bool:
        """In-memory rate limiting"""
        user_requests = self.rate_limiters[user_id]
        
        # Clean old requests (older than 1 hour)
        user_requests[:] = [req_time for req_time in user_requests if current_time - req_time < 3600]
        
        # Check minute limit
        recent_requests = [req_time for req_time in user_requests if current_time - req_time < 60]
        if len(recent_requests) >= self.config['rate_limiting']['requests_per_minute']:
            return True
        
        # Check hour limit
        if len(user_requests) >= self.config['rate_limiting']['requests_per_hour']:
            return True
        
        # Add current request
        user_requests.append(current_time)
        return False
    
    def get_client_ip(self, request: web_request.Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        return request.remote
    
    def is_local_ip(self, ip: str) -> bool:
        """Check if IP is local/private"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private or ip_obj.is_loopback
        except ValueError:
            return False
    
    # Authentication endpoints
    async def login(self, request: web_request.Request):
        """User login endpoint"""
        try:
            data = await request.json()
            email = data.get('email', '').lower().strip()
            password = data.get('password', '')
            
            if not email or not password:
                raise web.HTTPBadRequest(text=json.dumps({'error': 'Email and password required'}))
            
            # Validate user credentials (mock implementation)
            user = await self.validate_user_credentials(email, password)
            
            if not user:
                await asyncio.sleep(1)  # Prevent timing attacks
                raise web.HTTPUnauthorized(text=json.dumps({'error': 'Invalid credentials'}))
            
            # Generate JWT token
            token_data = {
                'user_id': user['id'],
                'email': user['email'],
                'role': user['role'],
                'permissions': user['permissions'],
                'exp': time.time() + (self.config['jwt']['expiration_hours'] * 3600),
                'iat': time.time()
            }
            
            access_token = jwt.encode(token_data, self.jwt_secret, algorithm=self.config['jwt']['algorithm'])
            
            # Generate refresh token
            refresh_token_data = {
                'user_id': user['id'],
                'type': 'refresh',
                'exp': time.time() + (self.config['jwt']['refresh_expiration_days'] * 86400),
                'iat': time.time()
            }
            
            refresh_token = jwt.encode(refresh_token_data, self.jwt_secret, algorithm=self.config['jwt']['algorithm'])
            
            response_data = {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'Bearer',
                'expires_in': self.config['jwt']['expiration_hours'] * 3600,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'role': user['role'],
                    'permissions': user['permissions']
                }
            }
            
            return web.json_response(response_data)
            
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text=json.dumps({'error': 'Invalid JSON'}))
        except Exception as e:
            logger.error(f"Login error: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({'error': 'Login failed'}))
    
    async def logout(self, request: web_request.Request):
        """User logout endpoint"""
        try:
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
                
                # Add token to revocation list
                if self.redis_client:
                    await self.redis_client.setex(f"revoked_token:{token}", 86400, "1")
            
            return web.json_response({'message': 'Logged out successfully'})
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({'error': 'Logout failed'}))
    
    async def refresh_token(self, request: web_request.Request):
        """Refresh JWT token endpoint"""
        try:
            data = await request.json()
            refresh_token = data.get('refresh_token')
            
            if not refresh_token:
                raise web.HTTPBadRequest(text=json.dumps({'error': 'Refresh token required'}))
            
            # Validate refresh token
            try:
                payload = jwt.decode(
                    refresh_token,
                    self.jwt_secret,
                    algorithms=[self.config['jwt']['algorithm']]
                )
                
                if payload.get('type') != 'refresh':
                    raise web.HTTPBadRequest(text=json.dumps({'error': 'Invalid token type'}))
                
                # Get user info
                user_id = payload.get('user_id')
                user = await self.get_user_by_id(user_id)
                
                if not user:
                    raise web.HTTPUnauthorized(text=json.dumps({'error': 'User not found'}))
                
                # Generate new access token
                token_data = {
                    'user_id': user['id'],
                    'email': user['email'],
                    'role': user['role'],
                    'permissions': user['permissions'],
                    'exp': time.time() + (self.config['jwt']['expiration_hours'] * 3600),
                    'iat': time.time()
                }
                
                access_token = jwt.encode(token_data, self.jwt_secret, algorithm=self.config['jwt']['algorithm'])
                
                return web.json_response({
                    'access_token': access_token,
                    'token_type': 'Bearer',
                    'expires_in': self.config['jwt']['expiration_hours'] * 3600
                })
                
            except jwt.InvalidTokenError:
                raise web.HTTPUnauthorized(text=json.dumps({'error': 'Invalid refresh token'}))
                
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text=json.dumps({'error': 'Invalid JSON'}))
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({'error': 'Token refresh failed'}))
    
    async def register(self, request: web_request.Request):
        """User registration endpoint"""
        try:
            data = await request.json()
            email = data.get('email', '').lower().strip()
            password = data.get('password', '')
            name = data.get('name', '').strip()
            
            # Validation
            if not email or not password or not name:
                raise web.HTTPBadRequest(text=json.dumps({'error': 'Email, password, and name required'}))
            
            if not self.is_valid_email(email):
                raise web.HTTPBadRequest(text=json.dumps({'error': 'Invalid email format'}))
            
            if not self.is_valid_password(password):
                raise web.HTTPBadRequest(text=json.dumps({
                    'error': 'Password must be at least 12 characters with complexity requirements'
                }))
            
            # Check if user exists
            if await self.user_exists(email):
                raise web.HTTPConflict(text=json.dumps({'error': 'User already exists'}))
            
            # Create user
            user = await self.create_user(email, password, name)
            
            return web.json_response({
                'message': 'User created successfully',
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'name': user['name'],
                    'role': user['role']
                }
            }, status=201)
            
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text=json.dumps({'error': 'Invalid JSON'}))
        except Exception as e:
            logger.error(f"Registration error: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({'error': 'Registration failed'}))
    
    async def get_user_info(self, request: web_request.Request):
        """Get current user information"""
        user = request['user']
        return web.json_response({
            'user': user,
            'auth_method': request['auth_method']
        })
    
    # API Key management endpoints
    async def generate_api_key(self, request: web_request.Request):
        """Generate new API key"""
        try:
            user = request['user']
            data = await request.json() if request.content_type == 'application/json' else {}
            
            # Generate API key
            api_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Store key info
            key_info = {
                'user_id': user['id'],
                'email': user['email'],
                'role': user['role'],
                'permissions': user.get('permissions', []),
                'created_at': time.time(),
                'last_used': None,
                'active': True,
                'name': data.get('name', 'Default API Key'),
                'expires_at': None  # No expiration by default
            }
            
            # Set expiration if provided
            if data.get('expires_in_days'):
                key_info['expires_at'] = time.time() + (data['expires_in_days'] * 86400)
            
            self.api_keys[key_hash] = key_info
            
            return web.json_response({
                'api_key': api_key,
                'key_id': key_hash[:16],  # Partial hash for identification
                'name': key_info['name'],
                'created_at': datetime.fromtimestamp(key_info['created_at']).isoformat(),
                'expires_at': datetime.fromtimestamp(key_info['expires_at']).isoformat() if key_info['expires_at'] else None
            }, status=201)
            
        except Exception as e:
            logger.error(f"API key generation error: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({'error': 'API key generation failed'}))
    
    async def list_api_keys(self, request: web_request.Request):
        """List user's API keys"""
        user = request['user']
        
        user_keys = []
        for key_hash, key_info in self.api_keys.items():
            if key_info['user_id'] == user['id']:
                user_keys.append({
                    'key_id': key_hash[:16],
                    'name': key_info['name'],
                    'created_at': datetime.fromtimestamp(key_info['created_at']).isoformat(),
                    'last_used': datetime.fromtimestamp(key_info['last_used']).isoformat() if key_info['last_used'] else None,
                    'expires_at': datetime.fromtimestamp(key_info['expires_at']).isoformat() if key_info['expires_at'] else None,
                    'active': key_info['active']
                })
        
        return web.json_response({'api_keys': user_keys})
    
    async def revoke_api_key(self, request: web_request.Request):
        """Revoke API key"""
        try:
            user = request['user']
            key_id = request.match_info['key_id']
            
            # Find and revoke key
            for key_hash, key_info in self.api_keys.items():
                if key_hash.startswith(key_id) and key_info['user_id'] == user['id']:
                    key_info['active'] = False
                    return web.json_response({'message': 'API key revoked successfully'})
            
            raise web.HTTPNotFound(text=json.dumps({'error': 'API key not found'}))
            
        except Exception as e:
            logger.error(f"API key revocation error: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({'error': 'API key revocation failed'}))
    
    # Gateway management endpoints  
    async def gateway_health(self, request: web_request.Request):
        """Gateway health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'services': {},
            'metrics': {
                'requests_total': self.metrics['requests_total'],
                'auth_failures': self.metrics['auth_failures'],
                'rate_limit_hits': self.metrics['rate_limit_hits'],
                'blocked_requests': self.metrics['blocked_requests'],
                'avg_response_time': sum(self.metrics['response_times'][-100:]) / len(self.metrics['response_times'][-100:]) if self.metrics['response_times'] else 0
            }
        }
        
        # Check service health
        for service_name, service_config in self.config['services'].items():
            try:
                async with ClientSession() as session:
                    async with session.get(
                        f"{service_config['url']}{service_config['health_endpoint']}",
                        timeout=5
                    ) as response:
                        if response.status == 200:
                            health_status['services'][service_name] = 'healthy'
                        else:
                            health_status['services'][service_name] = 'unhealthy'
            except Exception:
                health_status['services'][service_name] = 'unreachable'
        
        return web.json_response(health_status)
    
    async def gateway_metrics(self, request: web_request.Request):
        """Gateway metrics endpoint"""
        metrics_data = {
            'requests_total': self.metrics['requests_total'],
            'requests_by_endpoint': dict(self.metrics['requests_by_endpoint']),
            'requests_by_user': dict(self.metrics['requests_by_user']),
            'auth_failures': self.metrics['auth_failures'],
            'rate_limit_hits': self.metrics['rate_limit_hits'],
            'blocked_requests': self.metrics['blocked_requests'],
            'response_times': {
                'avg': sum(self.metrics['response_times']) / len(self.metrics['response_times']) if self.metrics['response_times'] else 0,
                'min': min(self.metrics['response_times']) if self.metrics['response_times'] else 0,
                'max': max(self.metrics['response_times']) if self.metrics['response_times'] else 0,
                'p95': self.percentile(self.metrics['response_times'], 95) if self.metrics['response_times'] else 0,
                'p99': self.percentile(self.metrics['response_times'], 99) if self.metrics['response_times'] else 0
            }
        }
        
        return web.json_response(metrics_data)
    
    async def list_services(self, request: web_request.Request):
        """List available services"""
        services = []
        for service_name, service_config in self.config['services'].items():
            services.append({
                'name': service_name,
                'url': service_config['url'],
                'timeout': service_config['timeout'],
                'retry_attempts': service_config['retry_attempts']
            })
        
        return web.json_response({'services': services})
    
    # Service proxy methods
    async def proxy_request(self, request: web_request.Request):
        """Proxy HTTP requests to backend services"""
        service_name = request.match_info['service']
        path = request.match_info['path']
        
        # Get service configuration
        service_config = self.config['services'].get(service_name)
        if not service_config:
            raise web.HTTPNotFound(text=json.dumps({'error': f'Service {service_name} not found'}))
        
        # Build target URL
        target_url = f"{service_config['url']}/{path}"
        if request.query_string:
            target_url += f"?{request.query_string}"
        
        # Proxy the request
        try:
            async with ClientSession() as session:
                # Copy headers (exclude hop-by-hop headers)
                headers = {k: v for k, v in request.headers.items() 
                          if k.lower() not in ['host', 'content-length', 'connection']}
                
                # Add user context
                if hasattr(request, 'user') and request.user:
                    headers['X-User-ID'] = str(request.user.get('id', ''))
                    headers['X-User-Email'] = request.user.get('email', '')
                    headers['X-User-Role'] = request.user.get('role', '')
                
                # Get request body
                body = await request.read() if request.can_read_body else None
                
                # Make request with retries
                for attempt in range(service_config['retry_attempts']):
                    try:
                        async with session.request(
                            method=request.method,
                            url=target_url,
                            headers=headers,
                            data=body,
                            timeout=service_config['timeout']
                        ) as response:
                            
                            # Copy response headers
                            response_headers = {k: v for k, v in response.headers.items()
                                             if k.lower() not in ['content-length', 'connection', 'transfer-encoding']}
                            
                            # Get response body
                            response_body = await response.read()
                            
                            return web.Response(
                                body=response_body,
                                status=response.status,
                                headers=response_headers
                            )
                            
                    except asyncio.TimeoutError:
                        if attempt == service_config['retry_attempts'] - 1:
                            raise web.HTTPGatewayTimeout(text=json.dumps({'error': 'Service timeout'}))
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        
                    except Exception as e:
                        if attempt == service_config['retry_attempts'] - 1:
                            logger.error(f"Service proxy error: {e}")
                            raise web.HTTPBadGateway(text=json.dumps({'error': 'Service unavailable'}))
                        await asyncio.sleep(0.5 * (attempt + 1))
                        
        except Exception as e:
            logger.error(f"Proxy request error: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({'error': 'Proxy error'}))
    
    async def proxy_websocket(self, request: web_request.Request):
        """Proxy WebSocket connections to backend services"""
        service_name = request.match_info['service']
        path = request.match_info['path']
        
        # Get service configuration
        service_config = self.config['services'].get(service_name)
        if not service_config:
            raise web.HTTPNotFound(text=json.dumps({'error': f'Service {service_name} not found'}))
        
        # Build target WebSocket URL
        target_url = service_config['url'].replace('http://', 'ws://').replace('https://', 'wss://')
        target_url += f"/{path}"
        
        # Proxy WebSocket connection
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            async with ClientSession() as session:
                async with session.ws_connect(target_url) as backend_ws:
                    # Proxy messages bidirectionally
                    async def proxy_client_to_backend():
                        async for msg in ws:
                            if msg.type == web.WSMsgType.TEXT:
                                await backend_ws.send_str(msg.data)
                            elif msg.type == web.WSMsgType.BINARY:
                                await backend_ws.send_bytes(msg.data)
                            elif msg.type == web.WSMsgType.ERROR:
                                break
                    
                    async def proxy_backend_to_client():
                        async for msg in backend_ws:
                            if msg.type == web.WSMsgType.TEXT:
                                await ws.send_str(msg.data)
                            elif msg.type == web.WSMsgType.BINARY:
                                await ws.send_bytes(msg.data)
                            elif msg.type == web.WSMsgType.ERROR:
                                break
                    
                    # Run both proxy directions concurrently
                    await asyncio.gather(
                        proxy_client_to_backend(),
                        proxy_backend_to_client(),
                        return_exceptions=True
                    )
                    
        except Exception as e:
            logger.error(f"WebSocket proxy error: {e}")
        
        return ws
    
    # Utility methods
    async def validate_user_credentials(self, email: str, password: str) -> Optional[Dict]:
        """Validate user credentials (mock implementation)"""
        # Mock user database
        mock_users = {
            'admin@xorb.security': {
                'id': '1',
                'email': 'admin@xorb.security',
                'name': 'Administrator',
                'password_hash': bcrypt.hashpw('admin123456789'.encode(), bcrypt.gensalt()).decode(),
                'role': 'admin',
                'permissions': ['read', 'write', 'admin']
            },
            'user@xorb.security': {
                'id': '2',
                'email': 'user@xorb.security',
                'name': 'Regular User',
                'password_hash': bcrypt.hashpw('user123456789'.encode(), bcrypt.gensalt()).decode(),
                'role': 'user',
                'permissions': ['read']
            }
        }
        
        user = mock_users.get(email)
        if user and bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
            return {
                'id': user['id'],
                'email': user['email'],
                'name': user['name'],
                'role': user['role'],
                'permissions': user['permissions']
            }
        
        return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID (mock implementation)"""
        # Mock implementation
        if user_id == '1':
            return {
                'id': '1',
                'email': 'admin@xorb.security',
                'name': 'Administrator',
                'role': 'admin',
                'permissions': ['read', 'write', 'admin']
            }
        elif user_id == '2':
            return {
                'id': '2',
                'email': 'user@xorb.security',
                'name': 'Regular User',
                'role': 'user',
                'permissions': ['read']
            }
        
        return None
    
    async def user_exists(self, email: str) -> bool:
        """Check if user exists (mock implementation)"""
        mock_users = ['admin@xorb.security', 'user@xorb.security']
        return email in mock_users
    
    async def create_user(self, email: str, password: str, name: str) -> Dict:
        """Create new user (mock implementation)"""
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        user = {
            'id': str(int(time.time())),  # Simple ID generation
            'email': email,
            'name': name,
            'password_hash': password_hash,
            'role': 'user',
            'permissions': ['read'],
            'created_at': time.time()
        }
        
        return user
    
    def is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def is_valid_password(self, password: str) -> bool:
        """Validate password complexity"""
        if len(password) < self.config['authentication']['password_min_length']:
            return False
        
        if not self.config['authentication']['password_complexity']:
            return True
        
        # Check complexity requirements
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*(),.?":{}|<>' for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def start_gateway(self):
        """Start the API gateway"""
        logger.info("Starting XORB API Gateway...")
        
        # Initialize Redis
        await self.init_redis()
        
        server_config = self.config['server']
        
        # Create SSL context if SSL is enabled
        ssl_context = None
        if server_config['ssl_enabled']:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(server_config['ssl_cert'], server_config['ssl_key'])
        
        # Start server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(
            runner,
            server_config['host'],
            server_config['port'],
            ssl_context=ssl_context
        )
        
        await site.start()
        
        protocol = 'https' if ssl_context else 'http'
        logger.info(f"🔒 API Gateway running on {protocol}://{server_config['host']}:{server_config['port']}")
        logger.info("✅ XORB API Gateway is ready!")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down API Gateway...")

async def main():
    """Main gateway function"""
    gateway = XORBAPIGateway()
    
    try:
        print("🚀 Initializing XORB API Gateway...")
        await gateway.start_gateway()
        
    except Exception as e:
        logger.error(f"Gateway startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())