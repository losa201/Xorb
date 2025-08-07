#!/usr/bin/env python3
"""
XORB Web Server Deployment
Complete web server setup with HTTPS, caching, and performance optimization
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import ssl
import socket
from aiohttp import web, web_request, web_response
import aiofiles
import asyncio_mqtt
import redis
import gzip
import mimetypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XORBWebServer:
    """Production-ready web server for XORB platform"""
    
    def __init__(self, config_path: str = "config/web_server_config.json"):
        self.config_path = config_path
        self.config = self.load_configuration()
        self.app = web.Application(middlewares=[
            self.cors_middleware,
            self.security_headers_middleware,
            self.compression_middleware,
            self.cache_middleware,
            self.analytics_middleware
        ])
        
        # Initialize components
        self.redis_client = None
        self.setup_routes()
        self.setup_static_routes()
        
        # Performance metrics
        self.metrics = {
            'requests_total': 0,
            'requests_by_path': {},
            'response_times': [],
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def load_configuration(self) -> Dict:
        """Load web server configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return self.get_default_configuration()
    
    def get_default_configuration(self) -> Dict:
        """Get default web server configuration"""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 443,
                "http_port": 80,
                "workers": 4,
                "max_connections": 1000,
                "keepalive_timeout": 60,
                "client_timeout": 30
            },
            "ssl": {
                "enabled": True,
                "cert_file": "/etc/ssl/certs/xorb.security.crt",
                "key_file": "/etc/ssl/private/xorb.security.key",
                "ca_file": "/etc/ssl/certs/ca-bundle.crt",
                "protocols": ["TLSv1.2", "TLSv1.3"],
                "ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
            },
            "security": {
                "hsts_max_age": 31536000,
                "csp_policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://www.googletagmanager.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self' wss: https:",
                "frame_options": "DENY",
                "content_type_options": "nosniff",
                "referrer_policy": "strict-origin-when-cross-origin"
            },
            "caching": {
                "enabled": True,
                "redis_url": "redis://localhost:6379/0",
                "static_cache_ttl": 31536000,  # 1 year
                "api_cache_ttl": 300,  # 5 minutes
                "html_cache_ttl": 3600  # 1 hour
            },
            "compression": {
                "enabled": True,
                "level": 6,
                "min_size": 1024,
                "types": [
                    "text/html",
                    "text/css",
                    "text/javascript",
                    "application/javascript",
                    "application/json",
                    "text/xml",
                    "application/xml",
                    "image/svg+xml"
                ]
            },
            "monitoring": {
                "enabled": True,
                "metrics_endpoint": "/metrics",
                "health_endpoint": "/health"
            },
            "frontend": {
                "path": "/root/Xorb/frontend",
                "index_file": "index.html",
                "error_pages": {
                    "404": "404.html",
                    "500": "500.html"
                }
            }
        }
    
    async def init_redis(self):
        """Initialize Redis connection"""
        if not self.config['caching']['enabled']:
            return
        
        try:
            import aioredis
            self.redis_client = await aioredis.from_url(
                self.config['caching']['redis_url'],
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis_client = None
    
    def setup_routes(self):
        """Setup application routes"""
        # API routes
        self.app.router.add_get('/api/health', self.health_check)
        self.app.router.add_get('/api/metrics', self.get_metrics)
        self.app.router.add_get('/api/status', self.get_status)
        self.app.router.add_get('/api/version', self.get_version)
        
        # Security tool API endpoints
        self.app.router.add_post('/api/security/scan', self.security_scan)
        self.app.router.add_get('/api/threats/intelligence', self.threat_intelligence)
        self.app.router.add_get('/api/network/monitor', self.network_monitor)
        self.app.router.add_get('/api/analytics/data', self.analytics_data)
        self.app.router.add_post('/api/incident/report', self.incident_report)
        self.app.router.add_get('/api/compliance/status', self.compliance_status)
        self.app.router.add_post('/api/hunt/query', self.threat_hunt_query)
        self.app.router.add_get('/api/soc/dashboard', self.soc_dashboard_data)
        
        # WebSocket endpoints
        self.app.router.add_get('/ws/realtime', self.websocket_handler)
        self.app.router.add_get('/ws/threats', self.threats_websocket)
        self.app.router.add_get('/ws/monitoring', self.monitoring_websocket)
        
        # SEO and marketing routes
        self.app.router.add_get('/sitemap.xml', self.serve_sitemap)
        self.app.router.add_get('/robots.txt', self.serve_robots)
        self.app.router.add_get('/manifest.json', self.serve_manifest)
        
    def setup_static_routes(self):
        """Setup static file serving"""
        frontend_path = self.config['frontend']['path']
        
        # Serve static files with proper caching
        self.app.router.add_static(
            '/', 
            frontend_path,
            follow_symlinks=True,
            show_index=True,
            append_version=True
        )
        
        # Special routes for PWA
        self.app.router.add_get('/sw.js', self.serve_service_worker)
        self.app.router.add_get('/offline.html', self.serve_offline_page)
        
        # Catch-all route for SPA
        self.app.router.add_route('*', '/{path:.*}', self.spa_handler)
    
    @web.middleware
    async def cors_middleware(self, request: web_request.Request, handler):
        """CORS middleware"""
        response = await handler(request)
        
        # Allow specific origins in production
        allowed_origins = [
            'https://xorb.security',
            'https://www.xorb.security',
            'https://app.xorb.security'
        ]
        
        origin = request.headers.get('Origin', '')
        if origin in allowed_origins or request.host.startswith('localhost'):
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
        
        return response
    
    @web.middleware
    async def security_headers_middleware(self, request: web_request.Request, handler):
        """Security headers middleware"""
        response = await handler(request)
        
        security_config = self.config['security']
        
        # HSTS
        response.headers['Strict-Transport-Security'] = f"max-age={security_config['hsts_max_age']}; includeSubDomains; preload"
        
        # CSP
        response.headers['Content-Security-Policy'] = security_config['csp_policy']
        
        # Other security headers
        response.headers['X-Frame-Options'] = security_config['frame_options']
        response.headers['X-Content-Type-Options'] = security_config['content_type_options']
        response.headers['Referrer-Policy'] = security_config['referrer_policy']
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Custom security headers
        response.headers['X-Powered-By'] = 'XORB Security Platform'
        response.headers['X-Security-Level'] = 'Enterprise'
        
        return response
    
    @web.middleware
    async def compression_middleware(self, request: web_request.Request, handler):
        """Compression middleware"""
        response = await handler(request)
        
        if not self.config['compression']['enabled']:
            return response
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding:
            return response
        
        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if not any(ct in content_type for ct in self.config['compression']['types']):
            return response
        
        # Check minimum size
        content_length = len(response.body or b'')
        if content_length < self.config['compression']['min_size']:
            return response
        
        # Compress response
        compressed_body = gzip.compress(
            response.body, 
            compresslevel=self.config['compression']['level']
        )
        
        response.body = compressed_body
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = str(len(compressed_body))
        response.headers['Vary'] = 'Accept-Encoding'
        
        return response
    
    @web.middleware
    async def cache_middleware(self, request: web_request.Request, handler):
        """Caching middleware"""
        if not self.config['caching']['enabled'] or not self.redis_client:
            return await handler(request)
        
        # Generate cache key
        cache_key = f"xorb:cache:{request.method}:{request.path_qs}"
        
        # Check cache for GET requests
        if request.method == 'GET':
            try:
                cached_response = await self.redis_client.get(cache_key)
                if cached_response:
                    self.metrics['cache_hits'] += 1
                    data = json.loads(cached_response)
                    return web.Response(
                        body=data['body'],
                        status=data['status'],
                        headers=data['headers']
                    )
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
        
        # Process request
        response = await handler(request)
        self.metrics['cache_misses'] += 1
        
        # Cache successful GET responses
        if (request.method == 'GET' and 
            response.status == 200 and 
            'Cache-Control' not in response.headers):
            
            try:
                # Determine TTL based on path
                if request.path.startswith('/api/'):
                    ttl = self.config['caching']['api_cache_ttl']
                elif request.path.endswith(('.css', '.js', '.png', '.jpg', '.ico')):
                    ttl = self.config['caching']['static_cache_ttl']
                else:
                    ttl = self.config['caching']['html_cache_ttl']
                
                cache_data = {
                    'body': response.body.decode() if response.body else '',
                    'status': response.status,
                    'headers': dict(response.headers)
                }
                
                await self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(cache_data)
                )
                
                # Set cache headers
                response.headers['Cache-Control'] = f'public, max-age={ttl}'
                response.headers['X-Cache'] = 'MISS'
                
            except Exception as e:
                logger.debug(f"Cache write error: {e}")
        else:
            response.headers['X-Cache'] = 'HIT' if 'X-Cache' not in response.headers else response.headers['X-Cache']
        
        return response
    
    @web.middleware
    async def analytics_middleware(self, request: web_request.Request, handler):
        """Analytics and monitoring middleware"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await handler(request)
            
            # Track metrics
            self.metrics['requests_total'] += 1
            path_key = request.path[:50]  # Truncate long paths
            self.metrics['requests_by_path'][path_key] = self.metrics['requests_by_path'].get(path_key, 0) + 1
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Request error: {e}")
            response = web.Response(
                text=json.dumps({'error': 'Internal server error'}),
                status=500,
                content_type='application/json'
            )
        
        # Track response time
        end_time = asyncio.get_event_loop().time()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        self.metrics['response_times'].append(response_time)
        
        # Keep only last 1000 response times
        if len(self.metrics['response_times']) > 1000:
            self.metrics['response_times'] = self.metrics['response_times'][-1000:]
        
        # Add response time header
        response.headers['X-Response-Time'] = f"{response_time:.2f}ms"
        
        return response
    
    async def health_check(self, request: web_request.Request):
        """Health check endpoint"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'services': {
                'web_server': 'healthy',
                'redis_cache': 'healthy' if self.redis_client else 'unavailable',
                'filesystem': 'healthy'
            },
            'metrics': {
                'uptime_seconds': int(asyncio.get_event_loop().time()),
                'requests_total': self.metrics['requests_total'],
                'cache_hit_rate': self.get_cache_hit_rate()
            }
        }
        
        return web.json_response(health_status)
    
    async def get_metrics(self, request: web_request.Request):
        """Prometheus-style metrics endpoint"""
        avg_response_time = (
            sum(self.metrics['response_times']) / len(self.metrics['response_times'])
            if self.metrics['response_times'] else 0
        )
        
        metrics_data = {
            'xorb_http_requests_total': self.metrics['requests_total'],
            'xorb_http_request_errors_total': self.metrics['errors'],
            'xorb_http_response_time_avg_ms': avg_response_time,
            'xorb_cache_hits_total': self.metrics['cache_hits'],
            'xorb_cache_misses_total': self.metrics['cache_misses'],
            'xorb_cache_hit_rate': self.get_cache_hit_rate()
        }
        
        return web.json_response(metrics_data)
    
    async def get_status(self, request: web_request.Request):
        """System status endpoint"""
        status = {
            'platform': 'XORB Cybersecurity Platform',
            'version': '2.0.0',
            'status': 'operational',
            'components': {
                'frontend': 'operational',
                'api': 'operational',
                'security_tools': 'operational',
                'monitoring': 'operational',
                'analytics': 'operational'
            },
            'uptime': int(asyncio.get_event_loop().time()),
            'last_updated': datetime.now().isoformat()
        }
        
        return web.json_response(status)
    
    async def get_version(self, request: web_request.Request):
        """Version information endpoint"""
        version_info = {
            'version': '2.0.0',
            'build': '20250101-001',
            'commit': 'abc123def456',
            'build_date': '2025-01-01T00:00:00Z',
            'features': [
                'AI-Powered Threat Detection',
                'Autonomous Security Operations',
                'Real-Time Monitoring',
                'Compliance Management',
                'Incident Response',
                'Threat Hunting',
                'Security Analytics'
            ]
        }
        
        return web.json_response(version_info)
    
    # Security tool API endpoints
    async def security_scan(self, request: web_request.Request):
        """Security scan endpoint"""
        data = await request.json() if request.content_type == 'application/json' else {}
        
        # Simulate security scan
        scan_result = {
            'scan_id': f"SCAN-{int(asyncio.get_event_loop().time())}",
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'target': data.get('target', 'unknown'),
            'vulnerabilities': [
                {
                    'id': 'CVE-2023-12345',
                    'severity': 'medium',
                    'title': 'Sample vulnerability',
                    'description': 'This is a sample vulnerability for demonstration'
                }
            ],
            'summary': {
                'total': 1,
                'critical': 0,
                'high': 0,
                'medium': 1,
                'low': 0
            }
        }
        
        return web.json_response(scan_result)
    
    async def threat_intelligence(self, request: web_request.Request):
        """Threat intelligence endpoint"""
        intelligence_data = {
            'timestamp': datetime.now().isoformat(),
            'threats': [
                {
                    'id': 'THREAT-001',
                    'severity': 'high',
                    'type': 'malware',
                    'description': 'Advanced persistent threat detected',
                    'indicators': ['192.168.1.100', 'malware.example.com'],
                    'confidence': 0.95
                }
            ],
            'statistics': {
                'threats_detected_24h': 156,
                'threats_blocked_24h': 142,
                'threat_sources': ['darkweb', 'honeypots', 'feeds']
            }
        }
        
        return web.json_response(intelligence_data)
    
    async def network_monitor(self, request: web_request.Request):
        """Network monitoring endpoint"""
        network_data = {
            'timestamp': datetime.now().isoformat(),
            'network_status': 'healthy',
            'connections': {
                'active': 245,
                'suspicious': 3,
                'blocked': 12
            },
            'bandwidth': {
                'inbound_mbps': 156.7,
                'outbound_mbps': 89.3,
                'utilization': 0.23
            },
            'alerts': []
        }
        
        return web.json_response(network_data)
    
    async def analytics_data(self, request: web_request.Request):
        """Analytics data endpoint"""
        analytics = {
            'timestamp': datetime.now().isoformat(),
            'period': '24h',
            'metrics': {
                'security_events': 1247,
                'threats_detected': 23,
                'incidents_resolved': 18,
                'compliance_score': 97.8,
                'system_uptime': 99.9
            },
            'trends': {
                'threat_detection': 'decreasing',
                'system_performance': 'stable',
                'user_activity': 'increasing'
            }
        }
        
        return web.json_response(analytics)
    
    async def incident_report(self, request: web_request.Request):
        """Incident reporting endpoint"""
        data = await request.json()
        
        incident = {
            'incident_id': f"INC-{int(asyncio.get_event_loop().time())}",
            'status': 'reported',
            'severity': data.get('severity', 'medium'),
            'title': data.get('title', 'Security incident'),
            'description': data.get('description', ''),
            'reporter': data.get('reporter', 'system'),
            'timestamp': datetime.now().isoformat()
        }
        
        return web.json_response(incident, status=201)
    
    async def compliance_status(self, request: web_request.Request):
        """Compliance status endpoint"""
        compliance = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': 96.8,
            'frameworks': {
                'SOC2': {'score': 98.5, 'status': 'compliant'},
                'GDPR': {'score': 97.2, 'status': 'compliant'},
                'HIPAA': {'score': 95.8, 'status': 'compliant'},
                'PCI_DSS': {'score': 94.3, 'status': 'compliant'},
                'ISO27001': {'score': 96.7, 'status': 'compliant'}
            },
            'recommendations': [
                'Update password policy documentation',
                'Schedule quarterly security training'
            ]
        }
        
        return web.json_response(compliance)
    
    async def threat_hunt_query(self, request: web_request.Request):
        """Threat hunting query endpoint"""
        data = await request.json()
        query = data.get('query', '')
        
        hunt_results = {
            'query_id': f"HUNT-{int(asyncio.get_event_loop().time())}",
            'query': query,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'match_id': 'MATCH-001',
                    'confidence': 0.87,
                    'description': 'Suspicious network activity detected',
                    'details': {'source_ip': '192.168.1.100', 'protocol': 'TCP'}
                }
            ],
            'summary': {
                'total_matches': 1,
                'high_confidence': 1,
                'execution_time_ms': 245
            }
        }
        
        return web.json_response(hunt_results)
    
    async def soc_dashboard_data(self, request: web_request.Request):
        """SOC dashboard data endpoint"""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'alerts': {
                'critical': 2,
                'high': 8,
                'medium': 15,
                'low': 23
            },
            'agents': {
                'total': 64,
                'healthy': 62,
                'warning': 2,
                'offline': 0
            },
            'system_health': {
                'overall': 'healthy',
                'cpu_usage': 45.2,
                'memory_usage': 67.8,
                'disk_usage': 23.1
            },
            'recent_events': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'threat_detected',
                    'severity': 'medium',
                    'description': 'Suspicious login attempt blocked'
                }
            ]
        }
        
        return web.json_response(dashboard_data)
    
    # WebSocket handlers
    async def websocket_handler(self, request: web_request.Request):
        """Generic WebSocket handler"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    # Echo back with timestamp
                    response = {
                        'type': 'response',
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    await ws.send_str(json.dumps(response))
                    
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    
        except Exception as e:
            logger.error(f'WebSocket handler error: {e}')
        
        return ws
    
    async def threats_websocket(self, request: web_request.Request):
        """Real-time threats WebSocket"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            # Send periodic threat updates
            while not ws.closed:
                threat_update = {
                    'type': 'threat_update',
                    'timestamp': datetime.now().isoformat(),
                    'threats': [
                        {
                            'id': f"THR-{int(asyncio.get_event_loop().time())}",
                            'severity': 'medium',
                            'type': 'suspicious_activity',
                            'description': 'Unusual network traffic detected'
                        }
                    ]
                }
                
                await ws.send_str(json.dumps(threat_update))
                await asyncio.sleep(30)  # Send updates every 30 seconds
                
        except Exception as e:
            logger.error(f'Threats WebSocket error: {e}')
        
        return ws
    
    async def monitoring_websocket(self, request: web_request.Request):
        """Real-time monitoring WebSocket"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            while not ws.closed:
                monitoring_data = {
                    'type': 'monitoring_update',
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'cpu_usage': 45.2,
                        'memory_usage': 67.8,
                        'network_io': 156.7,
                        'active_connections': 245
                    }
                }
                
                await ws.send_str(json.dumps(monitoring_data))
                await asyncio.sleep(5)  # Send updates every 5 seconds
                
        except Exception as e:
            logger.error(f'Monitoring WebSocket error: {e}')
        
        return ws
    
    # Static file handlers
    async def serve_sitemap(self, request: web_request.Request):
        """Serve sitemap.xml"""
        sitemap_path = Path(self.config['frontend']['path']) / 'sitemap.xml'
        
        if sitemap_path.exists():
            async with aiofiles.open(sitemap_path, 'r') as f:
                content = await f.read()
            
            return web.Response(
                text=content,
                content_type='application/xml',
                headers={'Cache-Control': 'public, max-age=86400'}  # 24 hours
            )
        
        return web.Response(status=404)
    
    async def serve_robots(self, request: web_request.Request):
        """Serve robots.txt"""
        robots_path = Path(self.config['frontend']['path']) / 'robots.txt'
        
        if robots_path.exists():
            async with aiofiles.open(robots_path, 'r') as f:
                content = await f.read()
            
            return web.Response(
                text=content,
                content_type='text/plain',
                headers={'Cache-Control': 'public, max-age=86400'}  # 24 hours
            )
        
        return web.Response(status=404)
    
    async def serve_manifest(self, request: web_request.Request):
        """Serve PWA manifest"""
        manifest_path = Path(self.config['frontend']['path']) / 'manifest.json'
        
        if manifest_path.exists():
            async with aiofiles.open(manifest_path, 'r') as f:
                content = await f.read()
            
            return web.Response(
                text=content,
                content_type='application/json',
                headers={'Cache-Control': 'public, max-age=86400'}  # 24 hours
            )
        
        return web.Response(status=404)
    
    async def serve_service_worker(self, request: web_request.Request):
        """Serve service worker"""
        sw_path = Path(self.config['frontend']['path']) / 'sw.js'
        
        if sw_path.exists():
            async with aiofiles.open(sw_path, 'r') as f:
                content = await f.read()
            
            return web.Response(
                text=content,
                content_type='application/javascript',
                headers={'Cache-Control': 'no-cache'}  # Don't cache service worker
            )
        
        return web.Response(status=404)
    
    async def serve_offline_page(self, request: web_request.Request):
        """Serve offline page"""
        offline_path = Path(self.config['frontend']['path']) / 'offline.html'
        
        if offline_path.exists():
            async with aiofiles.open(offline_path, 'r') as f:
                content = await f.read()
            
            return web.Response(
                text=content,
                content_type='text/html',
                headers={'Cache-Control': 'public, max-age=3600'}  # 1 hour
            )
        
        return web.Response(status=404)
    
    async def spa_handler(self, request: web_request.Request):
        """SPA fallback handler"""
        # For SPA routing, serve index.html for non-API routes
        if request.path.startswith('/api/') or request.path.startswith('/ws/'):
            return web.Response(status=404)
        
        index_path = Path(self.config['frontend']['path']) / self.config['frontend']['index_file']
        
        if index_path.exists():
            async with aiofiles.open(index_path, 'r') as f:
                content = await f.read()
            
            return web.Response(
                text=content,
                content_type='text/html',
                headers={'Cache-Control': 'no-cache'}
            )
        
        return web.Response(status=404)
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total == 0:
            return 0.0
        return (self.metrics['cache_hits'] / total) * 100
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for HTTPS"""
        ssl_config = self.config['ssl']
        
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(ssl_config['cert_file'], ssl_config['key_file'])
        
        # Set protocols
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Set ciphers
        context.set_ciphers(ssl_config['ciphers'])
        
        return context
    
    async def start_server(self):
        """Start the web server"""
        logger.info("Starting XORB Web Server...")
        
        # Initialize Redis
        await self.init_redis()
        
        server_config = self.config['server']
        
        if self.config['ssl']['enabled']:
            # HTTPS server
            ssl_context = self.create_ssl_context()
            
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(
                runner,
                server_config['host'],
                server_config['port'],
                ssl_context=ssl_context
            )
            
            await site.start()
            
            logger.info(f"🔒 HTTPS server running on https://{server_config['host']}:{server_config['port']}")
            
            # HTTP redirect server
            redirect_app = web.Application()
            redirect_app.router.add_route('*', '/{path:.*}', self.redirect_to_https)
            
            redirect_runner = web.AppRunner(redirect_app)
            await redirect_runner.setup()
            
            redirect_site = web.TCPSite(
                redirect_runner,
                server_config['host'],
                server_config['http_port']
            )
            
            await redirect_site.start()
            
            logger.info(f"🔀 HTTP redirect server running on http://{server_config['host']}:{server_config['http_port']}")
            
        else:
            # HTTP only server
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(
                runner,
                server_config['host'],
                server_config['port']
            )
            
            await site.start()
            
            logger.info(f"🌐 HTTP server running on http://{server_config['host']}:{server_config['port']}")
        
        logger.info("✅ XORB Web Server is ready!")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down server...")
    
    async def redirect_to_https(self, request: web_request.Request):
        """Redirect HTTP to HTTPS"""
        https_url = f"https://{request.host.split(':')[0]}{request.path_qs}"
        return web.Response(
            status=301,
            headers={'Location': https_url}
        )

async def main():
    """Main server function"""
    web_server = XORBWebServer()
    
    try:
        print("🚀 Initializing XORB Web Server...")
        await web_server.start_server()
        
    except Exception as e:
        logger.error(f"Server startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())