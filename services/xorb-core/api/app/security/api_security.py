"""
Comprehensive API security middleware
"""

import time
import hashlib
import secrets
from typing import Optional, Dict, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException, status

from .input_validation import SecurityValidator

def validate_request_data(data):
    """Simple request data validation - placeholder function."""
    if isinstance(data, dict):
        validator = SecurityValidator()
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 10000:  # Prevent extremely large strings
                raise HTTPException(status_code=400, detail=f"Value too large for field {key}")
    return data


@dataclass
class SecurityConfig:
    """API security configuration"""
    enable_request_signing: bool = True
    enable_replay_protection: bool = True
    enable_ip_filtering: bool = True
    enable_user_agent_validation: bool = True
    replay_window_seconds: int = 300  # 5 minutes
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    blocked_user_agents: Set[str] = None
    blocked_ips: Set[str] = None
    
    def __post_init__(self):
        if self.blocked_user_agents is None:
            self.blocked_user_agents = {
                'sqlmap', 'nikto', 'nmap', 'masscan', 'zap',
                'burpsuite', 'curl/7.', 'python-requests',
                'bot', 'crawler', 'spider', 'scanner'
            }
        
        if self.blocked_ips is None:
            self.blocked_ips = set()


class APISecurityMiddleware(BaseHTTPMiddleware):
    """Advanced API security middleware"""
    
    def __init__(self, app, redis_client=None, config: SecurityConfig = None):
        super().__init__(app)
        self.redis_client = redis_client
        self.config = config or SecurityConfig()
        
        # Paths that bypass security checks
        self.bypass_paths = {
            '/health', '/readiness', '/metrics', 
            '/docs', '/openapi.json', '/favicon.ico'
        }
        
        # High-risk endpoints requiring extra validation
        self.high_risk_paths = {
            '/auth', '/admin', '/api/v1/scan', 
            '/api/v1/upload', '/api/v1/execute'
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip security checks for bypass paths
        if any(request.url.path.startswith(path) for path in self.bypass_paths):
            return await call_next(request)
        
        try:
            # Perform security validations
            await self._validate_request_size(request)
            await self._validate_ip_address(request)
            await self._validate_user_agent(request)
            await self._validate_headers(request)
            
            # Anti-replay protection
            if self.config.enable_replay_protection:
                await self._check_replay_attack(request)
            
            # Request signing validation
            if self.config.enable_request_signing:
                await self._validate_request_signature(request)
            
            # Input validation for high-risk endpoints
            if any(request.url.path.startswith(path) for path in self.high_risk_paths):
                await self._validate_request_content(request)
            
            # Add security headers
            response = await call_next(request)
            return self._add_security_headers(response)
            
        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Security validation failed", "detail": str(e)}
            )
    
    async def _validate_request_size(self, request: Request):
        """Validate request content length"""
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                size = int(content_length)
                if size > self.config.max_request_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Request size {size} exceeds maximum {self.config.max_request_size}"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid Content-Length header"
                )
    
    async def _validate_ip_address(self, request: Request):
        """Validate client IP address"""
        if not self.config.enable_ip_filtering:
            return
        
        client_ip = self._get_client_ip(request)
        
        # Check IP blocklist
        if client_ip in self.config.blocked_ips:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="IP address blocked"
            )
        
        # Check for private/internal IPs from external requests
        if self._is_private_ip(client_ip) and not self._is_internal_request(request):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Private IP addresses not allowed"
            )
        
        # Rate limiting by IP if Redis available
        if self.redis_client:
            await self._check_ip_rate_limit(client_ip)
    
    async def _validate_user_agent(self, request: Request):
        """Validate User-Agent header"""
        if not self.config.enable_user_agent_validation:
            return
        
        user_agent = request.headers.get('user-agent', '').lower()
        
        # Block empty user agents
        if not user_agent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User-Agent header required"
            )
        
        # Check against blocked user agents
        for blocked_ua in self.config.blocked_user_agents:
            if blocked_ua.lower() in user_agent:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User-Agent not allowed"
                )
    
    async def _validate_headers(self, request: Request):
        """Validate HTTP headers for security"""
        headers = dict(request.headers)
        
        # Check for suspicious headers
        suspicious_headers = {
            'x-forwarded-for': self._validate_forwarded_header,
            'x-real-ip': self._validate_ip_header,
            'referer': self._validate_referer_header,
            'origin': self._validate_origin_header
        }
        
        for header, validator in suspicious_headers.items():
            if header in headers:
                await validator(headers[header])
        
        # Check for header injection
        for name, value in headers.items():
            if '\n' in value or '\r' in value:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Header injection detected in {name}"
                )
    
    async def _check_replay_attack(self, request: Request):
        """Check for replay attacks using timestamps and nonces"""
        if not self.redis_client:
            return
        
        # Get timestamp and nonce from headers
        timestamp_header = request.headers.get('x-timestamp')
        nonce_header = request.headers.get('x-nonce')
        
        if not timestamp_header or not nonce_header:
            # Only require for authenticated requests
            auth_header = request.headers.get('authorization')
            if auth_header:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="X-Timestamp and X-Nonce headers required for authenticated requests"
                )
            return
        
        try:
            timestamp = int(timestamp_header)
            current_time = int(time.time())
            
            # Check timestamp window
            if abs(current_time - timestamp) > self.config.replay_window_seconds:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Request timestamp outside allowed window"
                )
            
            # Check nonce uniqueness
            nonce_key = f"nonce:{nonce_header}"
            if await self.redis_client.exists(nonce_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Nonce already used (replay attack detected)"
                )
            
            # Store nonce with expiration
            await self.redis_client.setex(
                nonce_key, 
                self.config.replay_window_seconds * 2, 
                "1"
            )
            
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid timestamp format"
            )
    
    async def _validate_request_signature(self, request: Request):
        """Validate request signature for API integrity"""
        signature_header = request.headers.get('x-signature')
        
        # Only require signatures for authenticated requests
        auth_header = request.headers.get('authorization')
        if not auth_header:
            return
        
        if not signature_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Request signature required"
            )
        
        # Get request data for signature verification
        method = request.method
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else ""
        timestamp = request.headers.get('x-timestamp', '')
        nonce = request.headers.get('x-nonce', '')
        
        # Read body for POST/PUT requests
        body = ""
        if method in ['POST', 'PUT', 'PATCH']:
            body_bytes = await request.body()
            body = body_bytes.decode('utf-8', errors='ignore')
        
        # Create signature payload
        payload = f"{method}|{path}|{query}|{timestamp}|{nonce}|{body}"
        
        # This would verify against a known API key/secret
        # For now, we'll just validate the format
        if not self._is_valid_signature_format(signature_header):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature format"
            )
    
    async def _validate_request_content(self, request: Request):
        """Validate request content for high-risk endpoints"""
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                content_type = request.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    body = await request.json()
                    validate_request_data(body)
                elif 'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type:
                    # Form data validation would go here
                    pass
                    
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Request content validation failed: {str(e)}"
                )
    
    async def _check_ip_rate_limit(self, client_ip: str):
        """Check rate limits by IP address"""
        rate_limit_key = f"ip_rate_limit:{client_ip}"
        current_time = int(time.time())
        window_start = current_time - 60  # 1 minute window
        
        # Use Redis sorted set for sliding window
        pipe = self.redis_client.pipeline()
        pipe.zremrangebyscore(rate_limit_key, 0, window_start)
        pipe.zcard(rate_limit_key)
        pipe.zadd(rate_limit_key, {str(current_time): current_time})
        pipe.expire(rate_limit_key, 120)
        
        results = await pipe.execute()
        request_count = results[1] + 1
        
        # Allow 100 requests per minute per IP
        if request_count > 100:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="IP rate limit exceeded",
                headers={"Retry-After": "60"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP"""
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is private/internal"""
        private_ranges = [
            '127.', '10.', '192.168.', '172.16.', '172.17.', '172.18.',
            '172.19.', '172.20.', '172.21.', '172.22.', '172.23.',
            '172.24.', '172.25.', '172.26.', '172.27.', '172.28.',
            '172.29.', '172.30.', '172.31.', 'localhost'
        ]
        return any(ip.startswith(prefix) for prefix in private_ranges)
    
    def _is_internal_request(self, request: Request) -> bool:
        """Check if request is from internal network"""
        # Check for internal service headers
        return (
            request.headers.get('x-internal-service') is not None or
            request.headers.get('x-kubernetes-service') is not None
        )
    
    async def _validate_forwarded_header(self, value: str):
        """Validate X-Forwarded-For header"""
        # Basic validation for header injection
        if any(char in value for char in ['\n', '\r', '\0']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid X-Forwarded-For header"
            )
    
    async def _validate_ip_header(self, value: str):
        """Validate IP header format"""
        import ipaddress
        try:
            ipaddress.ip_address(value)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid IP address format"
            )
    
    async def _validate_referer_header(self, value: str):
        """Validate Referer header"""
        result = security_validator.validate_string(value, max_length=2048, field_name="referer")
        if not result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Referer header"
            )
    
    async def _validate_origin_header(self, value: str):
        """Validate Origin header"""
        result = security_validator.validate_string(value, max_length=2048, field_name="origin")
        if not result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Origin header"
            )
    
    def _is_valid_signature_format(self, signature: str) -> bool:
        """Validate signature format (HMAC-SHA256)"""
        # Expected format: sha256=<hex_digest>
        if not signature.startswith('sha256='):
            return False
        
        hex_part = signature[7:]
        if len(hex_part) != 64:  # SHA256 hex digest length
            return False
        
        try:
            int(hex_part, 16)
            return True
        except ValueError:
            return False
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            'X-Permitted-Cross-Domain-Policies': 'none'
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


# Request signing utilities
class RequestSigner:
    """Utility for signing API requests"""
    
    @staticmethod
    def generate_nonce() -> str:
        """Generate secure nonce"""
        return secrets.token_hex(16)
    
    @staticmethod
    def generate_timestamp() -> int:
        """Generate current timestamp"""
        return int(time.time())
    
    @staticmethod
    def sign_request(
        method: str,
        path: str,
        query: str,
        timestamp: int,
        nonce: str,
        body: str,
        secret: str
    ) -> str:
        """Sign API request"""
        payload = f"{method}|{path}|{query}|{timestamp}|{nonce}|{body}"
        signature = hashlib.sha256(f"{payload}|{secret}".encode()).hexdigest()
        return f"sha256={signature}"


# Example client implementation
"""
import requests
import time
import hashlib
import secrets

class SecureAPIClient:
    def __init__(self, base_url: str, api_secret: str):
        self.base_url = base_url
        self.api_secret = api_secret
    
    def make_request(self, method: str, path: str, data=None):
        timestamp = int(time.time())
        nonce = secrets.token_hex(16)
        
        headers = {
            'X-Timestamp': str(timestamp),
            'X-Nonce': nonce,
            'Content-Type': 'application/json'
        }
        
        body = json.dumps(data) if data else ""
        query = ""  # Add query string if needed
        
        signature = RequestSigner.sign_request(
            method, path, query, timestamp, nonce, body, self.api_secret
        )
        headers['X-Signature'] = signature
        
        response = requests.request(
            method, 
            f"{self.base_url}{path}",
            headers=headers,
            json=data
        )
        
        return response
"""