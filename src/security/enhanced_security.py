"""
XORB Enhanced Security Module
Implements comprehensive security features including request validation, rate limiting, and compliance monitoring
"""

import logging
import time
import hashlib
import ipaddress
from functools import wraps
from typing import Callable, Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

try:
    import aioredis
    from aiohttp import web
    
    AIORedisAvailable = True
except ImportError:
    AIORedisAvailable = False

logger = logging.getLogger("XORB-EnhancedSecurity")

class SecurityConfig:
    """Security configuration settings"""
    
    # Request validation settings
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_PARAM_LENGTH = 2048
    MAX_HEADER_VALUE_LENGTH = 4096
    MAX_COOKIE_LENGTH = 4096
    
    # Rate limiting settings
    RATE_LIMITING = {
        "default": {
            "requests": 100,
            "window": 60,  # 100 requests per minute
            "block_time": 300  # 5 minutes block
        },
        "authenticated": {
            "requests": 500,
            "window": 60,  # 500 requests per minute
            "block_time": 60  # 1 minute block
        },
        "api_key": {
            "requests": 1000,
            "window": 60,  # 1000 requests per minute
            "block_time": 60  # 1 minute block
        }
    }
    
    # Security headers
    @staticmethod
    def get_secure_headers() -> Dict[str, str]:
        """Get comprehensive security headers"""
        return {
            'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Content-Security-Policy': """
                default-src 'none';
                script-src 'self';
                connect-src 'self';
                img-src 'self' data:;
                style-src 'self';
                font-src 'self';
                object-src 'none';
                base-uri 'self';
                frame-ancestors 'none';
                form-action 'self';
                upgrade-insecure-requests;
            """.strip(),
            'Referrer-Policy': 'no-referrer',
            'Permissions-Policy': (
                'geolocation=(), microphone=(), camera=(), magnetometer=(), gyroscope=(), payment=()'
            ),
            'X-Permitted-Cross-Domain-Policies': 'none',
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Request-ID': 'generated',  # Will be replaced with actual ID
            'X-Security-Timestamp': str(int(time.time())),
            'X-Security-ID': 'XORB-SEC-2025-Q3'
        }
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        "../", "..%2f", "%2e%2e", "script", "onerror", "onload",
        "eval", "system(", "phpinfo", "base64", "cmd=", "exec=",
        "passthru", "shell_exec", "window\.", "document\.",
        "alert\(", "prompt\(", "confirm\(", "eval\(",
        "expression\(", "javascript:", "vbscript:", "data:",
        "file:", "gopher:", "expect:", "cmd:", "ftp:",
        "ssrf", "xxe", "xss", "csrf", "lfi", "rfi", "rce"
    ]
    
    # Suspicious file extensions
    SUSPICIOUS_EXTENSIONS = [
        ".php", ".asp", ".aspx", ".jsp", ".exe", ".bat", ".sh",
        ".cgi", ".pl", ".py", ".rb", ".jsp", ".jspx", ".php5", ".phtml"
    ]

class RequestValidator:
    """Validates incoming requests for security compliance"""
    
    def __init__(self):
        self.logger = logging.getLogger("XORB-RequestValidator")
        
    def validate_request(self, request: web.Request) -> bool:
        """Validate an incoming request"""
        try:
            # Validate request size
            if not self._validate_request_size(request):
                return False
            
            # Validate request path
            if not self._validate_request_path(request.path):
                return False
            
            # Validate request headers
            if not self._validate_request_headers(request.headers):
                return False
            
            # Validate request cookies
            if not self._validate_request_cookies(request.cookies):
                return False
            
            # Validate request parameters
            if not self._validate_request_params(request):
                return False
            
            # Validate content type if present
            if request.headers.get('Content-Type'):
                if not self._validate_content_type(request.headers['Content-Type']):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Request validation error: {str(e)}")
            return False
    
    def _validate_request_size(self, request: web.Request) -> bool:
        """Validate the size of the request"""
        if request.content_length and request.content_length > SecurityConfig.MAX_REQUEST_SIZE:
            self.logger.warning("⚠️ Request too large")
            return False
        return True
    
    def _validate_request_path(self, path: str) -> bool:
        """Validate the request path for suspicious patterns"""
        # Check for path traversal attempts
        if any(pattern in path.lower() for pattern in SecurityConfig.SUSPICIOUS_EXTENSIONS):
            self.logger.warning(f"⚠️ Suspicious file extension in path: {path}")
            return False
            
        # Check for path traversal patterns
        if any(pattern in path.lower() for pattern in ["../", "..%2f", "%2e%2e"]):
            self.logger.warning(f"⚠️ Path traversal attempt: {path}")
            return False
            
        return True
    
    def _validate_request_headers(self, headers: Dict[str, str]) -> bool:
        """Validate request headers for suspicious content"""
        for header, value in headers.items():
            # Check header value length
            if len(value) > SecurityConfig.MAX_HEADER_VALUE_LENGTH:
                self.logger.warning(f"⚠️ Header value too long: {header}")
                return False
                
            # Check for suspicious patterns
            if any(pattern.lower() in value.lower() for pattern in SecurityConfig.SUSPICIOUS_PATTERNS):
                self.logger.warning(f"⚠️ Suspicious header detected: {header}: {value}")
                return False
            
        return True
    
    def _validate_request_cookies(self, cookies: Dict[str, str]) -> bool:
        """Validate request cookies for suspicious content"""
        for cookie_name, cookie_value in cookies.items():
            # Check cookie length
            if len(cookie_name) + len(cookie_value) > SecurityConfig.MAX_COOKIE_LENGTH:
                self.logger.warning(f"⚠️ Cookie too long: {cookie_name}")
                return False
                
            # Check for suspicious patterns
            if any(pattern.lower() in cookie_value.lower() for pattern in SecurityConfig.SUSPICIOUS_PATTERNS):
                self.logger.warning(f"⚠️ Suspicious cookie detected: {cookie_name}: {cookie_value}")
                return False
            
        return True
    
    def _validate_request_params(self, request: web.Request) -> bool:
        """Validate request parameters"""
        # Validate query parameters
        for param, value in request.query.items():
            if len(param) > SecurityConfig.MAX_PARAM_LENGTH or len(value) > SecurityConfig.MAX_PARAM_LENGTH:
                self.logger.warning(f"⚠️ Parameter too long: {param}")
                return False
                
            if any(pattern.lower() in value.lower() for pattern in SecurityConfig.SUSPICIOUS_PATTERNS):
                self.logger.warning(f"⚠️ Suspicious parameter detected: {param}: {value}")
                return False
            
        # Validate post parameters if present
        if request.has_body:
            try:
                async def validate_post_params():
                    post_data = await request.post()
                    for param, value in post_data.items():
                        if len(param) > SecurityConfig.MAX_PARAM_LENGTH or len(str(value)) > SecurityConfig.MAX_PARAM_LENGTH:
                            self.logger.warning(f"⚠️ Post parameter too long: {param}")
                            return False
                            
                        if any(pattern.lower() in str(value).lower() for pattern in SecurityConfig.SUSPICIOUS_PATTERNS):
                            self.logger.warning(f"⚠️ Suspicious post parameter detected: {param}: {value}")
                            return False
                    return True
                
                return validate_post_params()
                
            except Exception as e:
                self.logger.error(f"Error validating post parameters: {str(e)}")
                return False
                
        return True
    
    def _validate_content_type(self, content_type: str) -> bool:
        """Validate content type header"""
        # Common dangerous content types
        dangerous_content_types = [
            "application/x-php",
            "text/x-php",
            "application/x-shockwave-flash",
            "application/x-msdownload"
        ]
        
        if any(ct in content_type.lower() for ct in dangerous_content_types):
            self.logger.warning(f"⚠️ Dangerous content type detected: {content_type}")
            return False
        
        return True

class RateLimiter:
    """Rate limiting implementation using Redis"""
    
    def __init__(self, redis_url: str = "redis://localhost", redis_password: str = None):
        self.redis_url = redis_url
        self.redis_password = redis_password
        self.redis = None
        self.logger = logging.getLogger("XORB-RateLimiter")
        self.enabled = AIORedisAvailable
        
        if not self.enabled:
            self.logger.warning("⚠️ aioredis not available - rate limiting disabled")
        
    async def initialize(self):
        """Initialize Redis connection"""
        if not self.enabled:
            return
            
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                password=self.redis_password,
                decode_responses=True
            )
            await self.redis.ping()
            self.logger.info("Redis connection established for rate limiting")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {str(e)}")
            self.enabled = False
    
    async def check_rate_limit(self, request: web.Request) -> bool:
        """Check if request is within rate limits"""
        if not self.enabled:
            return True
            
        try:
            # Get client identifier
            client_id = self._get_client_id(request)
            
            # Get rate limiting configuration
            rate_config = self._get_rate_config(request)
            
            # Create Redis key
            key = f"rate_limit:{client_id}:{int(time.time() / rate_config['window'])}"
            
            # Increment counter
            current = await self.redis.incr(key)
            
            # Set expiration if this is a new key
            if current == 1:
                await self.redis.expire(key, rate_config['window'] + 10)  # Add buffer
            
            # Check if limit exceeded
            if current > rate_config['requests']:
                self.logger.warning(f"Rate limit exceeded for {client_id}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {str(e)}")
            return True  # Allow request on error
    
    def _get_client_id(self, request: web.Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get API key from headers
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()}"
        
        # Try to get user from request (if authenticated)
        if hasattr(request, 'user') and request.user:
            return f"user:{request.user.get('id', 'unknown')}"
        
        # Fallback to IP address
        try:
            # Handle proxy headers
            ip = request.headers.get('X-Forwarded-For', request.remote)
            # Validate and normalize IP
            ip_obj = ipaddress.ip_address(ip.split(',')[0].strip())
            return f"ip:{ip_obj.compressed}"
        except:
            return "ip:unknown"
    
    def _get_rate_config(self, request: web.Request) -> Dict[str, int]:
        """Get rate limiting configuration based on request"""
        # Check for API key
        if request.headers.get('X-API-Key'):
            return SecurityConfig.RATE_LIMITING.get('api_key', SecurityConfig.RATE_LIMITING['default'])
        
        # Check for authenticated user
        if hasattr(request, 'user') and request.user:
            return SecurityConfig.RATE_LIMITING.get('authenticated', SecurityConfig.RATE_LIMITING['default'])
        
        # Default rate limiting
        return SecurityConfig.RATE_LIMITING['default']

class ComplianceMonitor:
    """Monitors requests for compliance with security policies"""
    
    def __init__(self):
        self.logger = logging.getLogger("XORB-ComplianceMonitor")
        
    async def check_compliance(self, request: web.Request) -> bool:
        """Check request against compliance policies"""
        try:
            # Check for PCI-DSS compliance
            if not await self._check_pci_dss(request):
                return False
            
            # Check for HIPAA compliance
            if not await self._check_hipaa(request):
                return False
            
            # Check for ISO 27001 compliance
            if not await self._check_iso_27001(request):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Compliance check error: {str(e)}")
            return False
    
    async def _check_pci_dss(self, request: web.Request) -> bool:
        """Check request against PCI-DSS requirements"""
        # Check for potential credit card numbers
        if request.has_body:
            try:
                post_data = await request.post()
                for key, value in post_data.items():
                    if self._contains_credit_card(str(value)):
                        self.logger.warning(f"PCI-DSS violation: Credit card number detected in field {key}")
                        return False
            except Exception as e:
                self.logger.error(f"PCI-DSS check error: {str(e)}")
                return False
        
        return True
    
    async def _check_hipaa(self, request: web.Request) -> bool:
        """Check request against HIPAA requirements"""
        # Check for potential PHI (Protected Health Information)
        if request.has_body:
            try:
                post_data = await request.post()
                for key, value in post_data.items():
                    if self._contains_phi(str(value)):
                        self.logger.warning(f"HIPAA violation: PHI detected in field {key}")
                        return False
            except Exception as e:
                self.logger.error(f"HIPAA check error: {str(e)}")
                return False
        
        return True
    
    async def _check_iso_27001(self, request: web.Request) -> bool:
        """Check request against ISO 27001 requirements"""
        # Check for potential sensitive information
        if request.has_body:
            try:
                post_data = await request.post()
                for key, value in post_data.items():
                    if self._contains_sensitive_data(str(value)):
                        self.logger.warning(f"ISO 27001 violation: Sensitive data detected in field {key}")
                        return False
            except Exception as e:
                self.logger.error(f"ISO 27001 check error: {str(e)}")
                return False
        
        return True
    
    def _contains_credit_card(self, text: str) -> bool:
        """Check if text contains a credit card number"""
        # Simple Luhn algorithm check
        import re
        
        # Remove non-numeric characters
        cleaned = re.sub(r'[^0-9]', '', text)
        
        # Check for 13-19 digit numbers
        if not (13 <= len(cleaned) <= 19):
            return False
        
        # Luhn algorithm
        def luhn_check(card_number):
            sum_ = 0
            num_digits = len(card_number)
            oddeven = num_digits & 1
            
            for i in range(num_digits):
                digit = int(card_number[i])
                if i & 1 == oddeven:
                    digit *= 2
                if digit > 9:
                    digit -= 9
                sum_ += digit
            
            return sum_ % 10 == 0
        
        return luhn_check(cleaned)
    
    def _contains_phi(self, text: str) -> bool:
        """Check if text contains potential PHI"""
        # Simple pattern matching for PHI
        phi_patterns = [
            # Social Security Numbers (SSN)
            r"\d{3}-\d{2}-\d{4}",
            # Dates (potential birth dates)
            r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b",
            # Phone numbers (potential patient contact info)
            r"\(?(\d{3})\)?[- ]?(\d{3})[- ]?(\d{4})",
            # Email addresses (potential patient contact info)
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        ]
        
        import re
        for pattern in phi_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _contains_sensitive_data(self, text: str) -> bool:
        """Check if text contains potential sensitive information"""
        # Check for potential secrets
        secret_patterns = [
            # API keys
            r"[aA][pP][iI]_?[kK][eE][yY].{0,20}=[\s"']{1,2}[\w\d]{32,}",
            # Passwords
            r"[pP][aA][sS][sS][wW][oO][rR][dD].{0,20}=[\s"']{1,2}[\w\d]{8,}",
            # Private keys
            r"-----BEGIN PRIVATE KEY-----",
            # AWS keys
            r"[a-zA-Z0-9]{20}",
            # Google API keys
            r"AIza[0-9A-Za-z\_\-]{35}",
            # Facebook client secrets
            r"[a-f0-9]{32}",
            # Twitter keys
            r"[1-9][0-9]{1,10}-[a-zA-Z0-9]{20,}",
            # Generic 32-character hex
            r"[a-fA-F0-9]{32}"
        ]
        
        import re
        for pattern in secret_patterns:
            if re.search(pattern, text):
                return True
        
        return False

class SecurityMiddleware:
    """Security middleware that combines request validation, rate limiting, and compliance monitoring"""
    
    def __init__(self, app, redis_url: str = "redis://localhost", redis_password: str = None):
        self.app = app
        self.logger = logging.getLogger("XORB-SecurityMiddleware")
        self.validator = RequestValidator()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(redis_url, redis_password)
        
        # Initialize compliance monitor
        self.compliance_monitor = ComplianceMonitor()
        
    async def initialize(self):
        """Initialize security middleware components"""
        # Initialize rate limiter
        await self.rate_limiter.initialize()
        
    def __call__(self, app):
        """Add security middleware to the application"""
        app.middlewares.append(self.security_middleware)
        return app
    
    async def security_middleware(self, handler):
        """Security middleware to process requests"""
        @wraps(handler)
        async def middleware_handler(request):
            # Start time for request timing
            start_time = time.time()
            
            # Log request details
            self.logger.debug(f"📥 Request: {request.method} {request.path}")
            
            # Validate request
            if not self.validator.validate_request(request):
                self.logger.warning(f"🚫 Invalid request: {request.method} {request.path}")
                return self._build_security_response(400, "Bad Request")
                
            # Check rate limiting
            if not await self.rate_limiter.check_rate_limit(request):
                self.logger.warning(f"🚫 Rate limit exceeded: {request.method} {request.path}")
                return self._build_security_response(429, "Too Many Requests")
                
            # Check compliance
            if not await self.compliance_monitor.check_compliance(request):
                self.logger.warning(f"🚫 Compliance violation: {request.method} {request.path}")
                return self._build_security_response(403, "Forbidden")
                
            # Add security headers
            response = await handler(request)
            
            # Add security headers
            headers = SecurityConfig.get_secure_headers()
            # Replace X-Request-ID with a unique ID for this request
            headers['X-Request-ID'] = hashlib.sha256(f"{time.time()}{request.path}{request.remote}".encode()).hexdigest()[:32]
            
            for header, value in headers.items():
                response.headers[header] = value
                
            # Log response details
            duration = (time.time() - start_time) * 1000  # in milliseconds
            self.logger.debug(f"📤 Response: {response.status} in {duration:.2f}ms")
            
            return response
            
        return middleware_handler
    
    def _build_security_response(self, status_code: int, message: str):
        """Build a secure response with security headers"""
        from aiohttp import web
        
        response = web.Response(
            text=f"{message}\n",
            status=status_code,
            content_type='text/plain'
        )
        
        # Add security headers
        headers = SecurityConfig.get_secure_headers()
        # Replace X-Request-ID with a unique ID for this response
        headers['X-Request-ID'] = hashlib.sha256(f"{time.time()}error{status_code}".encode()).hexdigest()[:32]
        
        for header, value in headers.items():
            response.headers[header] = value
            
        return response

# For backward compatibility with existing code
async def check_request_safety(request) -> bool:
    """Check if request is safe to process"""
    validator = RequestValidator()
    return validator.validate_request(request)