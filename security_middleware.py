"""
XORB Security Middleware
Implements comprehensive security headers and secure defaults
"""

import logging
import time
from functools import wraps
from typing import Callable, Dict, Any

logger = logging.getLogger("XORB-Security")

class SecurityHeaders:
    """Security headers for HTTP responses"""
    
    @staticmethod
    def get_secure_headers() -> Dict[str, str]:
        """Get comprehensive security headers"""
        return {
            'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',
            'X-Content-Type-Options': 'nosniff',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Content-Security-Policy': """
                default-src 'none';
                script-src 'self' 'unsafe-inline' 'unsafe-eval';
                connect-src 'self';
                img-src 'self' data:;
                style-src 'self' 'unsafe-inline';
                font-src 'self';
                object-src 'none';
                base-uri 'self';
                frame-ancestors 'none';
                form-action 'self';
                upgrade-insecure-requests;
            """.strip(),
            'Referrer-Policy': 'no-referrer-when-downgrade',
            'Permissions-Policy': (
                'geolocation=(), microphone=(), camera=(), magnetometer=(), gyroscope=(), payment=()'
            ),
            'X-Permitted-Cross-Domain-Policies': 'none',
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    
    @classmethod
    def apply_headers(cls, handler: Callable) -> Callable:
        """Apply security headers to a web handler"""
        @wraps(handler)
        async def wrapper(request, *args, **kwargs):
            response = await handler(request, *args, **kwargs)
            
            # Add security headers
            headers = cls.get_secure_headers()
            for header, value in headers.items():
                response.headers[header] = value
                
            # Add security timestamp
            response.headers['X-Request-Timestamp'] = str(int(time.time()))
            
            # Add security ID
            response.headers['X-Security-ID'] = 'XORB-SEC-2025-Q3'
            
            return response
        return wrapper


class SecurityMiddleware:
    """Security middleware for all services"""
    
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger("XORB-Security-Middleware")
        
    def setup(self):
        """Setup security middleware"""
        # Add security headers to all responses
        self.app.middlewares.append(self.security_middleware)
        
        # Setup security logging
        self.logger.info("🔒 Security middleware initialized")
        
        return self.app
    
    async def security_middleware(self, handler):
        """Security middleware to process requests"""
        async def middleware_handler(request):
            # Start time for request timing
            start_time = time.time()
            
            # Log request details
            self.logger.debug(f"📥 Request: {request.method} {request.path}")
            
            # Check for security threats
            if not await self.check_request_safety(request):
                self.logger.warning(f"🚫 Blocked request: {request.method} {request.path}")
                return self._build_security_response(403, "Forbidden")
                
            # Validate request headers
            if not self.validate_request_headers(request):
                self.logger.warning(f"🚫 Invalid headers: {request.method} {request.path}")
                return self._build_security_response(400, "Bad Request")
                
            # Validate request parameters
            if not await self.validate_request_params(request):
                self.logger.warning(f"🚫 Invalid parameters: {request.method} {request.path}")
                return self._build_security_response(400, "Bad Request")
            
            # Process the request
            try:
                response = await handler(request)
                
                # Add security headers
                SecurityHeaders.apply_headers(handler)
                
                # Log response details
                duration = (time.time() - start_time) * 1000  # in milliseconds
                self.logger.debug(f"📤 Response: {response.status} in {duration:.2f}ms")
                
                return response
                
            except Exception as e:
                # Handle security-related exceptions
                self.logger.error(f"❌ Security error: {str(e)}")
                return self._build_security_response(500, "Internal Server Error")
        
        return middleware_handler
    
    async def check_request_safety(self, request) -> bool:
        """Check if request is safe to process"""
        try:
            # Check request size
            if request.content_length and request.content_length > 1024 * 1024 * 10:  # 10MB
                self.logger.warning("⚠️ Request too large")
                return False
            
            # Check for suspicious headers
            if self._check_suspicious_headers(request.headers):
                return False
            
            # Check for suspicious path
            if self._check_suspicious_path(request.path):
                return False
            
            # Validate request content
            if not await self.validate_request_content(request):
                return False
            
            # Check rate limiting
            if not await self._check_rate_limiting(request):
                return False
            
            # Trigger compliance monitoring
            await self._trigger_compliance_check(request)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security check error: {str(e)}")
            return False
    
    def _check_suspicious_headers(self, headers) -> bool:
        """Check for suspicious header patterns"""
        # Check for common attack patterns in headers
        suspicious_patterns = [
            "../", "..%2f", "%2e%2e", "script", "onerror", "onload",
            "eval", "system(", "phpinfo", "base64", "cmd=", "exec=",
            "passthru", "shell_exec", "window\.", "document\.",
            "alert\(", "prompt\(", "confirm\(", "eval\(",
            "expression\(", "javascript:", "vbscript:", "data:",
            "file:", "gopher:", "expect:", "cmd:", "ftp:",
            "ssrf", "xxe", "xss", "csrf", "lfi", "rfi", "rce"
        ]
        
        for header, value in headers.items():
            if any(pattern.lower() in str(value).lower() for pattern in suspicious_patterns):
                self.logger.warning(f"⚠️ Suspicious header detected: {header}: {value}")
                return True
        
        return False
    
    def _check_suspicious_path(self, path: str) -> bool:
        """Check for suspicious path patterns"""
        # Check for path traversal attempts
        suspicious_patterns = [
            "../", "..%2f", "%2e%2e", "//", "\\", "~", "%00",
            ".php", ".asp", ".aspx", ".jsp", ".exe", ".bat", ".sh",
            "cgi-bin", "wp-admin", "wp-login", "admin", "login",
            "config", "backup", "sql", "dump", "db", "database"
        ]
        
        if any(pattern.lower() in path.lower() for pattern in suspicious_patterns):
            return True
        
        return False
    
    def _build_security_response(self, status_code: int, message: str):
        """Build a secure response with security headers"""
        from aiohttp import web
        
        response = web.Response(
            text=f"{message}\n",
            status=status_code,
            content_type='text/plain'
        )
        
        # Add security headers
        for header, value in SecurityHeaders.get_secure_headers().items():
            response.headers[header] = value
            
        return response

# Security configuration defaults
SECURE_SETTINGS = {
    "session": {
        "cookie_secure": True,
        "cookie_http_only": True,
        "cookie_samesite": "strict",
        "timeout": 3600,  # 1 hour
        "refresh_timeout": 1800,  # 30 minutes
        "max_age": 86400,  # 24 hours
        "renew_before": 300  # 5 minutes before expiration
    },
    "rate_limiting": {
        "default": {
            "requests": 100,
            "window": 60,  # 100 requests per minute
            "block_time": 300  # 5 minutes block
        },
        "authenticated": {
            "requests": 500,
            "window": 60,  # 500 requests per minute
            "block_time": 60  # 1 minute block
        }
    },
    "encryption": {
        "default_algorithm": "AES-256-GCM",
        "key_length": 256,
        "hash_iterations": 100000,
        "hash_algorithm": "SHA-256",
        "tls_version": "TLSv1.3"
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "json_format": True,
        "redact_fields": [
            "password", "token", "secret", "key", "auth", "credit_card",
            "ssn", "social_security", "passport", "bank_account"
        ]
    }
}

__all__ = ['SecurityHeaders', 'SecurityMiddleware', 'SECURE_SETTINGS']

if __name__ == "__main__":
    # Example usage
    from aiohttp import web
    
    app = web.Application()
    
    # Setup security middleware
    security_middleware = SecurityMiddleware(app)
    app = security_middleware.setup()
    
    # Add routes
    async def hello(request):
        return web.Response(text="Hello, secure world\n")
    
    app.router.add_get('/', hello)
    
    # Run the application
    web.run_app(app, port=8080)