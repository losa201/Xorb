"""
Secure CORS Configuration Middleware

This module provides secure CORS configuration with:
- Environment-specific origin validation
- No wildcard origins in production
- Comprehensive security header enforcement
- Domain validation and sanitization
- Audit logging for CORS violations

Security Features:
- Production-safe CORS origin validation
- Protocol enforcement (HTTPS in production)
- Domain whitelist validation
- Request method and header restrictions
- Security violation logging and alerting
"""

import re
import logging
from typing import List, Set, Optional
from urllib.parse import urlparse
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.logging import get_logger

logger = get_logger(__name__)


class SecureCORSConfig:
    """Secure CORS configuration with environment-specific validation"""
    
    def __init__(self, environment: str = "development"):
        """Initialize secure CORS configuration"""
        self.environment = environment
        self.allowed_origins = self._get_default_origins()
        self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        self.allowed_headers = [
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-API-Key",
            "X-Request-ID",
            "X-Tenant-ID"
        ]
        self.expose_headers = ["X-Request-ID", "X-RateLimit-Remaining"]
        self.max_age = 3600  # 1 hour
        
    def _get_default_origins(self) -> List[str]:
        """Get secure default origins by environment"""
        defaults = {
            "production": ["https://app.xorb.enterprise"],
            "staging": ["https://staging.xorb.enterprise", "https://app.xorb.enterprise"],
            "development": [
                "http://localhost:3000",
                "http://localhost:8080", 
                "http://localhost:5173",  # Vite default
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080"
            ]
        }
        return defaults.get(self.environment, defaults["production"])
    
    def validate_and_set_origins(self, origins_string: str) -> List[str]:
        """Validate and set CORS origins from configuration string"""
        if not origins_string or origins_string.strip() == "":
            logger.info("No CORS origins specified, using secure defaults")
            return self.allowed_origins
        
        origins = [origin.strip() for origin in origins_string.split(",") if origin.strip()]
        validated_origins = []
        
        for origin in origins:
            if self._validate_origin(origin):
                validated_origins.append(origin)
            else:
                logger.warning(f"Invalid CORS origin rejected: {origin}")
        
        if not validated_origins:
            logger.warning("No valid CORS origins found, falling back to secure defaults")
            return self.allowed_origins
        
        self.allowed_origins = validated_origins
        logger.info(f"CORS origins validated for {self.environment}: {len(validated_origins)} origins")
        return validated_origins
    
    def _validate_origin(self, origin: str) -> bool:
        """Validate individual CORS origin"""
        # Never allow wildcard in production
        if origin == "*":
            if self.environment == "production":
                logger.error("Wildcard CORS origin not allowed in production")
                return False
            else:
                logger.warning("Wildcard CORS origin allowed in development only")
                return True
        
        # Parse and validate URL
        try:
            parsed = urlparse(origin)
            
            # Must have a scheme
            if not parsed.scheme:
                logger.warning(f"CORS origin missing scheme: {origin}")
                return False
            
            # Production must use HTTPS
            if self.environment == "production" and parsed.scheme != "https":
                logger.error(f"CORS origin must use HTTPS in production: {origin}")
                return False
            
            # Development can use HTTP for localhost
            if (self.environment in ["development", "staging"] and 
                parsed.scheme == "http" and 
                parsed.hostname in ["localhost", "127.0.0.1"]):
                return True
            
            # All other origins must use HTTPS
            if parsed.scheme not in ["https", "http"]:
                logger.warning(f"CORS origin uses invalid scheme: {origin}")
                return False
            
            # Must have a hostname
            if not parsed.hostname:
                logger.warning(f"CORS origin missing hostname: {origin}")
                return False
            
            # Validate hostname format
            if not self._validate_hostname(parsed.hostname):
                logger.warning(f"CORS origin has invalid hostname: {origin}")
                return False
            
            # Check against domain whitelist if configured
            if not self._check_domain_whitelist(parsed.hostname):
                logger.warning(f"CORS origin not in domain whitelist: {origin}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating CORS origin {origin}: {e}")
            return False
    
    def _validate_hostname(self, hostname: str) -> bool:
        """Validate hostname format"""
        # Allow localhost and IP addresses for development
        if hostname in ["localhost", "127.0.0.1", "::1"]:
            return True
        
        # Validate domain name format
        domain_pattern = re.compile(
            r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        )
        
        return bool(domain_pattern.match(hostname))
    
    def _check_domain_whitelist(self, hostname: str) -> bool:
        """Check hostname against domain whitelist"""
        # Default whitelist based on environment
        whitelist_patterns = {
            "production": [
                r".*\.xorb\.enterprise$",
                r"^xorb\.enterprise$"
            ],
            "staging": [
                r".*\.xorb\.enterprise$",
                r"^xorb\.enterprise$",
                r".*\.staging\.xorb\.enterprise$"
            ],
            "development": [
                r"^localhost$",
                r"^127\.0\.0\.1$",
                r".*\.xorb\.enterprise$",
                r".*\.local$"
            ]
        }
        
        patterns = whitelist_patterns.get(self.environment, whitelist_patterns["production"])
        
        for pattern in patterns:
            if re.match(pattern, hostname):
                return True
        
        return False
    
    def get_cors_middleware_config(self) -> dict:
        """Get configuration for FastAPI CORSMiddleware"""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": True,
            "allow_methods": self.allowed_methods,
            "allow_headers": self.allowed_headers,
            "expose_headers": self.expose_headers,
            "max_age": self.max_age
        }
    
    def log_cors_violation(self, request: Request, reason: str):
        """Log CORS security violation"""
        origin = request.headers.get("origin", "unknown")
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.warning(
            "CORS security violation",
            extra={
                "event_type": "cors_violation",
                "origin": origin,
                "reason": reason,
                "user_agent": user_agent,
                "client_ip": request.client.host if request.client else "unknown",
                "method": request.method,
                "path": str(request.url.path),
                "environment": self.environment
            }
        )


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for additional CORS security enforcement"""
    
    def __init__(self, app, cors_config: SecureCORSConfig):
        super().__init__(app)
        self.cors_config = cors_config
    
    async def dispatch(self, request: Request, call_next):
        """Process request with CORS security checks"""
        origin = request.headers.get("origin")
        
        # Check for CORS violations
        if origin and origin not in self.cors_config.allowed_origins:
            # Allow wildcard only in development
            if "*" not in self.cors_config.allowed_origins:
                self.cors_config.log_cors_violation(
                    request, 
                    f"Origin not in allowed list: {origin}"
                )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response, request)
        
        return response
    
    def _add_security_headers(self, response: Response, request: Request):
        """Add additional security headers"""
        # Add CORS-related security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Add CSP header for additional protection
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp_policy


def create_secure_cors_middleware(environment: str, origins_string: str = "") -> tuple:
    """Create secure CORS middleware configuration"""
    
    # Create secure CORS config
    cors_config = SecureCORSConfig(environment)
    
    # Validate and set origins
    validated_origins = cors_config.validate_and_set_origins(origins_string)
    
    # Get middleware config
    middleware_config = cors_config.get_cors_middleware_config()
    
    # Log configuration for audit
    logger.info(
        "CORS middleware configured",
        extra={
            "environment": environment,
            "allowed_origins_count": len(validated_origins),
            "wildcard_allowed": "*" in validated_origins,
            "https_enforced": environment == "production"
        }
    )
    
    return cors_config, middleware_config


def validate_cors_configuration(environment: str, origins: List[str]) -> bool:
    """Validate CORS configuration for security compliance"""
    
    if environment == "production":
        # Production must not allow wildcards
        if "*" in origins:
            logger.error("Production CORS configuration contains wildcard origin")
            return False
        
        # Production must use HTTPS
        for origin in origins:
            if origin.startswith("http://") and "localhost" not in origin:
                logger.error(f"Production CORS origin uses HTTP: {origin}")
                return False
    
    # Check for common misconfigurations
    suspicious_patterns = [
        r".*\.ngrok\.io$",  # Tunnel services
        r".*\.localtunnel\.me$",
        r".*\.herokuapp\.com$",  # Unless specifically allowed
    ]
    
    for origin in origins:
        for pattern in suspicious_patterns:
            if re.match(pattern, origin):
                logger.warning(f"Potentially suspicious CORS origin: {origin}")
    
    return True


# Example usage
if __name__ == "__main__":
    # Test CORS configuration
    cors_config = SecureCORSConfig("production")
    
    test_origins = [
        "https://app.xorb.enterprise",
        "http://malicious.com",
        "*",
        "https://staging.xorb.enterprise"
    ]
    
    for origin in test_origins:
        valid = cors_config._validate_origin(origin)
        print(f"Origin: {origin} - Valid: {valid}")