"""
Comprehensive Input Validation Middleware
Sanitizes and validates all incoming requests to prevent injection attacks
"""

import re
import json
import bleach
from typing import Any, Dict, List, Optional, Union
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class SecurityViolation(Exception):
    """Security violation detected in input"""
    pass


class InputSanitizer:
    """Advanced input sanitization and validation"""
    
    def __init__(self):
        # XSS prevention patterns
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'<style[^>]*>.*?</style>',
            r'vbscript:',
            r'data:text/html',
            r'expression\s*\(',
            r'@import',
            r'&\#x[0-9a-f]+;',  # Hex entities
            r'&\#[0-9]+;',      # Decimal entities
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r"('|(\\'))+.*(;|--|#)",
            r"(\s|^)(union|select|insert|update|delete|drop|create|alter|exec|execute)\s",
            r"(\s|^)(or|and)\s+\d+\s*=\s*\d+",
            r"(\s|^)(or|and)\s+['\"][^'\"]*['\"]\s*=\s*['\"][^'\"]*['\"]",
            r"(exec|execute|sp_|xp_)",
            r"(waitfor|delay)",
            r"(benchmark|sleep)\s*\(",
            r"(information_schema|sysobjects|syscolumns)",
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r'[;&|`$()\[\]{}]',
            r'(rm\s+|del\s+|format\s+)',
            r'(\|\s*nc\s|\|\s*netcat\s)',
            r'(wget\s+|curl\s+).*(\||;)',
            r'(/bin/|/usr/bin/|cmd\.exe|powershell)',
            r'(chmod\s+|chown\s+|sudo\s+)',
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r'\.\./|\.\.\.',
            r'%2e%2e%2f|%2e%2e/',
            r'\.\.\\|\.\.\\',
            r'%2e%2e%5c',
        ]
        
        # LDAP injection patterns
        self.ldap_patterns = [
            r'[*()\\]',
            r'&\s*\|',
            r'\|\s*&',
        ]
        
        # NoSQL injection patterns
        self.nosql_patterns = [
            r'\$where',
            r'\$ne|\$gt|\$lt|\$gte|\$lte',
            r'\$regex|\$exists|\$type',
            r'\{\s*\$',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = {
            'xss': [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns],
            'sql': [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_patterns],
            'command': [re.compile(pattern, re.IGNORECASE) for pattern in self.command_patterns],
            'path_traversal': [re.compile(pattern, re.IGNORECASE) for pattern in self.path_traversal_patterns],
            'ldap': [re.compile(pattern, re.IGNORECASE) for pattern in self.ldap_patterns],
            'nosql': [re.compile(pattern, re.IGNORECASE) for pattern in self.nosql_patterns],
        }
        
        # Allowed HTML tags and attributes for content that may contain HTML
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'blockquote', 'code', 'pre'
        ]
        self.allowed_attributes = {
            '*': ['class'],
            'a': ['href', 'title'],
            'abbr': ['title'],
            'acronym': ['title'],
        }
    
    def sanitize_string(self, value: str, field_name: str = "", 
                       allow_html: bool = False, 
                       max_length: Optional[int] = None) -> str:
        """Sanitize a string value"""
        if not isinstance(value, str):
            return str(value)
        
        original_value = value
        
        # Length validation
        if max_length and len(value) > max_length:
            raise SecurityViolation(f"Field '{field_name}' exceeds maximum length of {max_length}")
        
        # Check for malicious patterns
        violations = self._detect_security_violations(value)
        if violations:
            logger.warning("Security violations detected", 
                         field=field_name, 
                         violations=violations,
                         value_preview=value[:100])
            raise SecurityViolation(f"Security violations detected in '{field_name}': {', '.join(violations)}")
        
        # HTML sanitization
        if allow_html:
            value = bleach.clean(
                value,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                protocols=['http', 'https', 'mailto'],
                strip=True
            )
        else:
            # Strip all HTML tags
            value = bleach.clean(value, tags=[], attributes={}, strip=True)
        
        # Additional cleaning
        value = self._clean_control_characters(value)
        value = self._normalize_unicode(value)
        
        # Log if value was modified
        if value != original_value:
            logger.info("Input sanitized", 
                       field=field_name,
                       original_length=len(original_value),
                       sanitized_length=len(value))
        
        return value
    
    def _detect_security_violations(self, value: str) -> List[str]:
        """Detect security violations in input"""
        violations = []
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(value):
                    violations.append(category)
                    break  # One violation per category is enough
        
        return violations
    
    def _clean_control_characters(self, value: str) -> str:
        """Remove control characters except common whitespace"""
        # Allow tab (9), newline (10), carriage return (13), and printable chars
        return ''.join(char for char in value 
                      if ord(char) >= 32 or ord(char) in [9, 10, 13])
    
    def _normalize_unicode(self, value: str) -> str:
        """Normalize unicode to prevent bypass attempts"""
        import unicodedata
        return unicodedata.normalize('NFKC', value)
    
    def sanitize_dict(self, data: Dict[str, Any], 
                     field_rules: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Sanitize dictionary data recursively"""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        field_rules = field_rules or {}
        
        for key, value in data.items():
            # Sanitize the key itself
            clean_key = self.sanitize_string(key, "dict_key", max_length=100)
            
            # Get field-specific rules
            rules = field_rules.get(key, {})
            allow_html = rules.get('allow_html', False)
            max_length = rules.get('max_length')
            
            if isinstance(value, str):
                sanitized[clean_key] = self.sanitize_string(
                    value, key, allow_html=allow_html, max_length=max_length
                )
            elif isinstance(value, dict):
                sanitized[clean_key] = self.sanitize_dict(value, field_rules)
            elif isinstance(value, list):
                sanitized[clean_key] = self.sanitize_list(value, field_rules)
            else:
                sanitized[clean_key] = value
        
        return sanitized
    
    def sanitize_list(self, data: List[Any], 
                     field_rules: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Any]:
        """Sanitize list data recursively"""
        if not isinstance(data, list):
            return data
        
        sanitized = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                sanitized.append(self.sanitize_string(item, f"list_item_{i}"))
            elif isinstance(item, dict):
                sanitized.append(self.sanitize_dict(item, field_rules))
            elif isinstance(item, list):
                sanitized.append(self.sanitize_list(item, field_rules))
            else:
                sanitized.append(item)
        
        return sanitized


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive input validation and sanitization"""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        self.sanitizer = InputSanitizer()
        
        # Configuration
        self.max_request_size = self.config.get('max_request_size', 10 * 1024 * 1024)  # 10MB
        self.max_json_depth = self.config.get('max_json_depth', 10)
        self.validate_headers = self.config.get('validate_headers', True)
        self.validate_query_params = self.config.get('validate_query_params', True)
        self.validate_json_body = self.config.get('validate_json_body', True)
        
        # Skip validation for certain paths
        self.skip_paths = self.config.get('skip_paths', [
            '/docs', '/redoc', '/openapi.json', '/metrics'
        ])
        
        # Field-specific validation rules
        self.field_rules = self.config.get('field_rules', {
            'description': {'allow_html': True, 'max_length': 5000},
            'content': {'allow_html': True, 'max_length': 10000},
            'name': {'max_length': 100},
            'username': {'max_length': 50},
            'email': {'max_length': 255},
            'password': {'max_length': 128},  # Will be hashed anyway
        })
    
    async def dispatch(self, request: Request, call_next):
        """Process request with input validation"""
        
        # Skip validation for certain paths
        if any(request.url.path.startswith(skip_path) for skip_path in self.skip_paths):
            return await call_next(request)
        
        try:
            # Validate request size
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > self.max_request_size:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": "Request too large"}
                )
            
            # Validate headers
            if self.validate_headers:
                await self._validate_headers(request)
            
            # Validate query parameters
            if self.validate_query_params:
                await self._validate_query_params(request)
            
            # Validate JSON body for POST/PUT/PATCH requests
            if (self.validate_json_body and 
                request.method in ['POST', 'PUT', 'PATCH'] and
                'application/json' in request.headers.get('content-type', '')):
                await self._validate_json_body(request)
            
            response = await call_next(request)
            return response
            
        except SecurityViolation as e:
            logger.warning("Security violation blocked", 
                         path=request.url.path,
                         method=request.method,
                         client_ip=request.client.host,
                         violation=str(e))
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": "Invalid input detected",
                    "error_type": "security_violation",
                    "message": "Request contains potentially malicious content"
                }
            )
        
        except Exception as e:
            logger.error("Input validation error", 
                        path=request.url.path,
                        error=str(e))
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Input validation failed"}
            )
    
    async def _validate_headers(self, request: Request):
        """Validate HTTP headers"""
        dangerous_headers = ['x-forwarded-for', 'x-real-ip', 'x-originating-ip']
        
        for header_name, header_value in request.headers.items():
            # Skip dangerous headers that could be used for spoofing
            if header_name.lower() in dangerous_headers:
                continue
            
            # Validate header values
            try:
                clean_value = self.sanitizer.sanitize_string(
                    header_value, f"header_{header_name}", max_length=1000
                )
                # In a real implementation, you might want to update the request headers
                # with sanitized values, but this requires more complex middleware
                
            except SecurityViolation:
                raise SecurityViolation(f"Malicious content in header: {header_name}")
    
    async def _validate_query_params(self, request: Request):
        """Validate query parameters"""
        for param_name, param_value in request.query_params.items():
            try:
                self.sanitizer.sanitize_string(
                    param_value, f"query_{param_name}", max_length=1000
                )
            except SecurityViolation:
                raise SecurityViolation(f"Malicious content in query parameter: {param_name}")
    
    async def _validate_json_body(self, request: Request):
        """Validate JSON request body"""
        try:
            # Read body
            body = await request.body()
            if not body:
                return
            
            # Parse JSON
            try:
                json_data = json.loads(body)
            except json.JSONDecodeError:
                raise SecurityViolation("Invalid JSON format")
            
            # Check JSON depth
            if self._calculate_json_depth(json_data) > self.max_json_depth:
                raise SecurityViolation("JSON nesting too deep")
            
            # Sanitize JSON data
            sanitized_data = self.sanitizer.sanitize_dict(json_data, self.field_rules)
            
            # In a real implementation, you would need to replace the request body
            # with sanitized data, which requires more complex middleware setup
            
        except UnicodeDecodeError:
            raise SecurityViolation("Invalid encoding in request body")
    
    def _calculate_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate the maximum depth of a JSON object"""
        if depth > self.max_json_depth:
            return depth
        
        if isinstance(obj, dict):
            return max([self._calculate_json_depth(v, depth + 1) for v in obj.values()] + [depth])
        elif isinstance(obj, list):
            return max([self._calculate_json_depth(item, depth + 1) for item in obj] + [depth])
        else:
            return depth


# Configuration presets
SECURITY_PRESETS = {
    'strict': {
        'max_request_size': 1 * 1024 * 1024,  # 1MB
        'max_json_depth': 5,
        'validate_headers': True,
        'validate_query_params': True,
        'validate_json_body': True,
        'field_rules': {
            'description': {'allow_html': False, 'max_length': 1000},
            'content': {'allow_html': False, 'max_length': 5000},
            'name': {'max_length': 50},
            'username': {'max_length': 30},
            'email': {'max_length': 255},
        }
    },
    'moderate': {
        'max_request_size': 10 * 1024 * 1024,  # 10MB
        'max_json_depth': 10,
        'validate_headers': True,
        'validate_query_params': True,
        'validate_json_body': True,
        'field_rules': {
            'description': {'allow_html': True, 'max_length': 5000},
            'content': {'allow_html': True, 'max_length': 10000},
            'name': {'max_length': 100},
            'username': {'max_length': 50},
            'email': {'max_length': 255},
        }
    },
    'permissive': {
        'max_request_size': 50 * 1024 * 1024,  # 50MB
        'max_json_depth': 20,
        'validate_headers': False,
        'validate_query_params': True,
        'validate_json_body': True,
        'skip_paths': ['/docs', '/redoc', '/openapi.json', '/metrics', '/upload']
    }
}


def get_validation_config(preset: str = 'moderate') -> Dict[str, Any]:
    """Get validation configuration by preset"""
    return SECURITY_PRESETS.get(preset, SECURITY_PRESETS['moderate'])