"""
Secure Logging Implementation
Prevents sensitive data leakage in logs with automatic sanitization
"""

import re
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

import structlog
from structlog.types import FilteringBoundLogger


@dataclass
class LoggingSecurityConfig:
    """Configuration for secure logging"""
    mask_sensitive_fields: bool = True
    hash_pii: bool = True
    truncate_long_values: bool = True
    max_value_length: int = 200
    enable_audit_trail: bool = True
    log_retention_days: int = 90
    enable_log_rotation: bool = True
    max_log_size_mb: int = 100


class SensitiveDataSanitizer:
    """Sanitizes sensitive data in log entries"""
    
    def __init__(self, config: LoggingSecurityConfig):
        self.config = config
        
        # Sensitive field patterns
        self.sensitive_fields = {
            'password', 'passwd', 'secret', 'key', 'token', 'authorization',
            'auth', 'credential', 'cred', 'api_key', 'private_key', 'session_id',
            'session', 'csrf_token', 'jwt', 'bearer', 'cookie', 'ssn', 'social_security',
            'credit_card', 'card_number', 'cvv', 'pin', 'bank_account', 'routing'
        }
        
        # PII field patterns  
        self.pii_fields = {
            'email', 'phone', 'telephone', 'mobile', 'address', 'street', 'zip', 'postal',
            'first_name', 'last_name', 'full_name', 'name', 'birth_date', 'dob',
            'license', 'passport', 'id_number', 'user_id', 'customer_id'
        }
        
        # Sensitive value patterns (regex)
        self.sensitive_patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),  # Email
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 'CARD'),            # Credit card
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),                                   # SSN
            (r'\b\d{10,15}\b', 'PHONE'),                                         # Phone numbers
            (r'Bearer\s+[A-Za-z0-9\-._~+/]+', 'BEARER_TOKEN'),                  # Bearer tokens
            (r'[A-Za-z0-9+/]{40,}={0,2}', 'BASE64_TOKEN'),                      # Base64 tokens
            (r'sk-[A-Za-z0-9]{48}', 'API_KEY'),                                 # API keys
            (r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID'),  # UUIDs
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), replacement) 
                                 for pattern, replacement in self.sensitive_patterns]
    
    def sanitize_value(self, key: str, value: Any) -> Any:
        """Sanitize a single value based on key and content"""
        if value is None:
            return value
        
        # Convert to string for processing
        str_value = str(value)
        
        # Check if field name indicates sensitive data
        key_lower = key.lower()
        
        if any(sensitive in key_lower for sensitive in self.sensitive_fields):
            return self._mask_sensitive_value(str_value)
        
        if any(pii in key_lower for pii in self.pii_fields):
            if self.config.hash_pii:
                return self._hash_value(str_value)
            else:
                return self._mask_sensitive_value(str_value)
        
        # Check content patterns
        sanitized_value = self._sanitize_by_pattern(str_value)
        
        # Truncate long values
        if self.config.truncate_long_values and len(sanitized_value) > self.config.max_value_length:
            sanitized_value = sanitized_value[:self.config.max_value_length] + "...[TRUNCATED]"
        
        return sanitized_value
    
    def _mask_sensitive_value(self, value: str) -> str:
        """Mask sensitive values"""
        if len(value) <= 4:
            return "*" * len(value)
        elif len(value) <= 8:
            return value[:2] + "*" * (len(value) - 2)
        else:
            return value[:3] + "*" * (len(value) - 6) + value[-3:]
    
    def _hash_value(self, value: str) -> str:
        """Hash PII values for privacy"""
        if not value:
            return value
        
        # Use SHA-256 hash with truncation for readability
        hash_obj = hashlib.sha256(value.encode('utf-8'))
        return f"hash:{hash_obj.hexdigest()[:16]}"
    
    def _sanitize_by_pattern(self, value: str) -> str:
        """Sanitize value based on content patterns"""
        for pattern, replacement in self.compiled_patterns:
            value = pattern.sub(f"[{replacement}]", value)
        
        return value
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary data"""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = self.sanitize_list(value)
            else:
                sanitized[key] = self.sanitize_value(key, value)
        
        return sanitized
    
    def sanitize_list(self, data: List[Any]) -> List[Any]:
        """Recursively sanitize list data"""
        if not isinstance(data, list):
            return data
        
        sanitized = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                sanitized.append(self.sanitize_dict(item))
            elif isinstance(item, list):
                sanitized.append(self.sanitize_list(item))
            else:
                sanitized.append(self.sanitize_value(f"item_{i}", item))
        
        return sanitized


class SecureLogProcessor:
    """Secure log processing with sanitization"""
    
    def __init__(self, config: LoggingSecurityConfig):
        self.config = config
        self.sanitizer = SensitiveDataSanitizer(config)
    
    def __call__(self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process log event with security sanitization"""
        
        # Add security metadata
        event_dict['_log_secure'] = True
        event_dict['_sanitized'] = self.config.mask_sensitive_fields
        event_dict['_timestamp'] = datetime.utcnow().isoformat()
        
        # Sanitize sensitive data
        if self.config.mask_sensitive_fields:
            event_dict = self.sanitizer.sanitize_dict(event_dict)
        
        # Add log level validation
        level = event_dict.get('level', 'info')
        if level not in ['debug', 'info', 'warning', 'error', 'critical']:
            event_dict['level'] = 'info'
        
        return event_dict


class AuditLogger:
    """Dedicated audit logger for security events"""
    
    def __init__(self, config: LoggingSecurityConfig):
        self.config = config
        self.sanitizer = SensitiveDataSanitizer(config)
        
        # Setup dedicated audit logger
        self.audit_logger = structlog.get_logger("audit")
    
    async def log_security_event(self, event_type: str, details: Dict[str, Any], 
                                severity: str = "info", user_id: Optional[str] = None):
        """Log security-related events"""
        
        audit_entry = {
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "details": self.sanitizer.sanitize_dict(details) if self.config.mask_sensitive_fields else details,
            "audit": True
        }
        
        # Log based on severity
        if severity == "critical":
            self.audit_logger.critical("Security event", **audit_entry)
        elif severity == "error":
            self.audit_logger.error("Security event", **audit_entry)
        elif severity == "warning":
            self.audit_logger.warning("Security event", **audit_entry)
        else:
            self.audit_logger.info("Security event", **audit_entry)
    
    async def log_authentication_event(self, event: str, user_id: Optional[str] = None, 
                                     ip_address: Optional[str] = None, 
                                     user_agent: Optional[str] = None,
                                     success: bool = True):
        """Log authentication events"""
        
        details = {
            "auth_event": event,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent[:100] if user_agent else None  # Truncate user agent
        }
        
        severity = "info" if success else "warning"
        await self.log_security_event("authentication", details, severity, user_id)
    
    async def log_authorization_event(self, resource: str, action: str, 
                                    user_id: Optional[str] = None,
                                    allowed: bool = True,
                                    reason: Optional[str] = None):
        """Log authorization events"""
        
        details = {
            "resource": resource,
            "action": action,
            "allowed": allowed,
            "reason": reason
        }
        
        severity = "info" if allowed else "warning"
        await self.log_security_event("authorization", details, severity, user_id)
    
    async def log_data_access_event(self, resource: str, operation: str,
                                  user_id: Optional[str] = None,
                                  record_count: Optional[int] = None):
        """Log data access events"""
        
        details = {
            "resource": resource,
            "operation": operation,
            "record_count": record_count
        }
        
        await self.log_security_event("data_access", details, "info", user_id)


class SecureLoggingFilter(logging.Filter):
    """Logging filter to prevent sensitive data leakage"""
    
    def __init__(self, config: LoggingSecurityConfig):
        super().__init__()
        self.config = config
        self.sanitizer = SensitiveDataSanitizer(config)
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize log records"""
        
        # Sanitize the log message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self.sanitizer._sanitize_by_pattern(record.msg)
        
        # Sanitize any additional arguments
        if hasattr(record, 'args') and record.args:
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, (dict, list)):
                    if isinstance(arg, dict):
                        sanitized_args.append(self.sanitizer.sanitize_dict(arg))
                    else:
                        sanitized_args.append(self.sanitizer.sanitize_list(arg))
                else:
                    sanitized_args.append(self.sanitizer.sanitize_value("arg", arg))
            record.args = tuple(sanitized_args)
        
        return True


def setup_secure_logging(config: LoggingSecurityConfig) -> AuditLogger:
    """Setup secure logging with sanitization"""
    
    # Create secure log processor
    secure_processor = SecureLogProcessor(config)
    
    # Configure structlog with security processor
    structlog.configure(
        processors=[
            secure_processor,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup standard logging filter
    logging_filter = SecureLoggingFilter(config)
    
    # Apply filter to all loggers
    for logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error', 'fastapi']:
        logger = logging.getLogger(logger_name)
        logger.addFilter(logging_filter)
    
    # Create audit logger
    audit_logger = AuditLogger(config)
    
    return audit_logger


# Global instances
_audit_logger: Optional[AuditLogger] = None
_logging_config: Optional[LoggingSecurityConfig] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger, _logging_config
    
    if _audit_logger is None:
        _logging_config = LoggingSecurityConfig()
        _audit_logger = setup_secure_logging(_logging_config)
    
    return _audit_logger


def get_secure_logger(name: str) -> FilteringBoundLogger:
    """Get secure logger instance"""
    return structlog.get_logger(name)


# Decorator for automatic audit logging
def audit_log(event_type: str, severity: str = "info"):
    """Decorator to automatically audit log function calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            
            # Extract user context if available
            user_id = None
            for arg in args:
                if hasattr(arg, 'user_id'):
                    user_id = arg.user_id
                    break
            
            try:
                result = await func(*args, **kwargs)
                
                await audit_logger.log_security_event(
                    event_type=event_type,
                    details={
                        "function": func.__name__,
                        "success": True,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    },
                    severity=severity,
                    user_id=user_id
                )
                
                return result
                
            except Exception as e:
                await audit_logger.log_security_event(
                    event_type=event_type,
                    details={
                        "function": func.__name__,
                        "success": False,
                        "error": str(e),
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    },
                    severity="error",
                    user_id=user_id
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't easily use async audit logging
            # In a real implementation, you might want to queue these for async processing
            return func(*args, **kwargs)
        
        return async_wrapper if hasattr(func, '__await__') else sync_wrapper
    
    return decorator