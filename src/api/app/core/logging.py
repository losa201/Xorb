"""
Production-grade structured logging configuration with correlation tracking
"""

import sys
import json
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar
from pathlib import Path

import structlog
from structlog.stdlib import LoggerFactory
try:
    from structlog.processors import JSONRenderer, TimeStamper, add_log_level, add_logger_name
except ImportError:
    # Fallback for older structlog versions
    from structlog.processors import JSONRenderer, TimeStamper, add_log_level
    add_logger_name = None


# Context variables for correlation tracking
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
tenant_id_var: ContextVar[str] = ContextVar('tenant_id', default='')


class CorrelationProcessor:
    """Add correlation context to log entries"""
    
    def __call__(self, logger, method_name, event_dict):
        # Add correlation tracking
        if correlation_id := correlation_id_var.get():
            event_dict['correlation_id'] = correlation_id
        if request_id := request_id_var.get():
            event_dict['request_id'] = request_id
        if user_id := user_id_var.get():
            event_dict['user_id'] = user_id
        if tenant_id := tenant_id_var.get():
            event_dict['tenant_id'] = tenant_id
            
        return event_dict


class SecurityProcessor:
    """Process security-sensitive log data"""
    
    SENSITIVE_FIELDS = {
        'password', 'token', 'secret', 'key', 'auth', 'credential',
        'ssn', 'credit_card', 'email', 'phone', 'address'
    }
    
    def __call__(self, logger, method_name, event_dict):
        # Mask sensitive data
        for key, value in event_dict.items():
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_FIELDS):
                if isinstance(value, str) and len(value) > 4:
                    event_dict[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    event_dict[key] = '***MASKED***'
        
        return event_dict


class PerformanceProcessor:
    """Add performance metrics to logs"""
    
    def __call__(self, logger, method_name, event_dict):
        # Add timestamp and performance context
        event_dict['timestamp'] = datetime.utcnow().isoformat()
        event_dict['logger_name'] = logger.name
        event_dict['level'] = method_name.upper()
        
        return event_dict


def setup_logging(
    log_level: str = "INFO",
    environment: str = "development",
    log_file: Optional[Path] = None,
    enable_json: bool = True,
    enable_colors: bool = True
) -> None:
    """
    Configure structured logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        environment: Runtime environment (development, staging, production)
        log_file: Optional log file path
        enable_json: Use JSON formatting for structured logs
        enable_colors: Enable colored output for console
    """
    
    # Configure processors based on environment
    processors = [
        structlog.stdlib.filter_by_level,
        CorrelationProcessor(),
        SecurityProcessor(),
        PerformanceProcessor(),
        add_log_level,
        TimeStamper(fmt="ISO"),
    ]
    
    # Add logger name processor if available
    if add_logger_name is not None:
        processors.insert(-1, add_logger_name)
    
    if environment == "production":
        # Production: JSON output, no colors
        processors.append(JSONRenderer())
        console_renderer = JSONRenderer()
    else:
        # Development: Human-readable with colors
        if enable_colors:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
            console_renderer = structlog.dev.ConsoleRenderer(colors=True)
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))
            console_renderer = structlog.dev.ConsoleRenderer(colors=False)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure file logging if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if enable_json:
            file_formatter = logging.Formatter('%(message)s')
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Set third-party library log levels
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


def set_correlation_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> None:
    """Set correlation context for the current request"""
    if correlation_id:
        correlation_id_var.set(correlation_id)
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if tenant_id:
        tenant_id_var.set(tenant_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return str(uuid.uuid4())


class LoggingMiddleware:
    """FastAPI middleware for request logging"""
    
    def __init__(self, app):
        self.app = app
        self.logger = get_logger("middleware.logging")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Generate correlation ID for request
        correlation_id = generate_correlation_id()
        request_id = str(uuid.uuid4())
        
        # Set correlation context
        set_correlation_context(
            correlation_id=correlation_id,
            request_id=request_id
        )
        
        # Log request start
        start_time = datetime.utcnow()
        method = scope.get("method", "")
        path = scope.get("path", "")
        
        self.logger.info(
            "Request started",
            method=method,
            path=path,
            correlation_id=correlation_id,
            request_id=request_id
        )
        
        # Process request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Log response
                status_code = message.get("status", 0)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                self.logger.info(
                    "Request completed",
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_seconds=duration,
                    correlation_id=correlation_id,
                    request_id=request_id
                )
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(
                "Request failed",
                method=method,
                path=path,
                error=str(e),
                duration_seconds=duration,
                correlation_id=correlation_id,
                request_id=request_id,
                exc_info=True
            )
            raise


# Pre-configured loggers for common use cases
security_logger = get_logger("security")
performance_logger = get_logger("performance")
audit_logger = get_logger("audit")
api_logger = get_logger("api")
database_logger = get_logger("database")