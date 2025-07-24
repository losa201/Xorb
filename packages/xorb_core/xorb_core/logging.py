"""
Xorb 2.0 - Centralized Structured Logging Configuration
Provides consistent JSON logging across all Xorb services.
"""
import structlog
import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO", service_name: Optional[str] = None) -> None:
    """
    Configure structured logging for Xorb services.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Name of the service for structured logging context
    """
    # Configure standard library logging
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        format="%(message)s"
    )
    
    # Prepare structlog processors
    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    # Add service name to log context if provided
    if service_name:
        def add_service_name(logger, method_name, event_dict):
            event_dict["service"] = service_name
            return event_dict
        processors.insert(0, add_service_name)
    
    # Add JSON renderer
    processors.append(structlog.processors.JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)