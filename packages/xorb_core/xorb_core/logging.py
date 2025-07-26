"""
Xorb Core Logging Module
Provides structured logging capabilities for the XORB platform
"""
import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.types import Processor


def configure_logging(
    level: str = "INFO", 
    service_name: str = "xorb", 
    enable_json: bool = True
) -> None:
    """Configure structured logging for XORB services."""
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.add_log_level,
        structlog.processors.CallsiteParameterAdder(
            parameters=[structlog.processors.CallsiteParameter.FUNC_NAME]
        ),
    ]
    
    # Add service name to all log entries
    def add_service_name(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_dict["service"] = service_name
        return event_dict
    
    processors.insert(0, add_service_name)
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


# Configure default logging on module import
configure_logging()