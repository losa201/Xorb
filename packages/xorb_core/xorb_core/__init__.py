"""
Xorb Core Package
Main package for XORB security intelligence platform core functionality
"""

__version__ = "2.0.0"

# Core imports
from .logging import configure_logging, get_logger

__all__ = [
    "configure_logging",
    "get_logger",
]