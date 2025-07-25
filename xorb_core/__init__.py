"""
Xorb Core - Hexagonal Architecture Implementation
Domain-Driven Design with Clean Architecture Principles

Layer Architecture:
- Domain: Pure business logic and entities
- Application: Use cases and abstract ports
- Infrastructure: Concrete implementations and external integrations
- Interfaces: REST/gRPC APIs and external interfaces

Dependencies flow inward: Interfaces -> Application -> Domain
Infrastructure is dependency-injected into Application layer
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Xorb Security Intelligence Team"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__license__",

    # Core modules
    "domain",
    "application",
    "infrastructure",
    "interfaces"
]

# Layer imports (optional - consumers can import layers directly)
from . import domain
from . import application
from . import infrastructure
from . import interfaces