"""
XORB API Security Module
"""
# Import what's actually available
try:
    from .api_security import APISecurityMiddleware, SecurityConfig
except ImportError:
    APISecurityMiddleware = None
    SecurityConfig = None

try:
    from .input_validation import BaseValidator, SecurityValidator
except ImportError:
    BaseValidator = None
    SecurityValidator = None

try:
    from .ptaas_security import SecurityPolicy, NetworkSecurityValidator
except ImportError:
    SecurityPolicy = None
    NetworkSecurityValidator = None

# Fallback auth implementations
from enum import Enum

class Role(Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

def require_admin():
    """Placeholder admin requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_permission(permission: Permission):
    """Placeholder permission requirement decorator"""
    def decorator(func):
        return func
    return decorator

__all__ = [
    "APISecurityMiddleware",
    "SecurityConfig",
    "BaseValidator",
    "SecurityValidator",
    "SecurityPolicy",
    "NetworkSecurityValidator",
    "Role",
    "Permission",
    "require_admin",
    "require_permission"
]
