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
from typing import Optional, Dict, Any
from dataclasses import dataclass

class Role(Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYSTEM_ADMIN = "system_admin"
    
    # Task permissions
    TASK_PRIORITY = "task_priority"
    TASK_MANAGEMENT = "task_management"
    TASK_READ = "task_read"
    TASK_SUBMIT = "task_submit"
    TASK_CANCEL = "task_cancel"
    
    # Orchestration permissions
    ORCHESTRATION = "orchestration"
    
    # PTaaS permissions
    PTAAS_ACCESS = "ptaas_access"
    
    # Enterprise permissions
    ENTERPRISE_ADMIN = "enterprise_admin"
    USER_MANAGEMENT = "user_management"
    TENANT_MANAGEMENT = "tenant_management"
    
    # Telemetry permissions
    TELEMETRY_READ = "telemetry_read"
    TELEMETRY_WRITE = "telemetry_write"
    
    # Evidence/Storage permissions
    EVIDENCE_READ = "evidence_read"
    EVIDENCE_WRITE = "evidence_write"
    EVIDENCE_DELETE = "evidence_delete"
    
    # Job permissions
    JOBS_READ = "jobs_read"
    JOBS_WRITE = "jobs_write"
    JOBS_CANCEL = "jobs_cancel"
    
    # Security operations permissions
    SECURITY_READ = "security_read"
    SECURITY_WRITE = "security_write"
    
    # Agent permissions
    AGENT_READ = "agent_read"
    AGENT_UPDATE = "agent_update"
    AGENT_DELETE = "agent_delete"

@dataclass
class SecurityContext:
    """Security context for request authorization"""
    user_id: str
    tenant_id: Optional[str] = None
    roles: list = None
    permissions: list = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}

def get_security_context() -> SecurityContext:
    """Get current security context"""
    return SecurityContext(user_id="anonymous")

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

def require_orchestrator():
    """Placeholder orchestrator requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_agent_management():
    """Placeholder agent management requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_permissions(permission: Permission):
    """Placeholder permissions requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_discovery():
    """Placeholder discovery requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_embeddings():
    """Placeholder embeddings requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_telemetry():
    """Placeholder telemetry requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_storage():
    """Placeholder storage requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_vectors():
    """Placeholder vectors requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_jobs():
    """Placeholder jobs requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_security_ops():
    """Placeholder security operations requirement decorator"""
    def decorator(func):
        return func
    return decorator

def require_ptaas_access():
    """Placeholder PTaaS access requirement decorator"""
    def decorator(func):
        return func
    return decorator

# Additional type needed by some routers
class UserClaims:
    """User claims for authentication"""
    def __init__(self, user_id: str = "anonymous", tenant_id: str = None):
        self.user_id = user_id
        self.tenant_id = tenant_id

__all__ = [
    "APISecurityMiddleware",
    "SecurityConfig", 
    "BaseValidator",
    "SecurityValidator",
    "SecurityPolicy",
    "NetworkSecurityValidator",
    "SecurityContext",
    "get_security_context",
    "Role",
    "Permission",
    "UserClaims",
    "require_admin",
    "require_permission",
    "require_orchestrator",
    "require_agent_management",
    "require_permissions",
    "require_discovery",
    "require_embeddings",
    "require_telemetry",
    "require_storage",
    "require_vectors",
    "require_jobs",
    "require_security_ops",
    "require_ptaas_access"
]