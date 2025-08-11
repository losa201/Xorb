"""
XORB API Security Module - Production RBAC Implementation
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

# Production RBAC Implementation
from ..auth.models import Role, Permission, UserClaims
from ..services.rbac_service import RBACContext as SecurityContext
from ..auth.rbac_dependencies import get_rbac_context as get_security_context

# Import all production RBAC decorators
from .rbac_decorators import (
    require_permission,
    require_permissions, 
    require_role,
    require_any_role,
    require_admin,
    require_tenant_admin,
    require_security_manager,
    require_security_analyst,
    require_user_management,
    require_ptaas_scan,
    require_ptaas_read,
    require_intelligence_read,
    require_system_admin,
    require_audit_read,
    require_orchestration,
    require_agent_management,
    require_discovery,
    require_embeddings,
    require_telemetry,
    require_storage,
    require_vectors,
    require_jobs,
    require_security_ops,
    require_ptaas_access,
    rbac_decorator
)

__all__ = [
    # Security infrastructure
    "APISecurityMiddleware",
    "SecurityConfig", 
    "BaseValidator",
    "SecurityValidator",
    "SecurityPolicy",
    "NetworkSecurityValidator",
    
    # RBAC core types
    "SecurityContext",
    "get_security_context",
    "Role",
    "Permission", 
    "UserClaims",
    
    # Permission-based decorators
    "require_permission",
    "require_permissions", 
    
    # Role-based decorators
    "require_role",
    "require_any_role",
    "require_admin",
    "require_tenant_admin",
    "require_security_manager",
    "require_security_analyst",
    
    # Common permission shortcuts
    "require_user_management",
    "require_ptaas_scan",
    "require_ptaas_read",
    "require_intelligence_read",
    "require_system_admin",
    "require_audit_read",
    
    # Service-specific decorators (backward compatible)
    "require_orchestration",
    "require_agent_management",
    "require_discovery",
    "require_embeddings",
    "require_telemetry",
    "require_storage",
    "require_vectors",
    "require_jobs",
    "require_security_ops",
    "require_ptaas_access",
    
    # Advanced decorator
    "rbac_decorator"
]