"""
Production RBAC Decorators
Replacement for placeholder security decorators with full RBAC implementation
"""

from ..auth.rbac_dependencies import (
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
    rbac_decorator
)


# Direct permission requirements for common operations
def require_orchestration():
    """Require orchestration permissions"""
    return require_permissions(["job:create", "job:priority"])


def require_agent_management():
    """Require agent management permissions"""
    return require_permissions(["agent:create", "agent:update", "agent:control"])


def require_discovery():
    """Require discovery service permissions"""
    return require_permission("ptaas:scan:create")


def require_embeddings():
    """Require embeddings service permissions"""
    return require_permission("intelligence:read")


def require_telemetry():
    """Require telemetry permissions"""
    return require_permission("telemetry:write")


def require_storage():
    """Require storage/evidence permissions"""
    return require_permission("evidence:write")


def require_vectors():
    """Require vector operations permissions"""
    return require_permission("evidence:read")


def require_jobs():
    """Require job management permissions"""
    return require_permission("job:create")


def require_security_ops():
    """Require security operations permissions"""
    return require_permissions(["ptaas:scan:create", "intelligence:analyze"])


def require_ptaas_access():
    """Require PTaaS access permissions"""
    return require_permission("ptaas:scan:read")


# Expose all RBAC functionality
__all__ = [
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
    
    # Service-specific decorators
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