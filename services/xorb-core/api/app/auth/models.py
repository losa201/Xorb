"""Authentication and authorization models."""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field


class Role(str, Enum):
    """User roles within a tenant."""
    SUPER_ADMIN = "super_admin"  # Cross-tenant access
    TENANT_ADMIN = "tenant_admin"  # Full tenant access
    SECURITY_ANALYST = "security_analyst"  # Read/write security data
    AUDITOR = "auditor"  # Read-only access
    VIEWER = "viewer"  # Limited read access


class Permission(str, Enum):
    """Granular permissions."""
    # Evidence & Uploads
    EVIDENCE_READ = "evidence:read"
    EVIDENCE_WRITE = "evidence:write"
    EVIDENCE_DELETE = "evidence:delete"
    
    # Findings
    FINDINGS_READ = "findings:read"
    FINDINGS_WRITE = "findings:write"
    
    # Jobs & Orchestration
    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    JOBS_CANCEL = "jobs:cancel"
    
    # Tenant Management
    TENANT_READ = "tenant:read"
    TENANT_WRITE = "tenant:write"
    
    # System Administration
    SYSTEM_ADMIN = "system:admin"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.SUPER_ADMIN: set(Permission),  # All permissions
    Role.TENANT_ADMIN: {
        Permission.EVIDENCE_READ, Permission.EVIDENCE_WRITE, Permission.EVIDENCE_DELETE,
        Permission.FINDINGS_READ, Permission.FINDINGS_WRITE,
        Permission.JOBS_READ, Permission.JOBS_WRITE, Permission.JOBS_CANCEL,
        Permission.TENANT_READ, Permission.TENANT_WRITE,
    },
    Role.SECURITY_ANALYST: {
        Permission.EVIDENCE_READ, Permission.EVIDENCE_WRITE,
        Permission.FINDINGS_READ, Permission.FINDINGS_WRITE,
        Permission.JOBS_READ, Permission.JOBS_WRITE,
    },
    Role.AUDITOR: {
        Permission.EVIDENCE_READ,
        Permission.FINDINGS_READ,
        Permission.JOBS_READ,
        Permission.TENANT_READ,
    },
    Role.VIEWER: {
        Permission.EVIDENCE_READ,
        Permission.FINDINGS_READ,
        Permission.JOBS_READ,
    },
}


class UserClaims(BaseModel):
    """User claims for authenticated requests."""
    user_id: UUID
    username: str
    email: Optional[str] = None
    name: Optional[str] = None
    tenant_id: UUID
    roles: List[Role] = Field(default_factory=list)
    permissions: List[Permission] = Field(default_factory=list)  # Changed from Set to List for serialization
    is_admin: bool = False
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    session_id: Optional[str] = None
    
    def model_post_init(self, __context) -> None:
        """Derive permissions from roles."""
        if not self.permissions:
            # Derive permissions from all assigned roles
            all_permissions = set()
            for role in self.roles:
                # Add all permissions for this role
                role_permissions = ROLE_PERMISSIONS.get(role, set())
                all_permissions.update(role_permissions)
            
            self.permissions = list(all_permissions)
            
        # Set admin flag
        self.is_admin = Role.SUPER_ADMIN in self.roles or Role.TENANT_ADMIN in self.roles
    
    def has_role(self, role: Role) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def is_super_admin(self) -> bool:
        """Check if user is super admin (cross-tenant access)."""
        return Role.SUPER_ADMIN in self.roles


class TokenData(BaseModel):
    """Token validation data."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class OIDCConfig(BaseModel):
    """OIDC provider configuration."""
    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: List[str] = Field(default_factory=lambda: ["openid", "profile", "email"])
    
    # Claims mapping
    tenant_claim: str = "tenant_id"
    roles_claim: str = "roles"
    name_claim: str = "name"
    email_claim: str = "email"