"""
Authentication models for the API
"""

from typing import Optional, List
from pydantic import BaseModel
from enum import Enum

class Role(Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    SUPER_ADMIN = "super_admin"

class Permission(Enum):
    """User permissions"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYSTEM_ADMIN = "system_admin"
    SUPER_ADMIN = "super_admin"
    
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

class UserClaims(BaseModel):
    """User claims from JWT token"""
    user_id: str
    tenant_id: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions