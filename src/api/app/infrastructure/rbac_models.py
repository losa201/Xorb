"""
RBAC Database Models
Production-ready Role-Based Access Control models for SQLAlchemy
"""

import uuid
from datetime import datetime
from typing import List, Set, Optional
from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, Text, 
    ForeignKey, UUID, Index, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from .database_models import Base


class RBACRole(Base):
    """Role model for RBAC system"""
    __tablename__ = "rbac_roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Role hierarchy
    is_system_role = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    level = Column(Integer, default=0, nullable=False)  # Higher level = more permissions
    parent_role_id = Column(UUID(as_uuid=True), ForeignKey('rbac_roles.id'), nullable=True)
    
    # Metadata and timestamps
    role_metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Relationships
    parent_role = relationship("RBACRole", remote_side=[id], backref="child_roles")
    permissions = relationship("RBACPermission", secondary="rbac_role_permissions", back_populates="roles")
    user_roles = relationship("RBACUserRole", back_populates="role", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<RBACRole(name='{self.name}', display_name='{self.display_name}')>"
    
    def get_inherited_permissions(self) -> Set[str]:
        """Get all permissions including inherited from parent roles"""
        permissions = set()
        
        # Add direct permissions
        for permission in self.permissions:
            if permission.is_active:
                permissions.add(permission.name)
        
        # Add inherited permissions from parent
        if self.parent_role:
            permissions.update(self.parent_role.get_inherited_permissions())
        
        return permissions


class RBACPermission(Base):
    """Permission model for RBAC system"""
    __tablename__ = "rbac_permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Permission categorization
    resource = Column(String(100), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    is_system_permission = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Metadata and timestamps
    role_metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Relationships
    roles = relationship("RBACRole", secondary="rbac_role_permissions", back_populates="permissions")
    user_permissions = relationship("RBACUserPermission", back_populates="permission", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_rbac_permissions_resource_action', 'resource', 'action'),
        Index('idx_rbac_permissions_name_active', 'name', 'is_active'),
    )
    
    def __repr__(self):
        return f"<RBACPermission(name='{self.name}', resource='{self.resource}', action='{self.action}')>"


class RBACRolePermission(Base):
    """Association table between roles and permissions"""
    __tablename__ = "rbac_role_permissions"
    
    role_id = Column(UUID(as_uuid=True), ForeignKey('rbac_roles.id'), primary_key=True)
    permission_id = Column(UUID(as_uuid=True), ForeignKey('rbac_permissions.id'), primary_key=True)
    
    # Metadata
    granted_at = Column(DateTime, default=func.now(), nullable=False)
    granted_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    role_metadata = Column(JSONB, default={})
    
    # Relationships
    role = relationship("RBACRole")
    permission = relationship("RBACPermission")
    
    def __repr__(self):
        return f"<RBACRolePermission(role_id='{self.role_id}', permission_id='{self.permission_id}')>"


class RBACUserRole(Base):
    """User role assignments (can be tenant-specific)"""
    __tablename__ = "rbac_user_roles"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey('rbac_roles.id'), primary_key=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), primary_key=True, nullable=True)
    
    # Assignment metadata
    granted_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    granted_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    role_metadata = Column(JSONB, default={})
    
    # Relationships
    user = relationship("UserModel", foreign_keys=[user_id])
    role = relationship("RBACRole", back_populates="user_roles")
    tenant = relationship("OrganizationModel", foreign_keys=[tenant_id])
    
    __table_args__ = (
        Index('idx_rbac_user_roles_user_active', 'user_id', 'is_active'),
        Index('idx_rbac_user_roles_role_active', 'role_id', 'is_active'),
        Index('idx_rbac_user_roles_tenant', 'tenant_id'),
        Index('idx_rbac_user_roles_expires', 'expires_at'),
    )
    
    def __repr__(self):
        return f"<RBACUserRole(user_id='{self.user_id}', role_id='{self.role_id}', tenant_id='{self.tenant_id}')>"
    
    def is_valid(self) -> bool:
        """Check if role assignment is currently valid"""
        if not self.is_active:
            return False
        
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        
        return True


class RBACUserPermission(Base):
    """Direct user permission assignments (overrides role permissions)"""
    __tablename__ = "rbac_user_permissions"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True)
    permission_id = Column(UUID(as_uuid=True), ForeignKey('rbac_permissions.id'), primary_key=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), primary_key=True, nullable=True)
    
    # Assignment metadata
    granted_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    granted_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    role_metadata = Column(JSONB, default={})
    
    # Relationships
    user = relationship("UserModel", foreign_keys=[user_id])
    permission = relationship("RBACPermission", back_populates="user_permissions")
    tenant = relationship("OrganizationModel", foreign_keys=[tenant_id])
    
    __table_args__ = (
        Index('idx_rbac_user_permissions_user_active', 'user_id', 'is_active'),
        Index('idx_rbac_user_permissions_permission_active', 'permission_id', 'is_active'),
        Index('idx_rbac_user_permissions_tenant', 'tenant_id'),
        Index('idx_rbac_user_permissions_expires', 'expires_at'),
    )
    
    def __repr__(self):
        return f"<RBACUserPermission(user_id='{self.user_id}', permission_id='{self.permission_id}', tenant_id='{self.tenant_id}')>"
    
    def is_valid(self) -> bool:
        """Check if permission assignment is currently valid"""
        if not self.is_active:
            return False
        
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        
        return True


# Add foreign key relationships to existing models
# This would be added to the UserModel class in database_models.py
"""
# Add these relationships to UserModel:
rbac_roles = relationship("RBACUserRole", foreign_keys=[RBACUserRole.user_id], back_populates="user")
rbac_permissions = relationship("RBACUserPermission", foreign_keys=[RBACUserPermission.user_id], back_populates="user")

# Add these relationships to OrganizationModel:
rbac_user_roles = relationship("RBACUserRole", foreign_keys=[RBACUserRole.tenant_id], back_populates="tenant")
rbac_user_permissions = relationship("RBACUserPermission", foreign_keys=[RBACUserPermission.tenant_id], back_populates="tenant")
"""