"""Tenant domain entities and models."""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Boolean, JSON, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class TenantStatus(str, Enum):
    """Tenant status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPROVISIONING = "deprovisioning"
    ARCHIVED = "archived"


class TenantPlan(str, Enum):
    """Tenant subscription plans."""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class Tenant(Base):
    """Tenant entity for multi-tenant isolation."""
    __tablename__ = "tenants"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    status = Column(String(50), nullable=False, default=TenantStatus.ACTIVE.value)
    plan = Column(String(50), nullable=False, default=TenantPlan.STARTER.value)
    
    # Metadata
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Contact information
    contact_email = Column(String(255))
    contact_name = Column(String(255))
    
    # Billing and limits
    max_users = Column(String(20), default="10")
    max_storage_gb = Column(String(20), default="100")
    
    # Security settings
    require_mfa = Column(Boolean, default=False)
    allowed_domains = Column(JSON, default=list)  # Email domains for auto-enrollment
    
    # Custom branding
    logo_url = Column(String(500))
    primary_color = Column(String(7))  # Hex color
    
    def __repr__(self):
        return f"<Tenant(id={self.id}, name={self.name}, status={self.status})>"


class TenantUser(Base):
    """User-tenant relationship with roles."""
    __tablename__ = "tenant_users"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(PGUUID(as_uuid=True), nullable=False)
    user_id = Column(String(255), nullable=False)  # OIDC sub claim
    email = Column(String(255), nullable=False)
    roles = Column(JSON, default=list)  # List of role strings
    
    # Status
    is_active = Column(Boolean, default=True)
    invited_at = Column(DateTime(timezone=True))
    joined_at = Column(DateTime(timezone=True))
    last_login = Column(DateTime(timezone=True))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<TenantUser(tenant_id={self.tenant_id}, user_id={self.user_id})>"


class Evidence(Base):
    """Evidence table with tenant isolation."""
    __tablename__ = "evidence"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(PGUUID(as_uuid=True), nullable=False)
    
    # Evidence metadata
    filename = Column(String(500), nullable=False)
    content_type = Column(String(100))
    size_bytes = Column(String(20))
    sha256_hash = Column(String(64))
    
    # Storage information
    storage_path = Column(String(1000))
    storage_backend = Column(String(50), default="filesystem")
    
    # Processing status
    status = Column(String(50), default="uploaded")
    processed_at = Column(DateTime(timezone=True))
    
    # User tracking
    uploaded_by = Column(String(255))  # User ID
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Evidence(id={self.id}, tenant_id={self.tenant_id}, filename={self.filename})>"


class Finding(Base):
    """Security findings with tenant isolation."""
    __tablename__ = "findings"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(PGUUID(as_uuid=True), nullable=False)
    
    # Finding details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    severity = Column(String(20))  # low, medium, high, critical
    status = Column(String(50), default="open")
    
    # Classification
    category = Column(String(100))
    tags = Column(JSON, default=list)
    
    # Evidence relationship
    evidence_ids = Column(JSON, default=list)  # List of evidence UUIDs
    
    # MITRE ATT&CK mapping
    attack_techniques = Column(JSON, default=list)
    attack_tactics = Column(JSON, default=list)
    
    # User tracking
    created_by = Column(String(255))  # User ID
    assigned_to = Column(String(255))  # User ID
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<Finding(id={self.id}, tenant_id={self.tenant_id}, title={self.title})>"


class EmbeddingVector(Base):
    """Vector embeddings with tenant isolation."""
    __tablename__ = "embedding_vectors"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(PGUUID(as_uuid=True), nullable=False)
    
    # Source reference
    source_type = Column(String(50))  # evidence, finding, text
    source_id = Column(PGUUID(as_uuid=True))
    
    # Embedding data
    content_hash = Column(String(64))
    embedding_model = Column(String(100))
    # Note: vector column will be added in migration with pgvector
    
    # Vector metadata
    vector_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<EmbeddingVector(id={self.id}, tenant_id={self.tenant_id}, source_type={self.source_type})>"


# Pydantic models for API
class TenantBase(BaseModel):
    """Base tenant model."""
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    contact_email: Optional[str] = None
    contact_name: Optional[str] = None
    settings: Dict = Field(default_factory=dict)


class TenantCreate(TenantBase):
    """Create tenant model."""
    plan: TenantPlan = TenantPlan.STARTER
    max_users: int = 10
    max_storage_gb: int = 100


class TenantUpdate(BaseModel):
    """Update tenant model."""
    name: Optional[str] = None
    status: Optional[TenantStatus] = None
    contact_email: Optional[str] = None
    contact_name: Optional[str] = None
    settings: Optional[Dict] = None
    max_users: Optional[int] = None
    max_storage_gb: Optional[int] = None
    require_mfa: Optional[bool] = None


class TenantResponse(TenantBase):
    """Tenant response model."""
    id: UUID
    status: TenantStatus
    plan: TenantPlan
    max_users: int
    max_storage_gb: int
    require_mfa: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TenantUserBase(BaseModel):
    """Base tenant user model."""
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    roles: List[str] = Field(default_factory=list)


class TenantUserCreate(TenantUserBase):
    """Create tenant user model."""
    user_id: str = Field(..., min_length=1)


class TenantUserUpdate(BaseModel):
    """Update tenant user model."""
    roles: Optional[List[str]] = None
    is_active: Optional[bool] = None


class TenantUserResponse(TenantUserBase):
    """Tenant user response model."""
    id: UUID
    tenant_id: UUID
    user_id: str
    is_active: bool
    joined_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}