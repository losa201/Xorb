"""
SQLAlchemy Database Models for XORB Enterprise Platform
Production-ready models with comprehensive relationships and constraints
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, Text, JSON, 
    ForeignKey, UUID, Enum, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class TenantStatus(enum.Enum):
    """Tenant status enumeration"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    PENDING = "pending"


class TenantPlan(enum.Enum):
    """Tenant plan enumeration"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ScanStatus(enum.Enum):
    """Scan status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SeverityLevel(enum.Enum):
    """Security finding severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class UserModel(Base):
    """User model with comprehensive authentication support"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Profile information
    first_name = Column(String(100))
    last_name = Column(String(100))
    full_name = Column(String(200))
    
    # Role and permissions
    roles = Column(ARRAY(String), default=["user"])
    permissions = Column(JSONB, default={})
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    
    # Security settings
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(32))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    last_login_at = Column(DateTime)
    password_changed_at = Column(DateTime, default=func.now())
    
    # Metadata
    user_metadata = Column(JSONB, default={})
    
    # Relationships
    organizations = relationship("UserOrganizationModel", back_populates="user", cascade="all, delete-orphan")
    auth_tokens = relationship("AuthTokenModel", back_populates="user", cascade="all, delete-orphan")
    scan_sessions = relationship("ScanSessionModel", back_populates="user")
    discovery_workflows = relationship("DiscoveryWorkflowModel", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index('idx_users_email_active', email, is_active),
        Index('idx_users_username_active', username, is_active),
        Index('idx_users_created_at', created_at),
        CheckConstraint('char_length(username) >= 3', name='check_username_length'),
        CheckConstraint(r"email ~ '^[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}$'", name='check_email_format'),
    )


class OrganizationModel(Base):
    """Organization model with multi-tenancy support"""
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), unique=True, nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    
    # Plan and billing
    plan_type = Column(String(50), default="free", nullable=False)
    billing_email = Column(String(255))
    
    # Organization settings
    settings = Column(JSONB, default={})
    features = Column(ARRAY(String), default=[])
    
    # Limits and quotas
    max_users = Column(Integer, default=10)
    max_scans_per_month = Column(Integer, default=100)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Ownership
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    owner = relationship("UserModel", foreign_keys=[owner_id])
    members = relationship("UserOrganizationModel", back_populates="organization", cascade="all, delete-orphan")
    tenants = relationship("TenantModel", back_populates="organization")
    scan_sessions = relationship("ScanSessionModel", back_populates="organization")
    
    # Indexes
    __table_args__ = (
        Index('idx_organizations_owner', owner_id),
        Index('idx_organizations_active', is_active),
        Index('idx_organizations_plan', plan_type),
        CheckConstraint('char_length(name) >= 2', name='check_organization_name_length'),
    )


class UserOrganizationModel(Base):
    """User-Organization relationship with roles"""
    __tablename__ = "user_organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Role in organization
    role = Column(String(50), default="member", nullable=False)
    permissions = Column(ARRAY(String), default=[])
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    joined_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("UserModel", back_populates="organizations")
    organization = relationship("OrganizationModel", back_populates="members")
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'organization_id', name='uq_user_organization'),
        Index('idx_user_org_user', user_id),
        Index('idx_user_org_organization', organization_id),
        Index('idx_user_org_role', role),
    )


class TenantModel(Base):
    """Tenant model for multi-tenancy support"""
    __tablename__ = "tenants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    
    # Tenant configuration
    plan = Column(Enum(TenantPlan), default=TenantPlan.FREE, nullable=False)
    status = Column(Enum(TenantStatus), default=TenantStatus.ACTIVE, nullable=False)
    settings = Column(JSONB, default={})
    
    # Organization relationship
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    organization = relationship("OrganizationModel", back_populates="tenants")
    
    # Indexes
    __table_args__ = (
        Index('idx_tenants_organization', organization_id),
        Index('idx_tenants_status', status),
        Index('idx_tenants_plan', plan),
    )


class AuthTokenModel(Base):
    """Authentication token model"""
    __tablename__ = "auth_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Token information
    token_type = Column(String(50), nullable=False)  # refresh, api_key, etc.
    token_value = Column(String(500), nullable=False, index=True)
    session_id = Column(String(100), index=True)
    
    # Token metadata
    name = Column(String(100))  # For API keys
    scopes = Column(ARRAY(String), default=[])
    
    # Expiration and revocation
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime)
    revoked_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Usage tracking
    last_used_at = Column(DateTime)
    usage_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("UserModel", back_populates="auth_tokens", foreign_keys=[user_id])
    revoker = relationship("UserModel", foreign_keys=[revoked_by])
    
    # Indexes
    __table_args__ = (
        Index('idx_auth_tokens_user', user_id),
        Index('idx_auth_tokens_session', session_id),
        Index('idx_auth_tokens_type', token_type),
        Index('idx_auth_tokens_expires', expires_at),
        Index('idx_auth_tokens_active', is_revoked, expires_at),
    )


class ScanSessionModel(Base):
    """Scan session model for PTaaS"""
    __tablename__ = "scan_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Ownership
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Scan configuration
    scan_profile = Column(String(100), nullable=False)
    targets = Column(JSONB, nullable=False)  # List of scan targets
    
    # Execution
    status = Column(Enum(ScanStatus), default=ScanStatus.QUEUED, nullable=False)
    progress = Column(Integer, default=0)
    
    # Results
    results = Column(JSONB, default={})
    findings_count = Column(Integer, default=0)
    critical_findings = Column(Integer, default=0)
    high_findings = Column(Integer, default=0)
    medium_findings = Column(Integer, default=0)
    low_findings = Column(Integer, default=0)
    
    # Timing
    scheduled_for = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Scan metadata
    scan_metadata = Column(JSONB, default={})
    tags = Column(ARRAY(String), default=[])
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("UserModel", back_populates="scan_sessions")
    organization = relationship("OrganizationModel", back_populates="scan_sessions")
    findings = relationship("SecurityFindingModel", back_populates="scan_session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_scan_sessions_user', user_id),
        Index('idx_scan_sessions_organization', organization_id),
        Index('idx_scan_sessions_status', status),
        Index('idx_scan_sessions_created', created_at),
        Index('idx_scan_sessions_profile', scan_profile),
    )


class SecurityFindingModel(Base):
    """Security finding model"""
    __tablename__ = "security_findings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_session_id = Column(UUID(as_uuid=True), ForeignKey("scan_sessions.id"), nullable=False)
    
    # Finding identification
    finding_id = Column(String(200), index=True)  # External tool finding ID
    rule_id = Column(String(200), index=True)     # Rule or template ID
    
    # Finding details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    severity = Column(Enum(SeverityLevel), nullable=False)
    confidence = Column(String(50))  # high, medium, low
    
    # Location
    target_host = Column(String(255), nullable=False)
    target_port = Column(Integer)
    target_path = Column(String(1000))
    target_url = Column(String(2000))
    
    # Classification
    category = Column(String(100))
    cwe_id = Column(String(20))       # Common Weakness Enumeration
    cvss_score = Column(String(10))   # CVSS score
    
    # Evidence
    evidence = Column(JSONB, default={})
    proof_of_concept = Column(Text)
    
    # Remediation
    remediation = Column(Text)
    references = Column(JSONB, default=[])
    
    # Status
    status = Column(String(50), default="open")  # open, fixed, false_positive, accepted
    
    # Finding metadata
    finding_metadata = Column(JSONB, default={})
    tags = Column(ARRAY(String), default=[])
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    scan_session = relationship("ScanSessionModel", back_populates="findings")
    
    # Indexes
    __table_args__ = (
        Index('idx_security_findings_session', scan_session_id),
        Index('idx_security_findings_severity', severity),
        Index('idx_security_findings_target', target_host),
        Index('idx_security_findings_category', category),
        Index('idx_security_findings_status', status),
        Index('idx_security_findings_created', created_at),
    )


class DiscoveryWorkflowModel(Base):
    """Discovery workflow model"""
    __tablename__ = "discovery_workflows"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    
    # Discovery details
    domain = Column(String(255), nullable=False, index=True)
    discovery_type = Column(String(100), default="comprehensive")
    
    # Execution
    status = Column(String(50), default="queued", nullable=False)
    progress = Column(Integer, default=0)
    
    # Results
    results = Column(JSONB, default={})
    
    # Analysis metadata
    analysis_metadata = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("UserModel", back_populates="discovery_workflows")
    
    # Indexes
    __table_args__ = (
        Index('idx_discovery_workflows_user', user_id),
        Index('idx_discovery_workflows_domain', domain),
        Index('idx_discovery_workflows_status', status),
        Index('idx_discovery_workflows_created', created_at),
    )


class EmbeddingRequestModel(Base):
    """Embedding request model"""
    __tablename__ = "embedding_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    
    # Request details
    model = Column(String(100), nullable=False)
    input_type = Column(String(50), nullable=False)
    text_count = Column(Integer, nullable=False)
    token_count = Column(Integer, default=0)
    
    # Processing
    status = Column(String(50), default="pending", nullable=False)
    processing_time_ms = Column(Integer)
    
    # Results
    embeddings_stored = Column(Boolean, default=False)
    
    # Vector metadata
    vector_metadata = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("UserModel")
    results = relationship("EmbeddingResultModel", back_populates="request", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_embedding_requests_user', user_id),
        Index('idx_embedding_requests_model', model),
        Index('idx_embedding_requests_status', status),
        Index('idx_embedding_requests_created', created_at),
    )


class EmbeddingResultModel(Base):
    """Embedding result model"""
    __tablename__ = "embedding_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("embedding_requests.id"), nullable=False)
    
    # Embedding data
    embeddings = Column(JSONB, nullable=False)  # Store embeddings as JSON
    dimensions = Column(Integer, nullable=False)
    
    # Processing info
    processing_time_ms = Column(Integer, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    request = relationship("EmbeddingRequestModel", back_populates="results")
    
    # Indexes
    __table_args__ = (
        Index('idx_embedding_results_request', request_id),
        Index('idx_embedding_results_created', created_at),
    )


class AuditLogModel(Base):
    """Audit log model for security events"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event details
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)  # auth, scan, admin, etc.
    event_action = Column(String(100), nullable=False)
    
    # Actor information
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    username = Column(String(50))
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    
    # Context
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    
    # Request details
    client_ip = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    session_id = Column(String(100))
    
    # Result
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    
    # Audit metadata
    audit_metadata = Column(JSONB, default={})
    
    # Timestamps
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("UserModel")
    organization = relationship("OrganizationModel")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_logs_timestamp', timestamp),
        Index('idx_audit_logs_user', user_id),
        Index('idx_audit_logs_event_type', event_type),
        Index('idx_audit_logs_category', event_category),
        Index('idx_audit_logs_client_ip', client_ip),
        Index('idx_audit_logs_session', session_id),
    )


class SystemConfigModel(Base):
    """System configuration model"""
    __tablename__ = "system_config"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration
    key = Column(String(200), unique=True, nullable=False, index=True)
    value = Column(JSONB, nullable=False)
    description = Column(Text)
    
    # Security
    is_secret = Column(Boolean, default=False, nullable=False)
    
    # Metadata
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    creator = relationship("UserModel", foreign_keys=[created_by])
    updater = relationship("UserModel", foreign_keys=[updated_by])
    
    # Indexes
    __table_args__ = (
        Index('idx_system_config_key', key),
        Index('idx_system_config_secret', is_secret),
    )