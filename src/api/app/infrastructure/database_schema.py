"""
Production Database Schema for XORB Enterprise Platform
Comprehensive schema with indexes, constraints, and performance optimizations
"""

from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, Text, JSON, ForeignKey, 
    UUID, Enum, Index, CheckConstraint, UniqueConstraint, BigInteger,
    DECIMAL, TIMESTAMP
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, INET
from sqlalchemy.sql import func
import enum
import uuid

Base = declarative_base()


# Enums
class TenantStatus(enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    PENDING = "pending"


class TenantPlan(enum.Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ScanStatus(enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SeverityLevel(enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComplianceFramework(enum.Enum):
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    GDPR = "gdpr"
    NIST = "nist"


class TokenType(enum.Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    SESSION = "session"


# Core Models
class TenantModel(Base):
    """Multi-tenant support model"""
    __tablename__ = "tenants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    
    # Subscription and billing
    plan = Column(Enum(TenantPlan), default=TenantPlan.BASIC, nullable=False)
    status = Column(Enum(TenantStatus), default=TenantStatus.ACTIVE, nullable=False)
    
    # Configuration and settings
    settings = Column(JSONB, default={})
    rate_limits = Column(JSONB, default={})
    feature_flags = Column(JSONB, default={})
    
    # Billing information
    billing_email = Column(String(255))
    subscription_id = Column(String(100))
    subscription_status = Column(String(50))
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    trial_ends_at = Column(TIMESTAMP)
    
    # Relationships
    users = relationship("UserModel", back_populates="tenant", cascade="all, delete-orphan")
    scan_sessions = relationship("ScanSessionModel", back_populates="tenant")
    
    # Indexes
    __table_args__ = (
        Index("idx_tenant_slug", "slug"),
        Index("idx_tenant_status", "status"),
        Index("idx_tenant_plan", "plan"),
    )


class UserModel(Base):
    """Enhanced user model with comprehensive security features"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    
    # Basic information
    username = Column(String(50), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Profile
    first_name = Column(String(100))
    last_name = Column(String(100))
    full_name = Column(String(200))
    phone_number = Column(String(20))
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='en')
    
    # Security and permissions
    roles = Column(ARRAY(String), default=["user"])
    permissions = Column(JSONB, default={})
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_service_account = Column(Boolean, default=False, nullable=False)
    
    # Security features
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(32))
    backup_codes = Column(ARRAY(String))
    
    # Login security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(TIMESTAMP)
    last_login_at = Column(TIMESTAMP)
    last_login_ip = Column(INET)
    
    # Password security
    password_changed_at = Column(TIMESTAMP, default=func.now())
    force_password_change = Column(Boolean, default=False)
    
    # Audit fields
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    deleted_at = Column(TIMESTAMP)
    
    # Metadata
    user_metadata = Column(JSONB, default={})
    preferences = Column(JSONB, default={})
    
    # Relationships
    tenant = relationship("TenantModel", back_populates="users")
    organizations = relationship("UserOrganizationModel", back_populates="user", cascade="all, delete-orphan")
    auth_tokens = relationship("AuthTokenModel", back_populates="user", cascade="all, delete-orphan")
    scan_sessions = relationship("ScanSessionModel", back_populates="user")
    discovery_workflows = relationship("DiscoveryWorkflowModel", back_populates="user")
    audit_logs = relationship("AuditLogModel", foreign_keys="AuditLogModel.user_id")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("tenant_id", "username", name="uq_tenant_username"),
        UniqueConstraint("tenant_id", "email", name="uq_tenant_email"),
        Index("idx_user_tenant_id", "tenant_id"),
        Index("idx_user_email", "email"),
        Index("idx_user_username", "username"),
        Index("idx_user_active", "is_active"),
        Index("idx_user_last_login", "last_login_at"),
        CheckConstraint("failed_login_attempts >= 0", name="ck_user_failed_attempts"),
    )


class OrganizationModel(Base):
    """Organization model for enterprise features"""
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    
    # Basic information
    name = Column(String(255), nullable=False)
    slug = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    website = Column(String(255))
    
    # Organization settings
    plan_type = Column(String(50), default="basic")
    settings = Column(JSONB, default={})
    security_settings = Column(JSONB, default={})
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # Relationships
    tenant = relationship("TenantModel")
    users = relationship("UserOrganizationModel", back_populates="organization")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_tenant_org_slug"),
        Index("idx_org_tenant_id", "tenant_id"),
        Index("idx_org_active", "is_active"),
    )


class UserOrganizationModel(Base):
    """Many-to-many relationship between users and organizations"""
    __tablename__ = "user_organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    
    # Role within organization
    role = Column(String(50), default="member", nullable=False)
    permissions = Column(JSONB, default={})
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    joined_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("UserModel", back_populates="organizations")
    organization = relationship("OrganizationModel", back_populates="users")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "organization_id", name="uq_user_organization"),
        Index("idx_user_org_user", "user_id"),
        Index("idx_user_org_org", "organization_id"),
    )


class ScanSessionModel(Base):
    """PTaaS scan session model"""
    __tablename__ = "scan_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), index=True)
    
    # Scan configuration
    name = Column(String(255))
    description = Column(Text)
    targets = Column(JSONB, nullable=False)  # List of targets
    scan_type = Column(String(50), nullable=False)  # quick, comprehensive, stealth, web-focused
    scan_profile = Column(String(50), default="basic")
    
    # Status and progress
    status = Column(Enum(ScanStatus), default=ScanStatus.QUEUED, nullable=False, index=True)
    progress_percentage = Column(Integer, default=0)
    current_phase = Column(String(100))
    
    # Configuration and parameters
    configuration = Column(JSONB, default={})
    environment_config = Column(JSONB, default={})
    scan_parameters = Column(JSONB, default={})
    
    # Results and findings
    results = Column(JSONB, default={})
    findings = Column(JSONB, default={})
    summary_stats = Column(JSONB, default={})
    
    # Compliance and reporting
    compliance_frameworks = Column(ARRAY(String))
    compliance_results = Column(JSONB, default={})
    
    # Error handling
    error_message = Column(Text)
    error_details = Column(JSONB)
    
    # Timing
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # Resource usage
    estimated_duration_minutes = Column(Integer)
    actual_duration_minutes = Column(Integer)
    cpu_usage_stats = Column(JSONB)
    memory_usage_stats = Column(JSONB)
    
    # Additional metadata  
    scan_metadata = Column(JSONB, default={})
    tags = Column(ARRAY(String))
    
    # Relationships
    tenant = relationship("TenantModel", back_populates="scan_sessions")
    user = relationship("UserModel", back_populates="scan_sessions")
    findings_detail = relationship("ScanFindingModel", back_populates="scan_session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_scan_tenant_id", "tenant_id"),
        Index("idx_scan_user_id", "user_id"),
        Index("idx_scan_status", "status"),
        Index("idx_scan_created_at", "created_at"),
        Index("idx_scan_type", "scan_type"),
        Index("idx_scan_completed_at", "completed_at"),
    )


class ScanFindingModel(Base):
    """Individual security findings from scans"""
    __tablename__ = "scan_findings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_session_id = Column(UUID(as_uuid=True), ForeignKey("scan_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Finding classification
    finding_type = Column(String(100), nullable=False)  # vulnerability, misconfiguration, etc.
    severity = Column(Enum(SeverityLevel), nullable=False, index=True)
    category = Column(String(100))
    subcategory = Column(String(100))
    
    # Finding details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    technical_details = Column(Text)
    
    # Target information
    target_host = Column(String(255))
    target_port = Column(Integer)
    target_service = Column(String(100))
    target_path = Column(String(1000))
    
    # Risk assessment
    risk_score = Column(DECIMAL(4, 2))  # 0.00 to 10.00
    cvss_score = Column(DECIMAL(3, 1))
    cvss_vector = Column(String(100))
    cve_ids = Column(ARRAY(String))
    
    # Evidence
    evidence = Column(JSONB)
    screenshots = Column(ARRAY(String))  # URLs or file paths
    raw_output = Column(Text)
    
    # Remediation
    remediation_advice = Column(Text)
    remediation_priority = Column(String(20))
    remediation_effort = Column(String(20))  # low, medium, high
    
    # Compliance mapping
    compliance_mappings = Column(JSONB)  # Maps to various frameworks
    mitre_attack_mapping = Column(JSONB)
    
    # Status tracking
    status = Column(String(50), default="new")  # new, verified, false_positive, fixed, accepted
    verified_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    verified_at = Column(TIMESTAMP)
    
    # Timestamps
    first_seen = Column(TIMESTAMP, default=func.now(), nullable=False)
    last_seen = Column(TIMESTAMP, default=func.now())
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # Relationships
    scan_session = relationship("ScanSessionModel", back_populates="findings_detail")
    
    # Indexes
    __table_args__ = (
        Index("idx_finding_scan_session", "scan_session_id"),
        Index("idx_finding_severity", "severity"),
        Index("idx_finding_type", "finding_type"),
        Index("idx_finding_status", "status"),
        Index("idx_finding_created_at", "created_at"),
        Index("idx_finding_target_host", "target_host"),
    )


class AuthTokenModel(Base):
    """Authentication tokens with enhanced security"""
    __tablename__ = "auth_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Token information
    token = Column(String(255), unique=True, nullable=False, index=True)
    token_type = Column(Enum(TokenType), default=TokenType.ACCESS, nullable=False)
    token_hash = Column(String(255))  # Hashed version for additional security
    
    # Scope and permissions
    scopes = Column(ARRAY(String))
    permissions = Column(JSONB, default={})
    
    # Status and lifecycle
    is_revoked = Column(Boolean, default=False, nullable=False)
    is_blacklisted = Column(Boolean, default=False, nullable=False)
    
    # Expiration
    expires_at = Column(TIMESTAMP, nullable=False, index=True)
    refresh_expires_at = Column(TIMESTAMP)
    
    # Security context
    client_ip = Column(INET)
    user_agent = Column(Text)
    client_id = Column(String(100))
    
    # Usage tracking
    last_used_at = Column(TIMESTAMP)
    usage_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    revoked_at = Column(TIMESTAMP)
    
    # Relationships
    user = relationship("UserModel", back_populates="auth_tokens")
    
    # Indexes
    __table_args__ = (
        Index("idx_token_user_id", "user_id"),
        Index("idx_token_expires_at", "expires_at"),
        Index("idx_token_revoked", "is_revoked"),
        Index("idx_token_type", "token_type"),
        Index("idx_token_last_used", "last_used_at"),
    )


class AuditLogModel(Base):
    """Comprehensive audit logging"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    
    # Event information
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False)  # auth, scan, admin, etc.
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    
    # Context
    description = Column(Text)
    details = Column(JSONB)
    
    # Request context
    ip_address = Column(INET)
    user_agent = Column(Text)
    request_id = Column(String(100))
    session_id = Column(String(100))
    
    # Result
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    
    # Risk scoring
    risk_level = Column(String(20))  # low, medium, high, critical
    risk_score = Column(Integer)
    
    # Timestamp
    timestamp = Column(TIMESTAMP, default=func.now(), nullable=False, index=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_audit_tenant_id", "tenant_id"),
        Index("idx_audit_user_id", "user_id"),
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_event_type", "event_type"),
        Index("idx_audit_event_category", "event_category"),
        Index("idx_audit_success", "success"),
    )


class DiscoveryWorkflowModel(Base):
    """Discovery workflow model"""
    __tablename__ = "discovery_workflows"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Workflow identification
    workflow_id = Column(String(100), unique=True, nullable=False, index=True)
    workflow_type = Column(String(50), default="domain_discovery")
    
    # Configuration
    domain = Column(String(255), nullable=False)
    configuration = Column(JSONB, default={})
    
    # Status
    status = Column(String(50), default="pending")
    progress = Column(Integer, default=0)
    
    # Results
    results = Column(JSONB, default={})
    discovered_assets = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("UserModel", back_populates="discovery_workflows")
    
    # Indexes
    __table_args__ = (
        Index("idx_discovery_user_id", "user_id"),
        Index("idx_discovery_workflow_id", "workflow_id"),
        Index("idx_discovery_status", "status"),
        Index("idx_discovery_created_at", "created_at"),
    )


class EmbeddingRequestModel(Base):
    """Embedding request tracking"""
    __tablename__ = "embedding_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Request details
    texts = Column(JSONB, nullable=False)
    model = Column(String(100), nullable=False)
    input_type = Column(String(50), default="query")
    
    # Processing
    status = Column(String(50), default="pending")
    tokens_used = Column(Integer, default=0)
    processing_time_ms = Column(Integer)
    
    # Results reference
    result_id = Column(UUID(as_uuid=True))
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    completed_at = Column(TIMESTAMP)
    
    # Indexes
    __table_args__ = (
        Index("idx_embedding_user_id", "user_id"),
        Index("idx_embedding_status", "status"),
        Index("idx_embedding_created_at", "created_at"),
    )


class EmbeddingResultModel(Base):
    """Embedding results storage"""
    __tablename__ = "embedding_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("embedding_requests.id"), nullable=False, index=True)
    
    # Results
    embeddings = Column(JSONB, nullable=False)
    dimensions = Column(Integer, nullable=False)
    
    # Metadata
    model_info = Column(JSONB)
    processing_metadata = Column(JSONB)
    
    # Timestamp
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_embedding_result_request", "request_id"),
        Index("idx_embedding_result_created", "created_at"),
    )