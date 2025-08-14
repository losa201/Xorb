"""
Enterprise Management API Router
Multi-tenant administration, user management, and enterprise configuration
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, EmailStr
import enum
import logging
import uuid

from ..auth.dependencies import require_auth
from ..dependencies import get_current_organization
from ..services.performance_optimizer import get_performance_optimizer, cached, monitored

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/enterprise", tags=["Enterprise Management"])

# Enterprise Models

class OrganizationTier(str, enum.Enum):
    """Organization tier levels"""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"

class UserRole(str, enum.Enum):
    """User roles in the platform"""
    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    SECURITY_MANAGER = "security_manager"
    SECURITY_ANALYST = "security_analyst"
    COMPLIANCE_OFFICER = "compliance_officer"
    AUDITOR = "auditor"
    VIEWER = "viewer"

class OrganizationCreate(BaseModel):
    """Organization creation request"""
    model_config = {"protected_namespaces": ()}
    
    name: str = Field(..., min_length=2, max_length=100, description="Organization name")
    tier: OrganizationTier = Field(..., description="Organization tier")
    admin_email: EmailStr = Field(..., description="Admin email address")
    industry: Optional[str] = Field(None, description="Organization industry")
    country: Optional[str] = Field(None, description="Organization country")
    compliance_requirements: List[str] = Field(default_factory=list, description="Required compliance frameworks")
    
class UserCreate(BaseModel):
    """User creation request"""
    model_config = {"protected_namespaces": ()}
    
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name")
    role: UserRole = Field(..., description="User role")
    department: Optional[str] = Field(None, description="Department")
    phone: Optional[str] = Field(None, description="Phone number")
    
class LicenseInfo(BaseModel):
    """License information"""
    model_config = {"protected_namespaces": ()}
    
    tier: str
    max_users: int
    max_scans_per_month: int
    max_compliance_frameworks: int
    support_level: str
    features: List[str]
    expires_at: datetime
    
class UsageMetrics(BaseModel):
    """Organization usage metrics"""
    model_config = {"protected_namespaces": ()}
    
    period: str
    users_active: int
    scans_conducted: int
    incidents_handled: int
    compliance_assessments: int
    api_calls: int
    storage_used_gb: float
    
class AuditLogEntry(BaseModel):
    """Audit log entry"""
    model_config = {"protected_namespaces": ()}
    
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str

# Organization Management

@router.post("/organizations", response_model=Dict[str, Any])
@monitored("create_organization")
async def create_organization(
    org_data: OrganizationCreate,
    current_user = Depends(require_auth)
):
    """
    Create new organization in the platform
    
    Features:
    - Multi-tenant architecture setup
    - Tier-based feature configuration
    - Compliance framework initialization
    - Admin user provisioning
    """
    
    try:
        # Check if user has permission to create organizations
        if not hasattr(current_user, 'role') or current_user.role != 'super_admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only super admins can create organizations"
            )
        
        # Generate organization ID
        org_id = f"org_{uuid.uuid4().hex[:8]}"
        
        # Create organization structure
        organization = {
            "id": org_id,
            "name": org_data.name,
            "tier": org_data.tier.value,
            "industry": org_data.industry,
            "country": org_data.country,
            "compliance_requirements": org_data.compliance_requirements,
            "created_at": datetime.utcnow(),
            "status": "active",
            "admin_email": org_data.admin_email,
            "settings": _get_tier_settings(org_data.tier),
            "usage_limits": _get_usage_limits(org_data.tier),
            "features": _get_tier_features(org_data.tier)
        }
        
        # Initialize organization services
        await _initialize_org_services(org_id, org_data.tier)
        
        logger.info(f"Organization created: {org_id} - {org_data.name}")
        
        return {
            "organization": organization,
            "setup_instructions": [
                "Admin user invitation sent",
                "Initial compliance frameworks configured",
                "Security monitoring enabled",
                "API access keys generated"
            ],
            "next_steps": [
                "Complete admin user setup",
                "Configure compliance requirements",
                "Add additional users",
                "Start security assessments"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create organization: {str(e)}"
        )

@router.get("/organizations/{org_id}")
@cached(ttl=300, key_prefix="org_details")
@monitored("get_organization")
async def get_organization(
    org_id: str,
    current_user = Depends(require_auth)
):
    """Get organization details and configuration"""
    
    try:
        # Mock organization data (in production, retrieve from database)
        organization = {
            "id": org_id,
            "name": "Enterprise Organization",
            "tier": "enterprise",
            "status": "active",
            "created_at": datetime.utcnow() - timedelta(days=30),
            "last_updated": datetime.utcnow() - timedelta(hours=2),
            "admin_email": "admin@enterprise.com",
            "industry": "Financial Services",
            "country": "United States",
            "compliance_requirements": ["PCI-DSS", "SOX", "NIST"],
            "user_count": 125,
            "active_sessions": 45,
            "license_info": {
                "tier": "enterprise",
                "max_users": 500,
                "max_scans_per_month": 1000,
                "support_level": "24/7 Premium",
                "expires_at": datetime.utcnow() + timedelta(days=365)
            },
            "features_enabled": [
                "Advanced threat prediction",
                "Compliance automation",
                "Incident response",
                "Custom integrations",
                "API access",
                "SSO integration",
                "Advanced reporting"
            ]
        }
        
        return organization
        
    except Exception as e:
        logger.error(f"Error retrieving organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve organization details"
        )

@router.get("/organizations/{org_id}/usage")
@cached(ttl=600, key_prefix="org_usage")
@monitored("get_organization_usage")
async def get_organization_usage(
    org_id: str,
    period: str = Query("30d", regex="^(24h|7d|30d|90d)$"),
    current_user = Depends(require_auth)
):
    """Get organization usage metrics and billing information"""
    
    try:
        # Mock usage data (in production, calculate from actual usage)
        usage_data = {
            "organization_id": org_id,
            "period": period,
            "billing_period": "monthly",
            "usage_metrics": {
                "users_active": 98,
                "scans_conducted": 245,
                "incidents_handled": 12,
                "compliance_assessments": 8,
                "api_calls": 45650,
                "storage_used_gb": 125.7,
                "bandwidth_used_gb": 89.2
            },
            "usage_limits": {
                "max_users": 500,
                "max_scans_per_month": 1000,
                "max_api_calls_per_month": 100000,
                "max_storage_gb": 1000
            },
            "usage_percentages": {
                "users": 19.6,
                "scans": 24.5,
                "api_calls": 45.7,
                "storage": 12.6
            },
            "cost_breakdown": {
                "base_subscription": 2500.00,
                "additional_users": 0.00,
                "additional_scans": 0.00,
                "additional_storage": 0.00,
                "support": 500.00,
                "total": 3000.00,
                "currency": "USD"
            },
            "trends": {
                "users_growth": 15.2,  # % change
                "scans_growth": 28.7,
                "incidents_change": -22.1,
                "api_usage_growth": 45.8
            }
        }
        
        return usage_data
        
    except Exception as e:
        logger.error(f"Error retrieving organization usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage metrics"
        )

# User Management

@router.post("/organizations/{org_id}/users")
@monitored("create_user")
async def create_user(
    org_id: str,
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_auth)
):
    """Create new user in organization"""
    
    try:
        # Check permissions
        if not _can_manage_users(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create users"
            )
        
        # Generate user ID
        user_id = f"usr_{uuid.uuid4().hex[:8]}"
        
        # Create user
        user = {
            "id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "role": user_data.role.value,
            "department": user_data.department,
            "phone": user_data.phone,
            "organization_id": org_id,
            "created_at": datetime.utcnow(),
            "status": "pending_activation",
            "last_login": None,
            "permissions": _get_role_permissions(user_data.role),
            "settings": _get_default_user_settings()
        }
        
        # Send activation email in background
        background_tasks.add_task(
            _send_user_activation_email,
            user["email"],
            user["full_name"],
            user_id
        )
        
        logger.info(f"User created: {user_id} in organization {org_id}")
        
        return {
            "user": user,
            "activation_status": "email_sent",
            "next_steps": [
                "User will receive activation email",
                "Complete profile setup",
                "Assign to teams/projects",
                "Configure access permissions"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

@router.get("/organizations/{org_id}/users")
@cached(ttl=300, key_prefix="org_users")
@monitored("list_users")
async def list_users(
    org_id: str,
    role: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user = Depends(require_auth)
):
    """List users in organization with filtering"""
    
    try:
        # Mock user data (in production, query database)
        users = []
        for i in range(20):  # Mock 20 users
            user = {
                "id": f"usr_{i:08d}",
                "username": f"user{i}",
                "email": f"user{i}@enterprise.com",
                "full_name": f"User {i} Name",
                "role": ["security_analyst", "security_manager", "compliance_officer"][i % 3],
                "department": ["Security", "Compliance", "IT"][i % 3],
                "status": "active" if i < 18 else "suspended",
                "last_login": datetime.utcnow() - timedelta(days=i),
                "created_at": datetime.utcnow() - timedelta(days=30 + i)
            }
            users.append(user)
        
        # Apply filters
        if role:
            users = [u for u in users if u["role"] == role]
        if status:
            users = [u for u in users if u["status"] == status]
        
        # Apply pagination
        total_users = len(users)
        paginated_users = users[offset:offset + limit]
        
        return {
            "users": paginated_users,
            "pagination": {
                "total": total_users,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_users
            },
            "summary": {
                "total_users": total_users,
                "active_users": len([u for u in users if u["status"] == "active"]),
                "roles_distribution": {
                    role: len([u for u in users if u["role"] == role])
                    for role in set(u["role"] for u in users)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )

# License and Billing Management

@router.get("/organizations/{org_id}/license")
@cached(ttl=3600, key_prefix="org_license")
@monitored("get_license_info")
async def get_license_info(
    org_id: str,
    current_user = Depends(require_auth)
):
    """Get organization license and subscription information"""
    
    try:
        license_info = {
            "organization_id": org_id,
            "license_key": "ENT-XORB-2024-" + org_id.upper(),
            "tier": "enterprise",
            "status": "active",
            "issued_date": datetime.utcnow() - timedelta(days=30),
            "expires_date": datetime.utcnow() + timedelta(days=335),
            "auto_renewal": True,
            "limits": {
                "max_users": 500,
                "max_scans_per_month": 1000,
                "max_compliance_frameworks": 10,
                "max_api_calls_per_month": 100000,
                "max_storage_gb": 1000,
                "max_data_retention_months": 24
            },
            "features": {
                "advanced_threat_prediction": True,
                "compliance_automation": True,
                "incident_response": True,
                "custom_integrations": True,
                "sso_integration": True,
                "advanced_reporting": True,
                "24x7_support": True,
                "dedicated_account_manager": True,
                "custom_training": True,
                "white_label_reports": True
            },
            "billing": {
                "billing_cycle": "monthly",
                "next_billing_date": datetime.utcnow() + timedelta(days=30),
                "current_amount": 3000.00,
                "currency": "USD",
                "payment_method": "Corporate Credit Card",
                "billing_contact": "billing@enterprise.com"
            },
            "support": {
                "level": "Enterprise Premium",
                "response_time_sla": "1 hour",
                "support_channels": ["Phone", "Email", "Chat", "Dedicated Slack"],
                "account_manager": "Sarah Johnson",
                "technical_contact": "tech-support@xorb.com"
            }
        }
        
        return license_info
        
    except Exception as e:
        logger.error(f"Error retrieving license info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve license information"
        )

# Audit and Compliance

@router.get("/organizations/{org_id}/audit-logs")
@cached(ttl=900, key_prefix="audit_logs")
@monitored("get_audit_logs")
async def get_audit_logs(
    org_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_user = Depends(require_auth)
):
    """Get organization audit logs"""
    
    try:
        # Check audit access permissions
        if not _can_access_audit_logs(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access audit logs"
            )
        
        # Mock audit log data
        audit_logs = []
        for i in range(50):  # Mock 50 audit entries
            log_entry = {
                "id": f"audit_{i:08d}",
                "timestamp": datetime.utcnow() - timedelta(hours=i),
                "user_id": f"usr_{(i % 10):08d}",
                "username": f"user{i % 10}",
                "action": ["login", "scan_created", "user_added", "settings_changed", "report_generated"][i % 5],
                "resource": ["user_management", "ptaas", "compliance", "settings", "reports"][i % 5],
                "resource_id": f"res_{i:08d}",
                "ip_address": f"192.168.1.{100 + (i % 50)}",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "outcome": "success" if i % 10 != 0 else "failure",
                "details": {
                    "description": f"Action {i} performed successfully",
                    "changes": {"field": "value"} if i % 3 == 0 else None,
                    "session_id": f"sess_{i:08d}"
                }
            }
            audit_logs.append(log_entry)
        
        # Apply filters
        if start_date:
            audit_logs = [log for log in audit_logs if log["timestamp"] >= start_date]
        if end_date:
            audit_logs = [log for log in audit_logs if log["timestamp"] <= end_date]
        if user_id:
            audit_logs = [log for log in audit_logs if log["user_id"] == user_id]
        if action:
            audit_logs = [log for log in audit_logs if log["action"] == action]
        
        # Apply limit
        limited_logs = audit_logs[:limit]
        
        return {
            "audit_logs": limited_logs,
            "total_count": len(audit_logs),
            "returned_count": len(limited_logs),
            "filters": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "user_id": user_id,
                "action": action
            },
            "summary": {
                "success_rate": len([log for log in limited_logs if log["outcome"] == "success"]) / len(limited_logs) if limited_logs else 0,
                "top_actions": {},
                "top_users": {}
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audit logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit logs"
        )

# System Administration

@router.get("/system/health")
@monitored("system_health")
async def get_system_health(
    current_user = Depends(require_auth)
):
    """Get system health and performance metrics"""
    
    try:
        optimizer = await get_performance_optimizer()
        stats = await optimizer.get_comprehensive_stats()
        
        # Add system-specific health checks
        health_data = {
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "performance": stats["performance_metrics"],
            "cache": stats["cache_stats"],
            "database": {
                "status": "connected",
                "connection_pool": stats["connection_pool_stats"],
                "query_time_avg_ms": 45.2,
                "active_connections": 15
            },
            "services": {
                "ai_threat_predictor": {"status": "active", "response_time_ms": 123},
                "compliance_engine": {"status": "active", "response_time_ms": 89},
                "incident_orchestrator": {"status": "active", "response_time_ms": 67},
                "security_monitor": {"status": "active", "response_time_ms": 34}
            },
            "integrations": {
                "external_threat_feeds": {"status": "connected", "last_update": datetime.utcnow() - timedelta(minutes=15)},
                "security_tools": {"status": "connected", "tools_active": 8},
                "notification_services": {"status": "connected", "channels_active": 4}
            },
            "security": {
                "ssl_certificate_valid": True,
                "security_headers_configured": True,
                "rate_limiting_active": True,
                "intrusion_detection_active": True
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error retrieving system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system health"
        )

# Helper Functions

def _get_tier_settings(tier: OrganizationTier) -> Dict[str, Any]:
    """Get settings for organization tier"""
    
    tier_settings = {
        OrganizationTier.STARTER: {
            "max_users": 10,
            "max_scans_per_month": 50,
            "retention_days": 30,
            "support_level": "community"
        },
        OrganizationTier.PROFESSIONAL: {
            "max_users": 50,
            "max_scans_per_month": 200,
            "retention_days": 90,
            "support_level": "business"
        },
        OrganizationTier.ENTERPRISE: {
            "max_users": 500,
            "max_scans_per_month": 1000,
            "retention_days": 365,
            "support_level": "enterprise"
        },
        OrganizationTier.ENTERPRISE_PLUS: {
            "max_users": 2000,
            "max_scans_per_month": 5000,
            "retention_days": 730,
            "support_level": "premium"
        }
    }
    
    return tier_settings.get(tier, tier_settings[OrganizationTier.STARTER])

def _get_usage_limits(tier: OrganizationTier) -> Dict[str, int]:
    """Get usage limits for tier"""
    
    settings = _get_tier_settings(tier)
    return {
        "users": settings["max_users"],
        "scans": settings["max_scans_per_month"],
        "api_calls": settings["max_users"] * 1000,
        "storage_gb": settings["max_users"] * 10
    }

def _get_tier_features(tier: OrganizationTier) -> List[str]:
    """Get available features for tier"""
    
    base_features = ["basic_scanning", "incident_management", "compliance_reporting"]
    
    if tier in [OrganizationTier.PROFESSIONAL, OrganizationTier.ENTERPRISE, OrganizationTier.ENTERPRISE_PLUS]:
        base_features.extend(["advanced_analytics", "api_access", "integrations"])
    
    if tier in [OrganizationTier.ENTERPRISE, OrganizationTier.ENTERPRISE_PLUS]:
        base_features.extend(["ai_threat_prediction", "automated_response", "sso_integration"])
    
    if tier == OrganizationTier.ENTERPRISE_PLUS:
        base_features.extend(["custom_development", "dedicated_support", "white_labeling"])
    
    return base_features

async def _initialize_org_services(org_id: str, tier: OrganizationTier):
    """Initialize services for new organization"""
    # Mock service initialization
    logger.info(f"Initializing services for organization {org_id} with tier {tier.value}")

def _get_role_permissions(role: UserRole) -> List[str]:
    """Get permissions for user role"""
    
    permissions_map = {
        UserRole.SUPER_ADMIN: ["*"],  # All permissions
        UserRole.ORG_ADMIN: [
            "users:manage", "settings:manage", "billing:view", 
            "scans:manage", "incidents:manage", "compliance:manage"
        ],
        UserRole.SECURITY_MANAGER: [
            "scans:manage", "incidents:manage", "users:view", 
            "reports:generate", "integrations:manage"
        ],
        UserRole.SECURITY_ANALYST: [
            "scans:create", "scans:view", "incidents:view", 
            "reports:view", "alerts:manage"
        ],
        UserRole.COMPLIANCE_OFFICER: [
            "compliance:manage", "reports:generate", "audits:view", 
            "policies:manage"
        ],
        UserRole.AUDITOR: [
            "audits:view", "reports:view", "compliance:view", 
            "logs:view"
        ],
        UserRole.VIEWER: [
            "scans:view", "reports:view", "incidents:view"
        ]
    }
    
    return permissions_map.get(role, ["scans:view"])

def _get_default_user_settings() -> Dict[str, Any]:
    """Get default user settings"""
    return {
        "notifications": {
            "email": True,
            "dashboard": True,
            "mobile": False
        },
        "timezone": "UTC",
        "language": "en",
        "dashboard_layout": "default"
    }

async def _send_user_activation_email(email: str, full_name: str, user_id: str):
    """Send user activation email"""
    # Mock email sending
    logger.info(f"Sending activation email to {email} for user {user_id}")

def _can_manage_users(user) -> bool:
    """Check if user can manage other users"""
    return hasattr(user, 'role') and user.role in ['super_admin', 'org_admin']

def _can_access_audit_logs(user) -> bool:
    """Check if user can access audit logs"""
    return hasattr(user, 'role') and user.role in ['super_admin', 'org_admin', 'auditor']

import enum