"""
Rate limiting administration endpoints.

Provides API endpoints for:
- Policy management and configuration
- Emergency controls and kill-switch
- Monitoring and health checks
- Performance tuning and optimization
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from ..auth.rbac_dependencies import require_permissions, RBACContext
from ..core.secure_tenant_context import get_tenant_context
from ..utils.rate_limit_manager import RateLimitManager

logger = structlog.get_logger("rate_limiting_admin")

router = APIRouter(prefix="/rate-limiting", tags=["Rate Limiting Admin"])


class PolicyOverrideRequest(BaseModel):
    """Request model for creating policy overrides"""
    scope: str = Field(..., description="Policy scope: ip, user, tenant, endpoint")
    requests_per_second: float = Field(..., gt=0, description="Requests per second limit")
    burst_size: int = Field(..., gt=0, description="Burst size allowance")
    description: Optional[str] = Field(None, description="Override description")


class EmergencyControlRequest(BaseModel):
    """Request model for emergency controls"""
    duration_seconds: Optional[int] = Field(300, gt=0, le=3600, description="Duration in seconds")
    reason: Optional[str] = Field(None, description="Reason for activation")


class PolicyStatsResponse(BaseModel):
    """Response model for policy statistics"""
    global_policies: Dict[str, Any]
    tenant_overrides: Dict[str, int]
    role_overrides: Dict[str, int]
    endpoint_overrides: int
    emergency_overrides: int
    cache_size: int


class HealthStatusResponse(BaseModel):
    """Response model for health status"""
    overall_status: str
    timestamp: float
    redis: Dict[str, Any]
    policies: Dict[str, Any]
    rate_limiter: Dict[str, Any]
    observability: Dict[str, Any]


async def get_rate_limit_manager(request: Request) -> RateLimitManager:
    """Get rate limit manager from app state"""
    if not hasattr(request.app.state, 'redis_client'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiting not initialized"
        )
    
    # Create manager with existing Redis connection
    manager = RateLimitManager()
    manager.redis_client = request.app.state.redis_client
    
    # Initialize components
    if hasattr(request.app.state, 'rate_limit_middleware'):
        middleware = request.app.state.rate_limit_middleware
        manager.policy_manager = middleware.policy_manager
        manager.emergency_limiter = middleware.emergency_limiter
        manager.observability = middleware.observability
    
    return manager


@router.get(
    "/policies",
    response_model=PolicyStatsResponse,
    summary="List Rate Limiting Policies",
    description="Get comprehensive list of all rate limiting policies and overrides"
)
async def list_policies(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:read", "admin"]))
):
    """List all rate limiting policies and overrides"""
    try:
        policies = await manager.list_policies()
        return PolicyStatsResponse(**policies)
    
    except Exception as e:
        logger.error("Failed to list policies", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve policies: {str(e)}"
        )


@router.post(
    "/policies/tenant/{tenant_id}",
    summary="Create Tenant Override",
    description="Create rate limiting override for specific tenant"
)
async def create_tenant_override(
    tenant_id: str,
    override_request: PolicyOverrideRequest,
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:write", "admin"]))
):
    """Create tenant-specific rate limit override"""
    try:
        success = await manager.create_tenant_override(
            tenant_id=tenant_id,
            scope=override_request.scope,
            requests_per_second=override_request.requests_per_second,
            burst_size=override_request.burst_size,
            description=override_request.description
        )
        
        if success:
            logger.info(
                "Tenant override created",
                tenant_id=tenant_id,
                scope=override_request.scope,
                requests_per_second=override_request.requests_per_second,
                created_by=rbac.user_id
            )
            return {"success": True, "message": "Tenant override created successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create tenant override"
            )
    
    except Exception as e:
        logger.error("Failed to create tenant override", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tenant override: {str(e)}"
        )


@router.post(
    "/policies/role/{role_name}",
    summary="Create Role Override",
    description="Create rate limiting override for specific role"
)
async def create_role_override(
    role_name: str,
    override_request: PolicyOverrideRequest,
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:write", "admin"]))
):
    """Create role-specific rate limit override"""
    try:
        success = await manager.create_role_override(
            role_name=role_name,
            scope=override_request.scope,
            requests_per_second=override_request.requests_per_second,
            burst_size=override_request.burst_size,
            description=override_request.description
        )
        
        if success:
            logger.info(
                "Role override created",
                role_name=role_name,
                scope=override_request.scope,
                requests_per_second=override_request.requests_per_second,
                created_by=rbac.user_id
            )
            return {"success": True, "message": "Role override created successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create role override"
            )
    
    except Exception as e:
        logger.error("Failed to create role override", error=str(e), role_name=role_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create role override: {str(e)}"
        )


@router.delete(
    "/policies/tenant/{tenant_id}",
    summary="Remove Tenant Override",
    description="Remove rate limiting override for specific tenant"
)
async def remove_tenant_override(
    tenant_id: str,
    scope: Optional[str] = None,
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:write", "admin"]))
):
    """Remove tenant-specific rate limit override"""
    try:
        success = await manager.remove_tenant_override(tenant_id, scope)
        
        if success:
            logger.info(
                "Tenant override removed",
                tenant_id=tenant_id,
                scope=scope,
                removed_by=rbac.user_id
            )
            return {"success": True, "message": "Tenant override removed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant override not found"
            )
    
    except Exception as e:
        logger.error("Failed to remove tenant override", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove tenant override: {str(e)}"
        )


@router.get(
    "/emergency/status",
    summary="Emergency Status",
    description="Get current emergency control status"
)
async def get_emergency_status(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:read", "admin"]))
):
    """Get emergency control status"""
    try:
        status_info = await manager.check_emergency_status()
        return status_info
    
    except Exception as e:
        logger.error("Failed to get emergency status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get emergency status: {str(e)}"
        )


@router.post(
    "/emergency/activate",
    summary="Activate Emergency Mode",
    description="Activate emergency rate limiting (very restrictive)"
)
async def activate_emergency_mode(
    request: EmergencyControlRequest,
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:emergency", "admin"]))
):
    """Activate emergency rate limiting"""
    try:
        success = await manager.activate_emergency_mode(request.duration_seconds)
        
        if success:
            logger.critical(
                "Emergency mode activated via API",
                duration_seconds=request.duration_seconds,
                reason=request.reason,
                activated_by=rbac.user_id
            )
            return {
                "success": True,
                "message": f"Emergency mode activated for {request.duration_seconds} seconds",
                "duration_seconds": request.duration_seconds
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to activate emergency mode"
            )
    
    except Exception as e:
        logger.error("Failed to activate emergency mode", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate emergency mode: {str(e)}"
        )


@router.post(
    "/emergency/deactivate",
    summary="Deactivate Emergency Mode",
    description="Deactivate emergency rate limiting"
)
async def deactivate_emergency_mode(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:emergency", "admin"]))
):
    """Deactivate emergency rate limiting"""
    try:
        success = await manager.deactivate_emergency_mode()
        
        if success:
            logger.critical(
                "Emergency mode deactivated via API",
                deactivated_by=rbac.user_id
            )
            return {"success": True, "message": "Emergency mode deactivated"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to deactivate emergency mode"
            )
    
    except Exception as e:
        logger.error("Failed to deactivate emergency mode", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate emergency mode: {str(e)}"
        )


@router.post(
    "/emergency/kill-switch/activate",
    summary="Activate Kill-Switch",
    description="⚠️ DANGER: Activate kill-switch (blocks ALL requests)"
)
async def activate_kill_switch(
    request: EmergencyControlRequest,
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:kill_switch", "super_admin"]))
):
    """Activate emergency kill-switch (blocks all requests)"""
    try:
        success = await manager.activate_kill_switch()
        
        if success:
            logger.critical(
                "KILL-SWITCH ACTIVATED via API - ALL REQUESTS WILL BE BLOCKED",
                reason=request.reason,
                activated_by=rbac.user_id
            )
            return {
                "success": True,
                "message": "⚠️ KILL-SWITCH ACTIVATED - ALL REQUESTS WILL BE BLOCKED",
                "warning": "Use /emergency/kill-switch/deactivate to restore service"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to activate kill-switch"
            )
    
    except Exception as e:
        logger.error("Failed to activate kill-switch", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate kill-switch: {str(e)}"
        )


@router.post(
    "/emergency/kill-switch/deactivate",
    summary="Deactivate Kill-Switch",
    description="Deactivate kill-switch and restore normal service"
)
async def deactivate_kill_switch(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:kill_switch", "super_admin"]))
):
    """Deactivate emergency kill-switch"""
    try:
        success = await manager.deactivate_kill_switch()
        
        if success:
            logger.critical(
                "Kill-switch deactivated via API - Service restored",
                deactivated_by=rbac.user_id
            )
            return {"success": True, "message": "Kill-switch deactivated - Service restored"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to deactivate kill-switch"
            )
    
    except Exception as e:
        logger.error("Failed to deactivate kill-switch", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate kill-switch: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthStatusResponse,
    summary="Rate Limiting Health",
    description="Get comprehensive health status of rate limiting system"
)
async def get_health_status(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:read", "admin"]))
):
    """Get comprehensive health status"""
    try:
        health = await manager.get_health_status()
        return HealthStatusResponse(**health)
    
    except Exception as e:
        logger.error("Failed to get health status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Rate Limiting Statistics",
    description="Get detailed rate limiting statistics and metrics"
)
async def get_rate_limiting_stats(
    scope: Optional[str] = None,
    tenant_id: Optional[str] = None,
    time_window_minutes: int = 5,
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:read", "admin"]))
):
    """Get rate limiting statistics"""
    try:
        stats = await manager.get_rate_limit_stats(
            scope=scope,
            tenant_id=tenant_id,
            time_window_minutes=time_window_minutes
        )
        return stats
    
    except Exception as e:
        logger.error("Failed to get rate limiting stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get(
    "/top-limited-ips",
    summary="Top Rate-Limited IPs",
    description="Get list of most rate-limited IP addresses"
)
async def get_top_limited_ips(
    limit: int = 10,
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:read", "admin"]))
):
    """Get top rate-limited IP addresses"""
    try:
        top_ips = await manager.get_top_limited_ips(limit)
        return {"top_limited_ips": top_ips, "limit": limit}
    
    except Exception as e:
        logger.error("Failed to get top limited IPs", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get top limited IPs: {str(e)}"
        )


@router.post(
    "/optimize",
    summary="Optimize Policies",
    description="Analyze current performance and suggest policy optimizations"
)
async def optimize_policies(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:write", "admin"]))
):
    """Get policy optimization recommendations"""
    try:
        recommendations = await manager.optimize_policies()
        return recommendations
    
    except Exception as e:
        logger.error("Failed to optimize policies", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize policies: {str(e)}"
        )


@router.post(
    "/maintenance/cleanup",
    summary="Cleanup Expired Data",
    description="Clean up expired rate limiting data and optimize performance"
)
async def cleanup_expired_data(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:write", "admin"]))
):
    """Clean up expired rate limiting data"""
    try:
        cleanup_results = await manager.cleanup_expired_data()
        
        logger.info(
            "Rate limiting cleanup performed",
            cleanup_results=cleanup_results,
            performed_by=rbac.user_id
        )
        
        return cleanup_results
    
    except Exception as e:
        logger.error("Failed to cleanup expired data", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup expired data: {str(e)}"
        )


@router.get(
    "/shadow-mode/status",
    summary="Shadow Mode Status",
    description="Get current shadow mode status"
)
async def get_shadow_mode_status(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:read", "admin"]))
):
    """Get shadow mode status"""
    try:
        status_info = await manager.get_shadow_mode_status()
        return status_info
    
    except Exception as e:
        logger.error("Failed to get shadow mode status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get shadow mode status: {str(e)}"
        )


@router.post(
    "/shadow-mode/enable",
    summary="Enable Shadow Mode",
    description="Enable shadow mode for safe testing (logs blocks but allows requests)"
)
async def enable_shadow_mode(
    percentage: float = 100.0,
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:write", "admin"]))
):
    """Enable shadow mode"""
    try:
        success = await manager.enable_shadow_mode(percentage)
        
        if success:
            logger.info(
                "Shadow mode enabled",
                percentage=percentage,
                enabled_by=rbac.user_id
            )
            return {
                "success": True,
                "message": f"Shadow mode enabled at {percentage}%",
                "percentage": percentage
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enable shadow mode"
            )
    
    except Exception as e:
        logger.error("Failed to enable shadow mode", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable shadow mode: {str(e)}"
        )


@router.post(
    "/shadow-mode/disable",
    summary="Disable Shadow Mode",
    description="Disable shadow mode and return to normal rate limiting"
)
async def disable_shadow_mode(
    manager: RateLimitManager = Depends(get_rate_limit_manager),
    rbac: RBACContext = Depends(require_permissions(["rate_limiting:write", "admin"]))
):
    """Disable shadow mode"""
    try:
        success = await manager.disable_shadow_mode()
        
        if success:
            logger.info(
                "Shadow mode disabled",
                disabled_by=rbac.user_id
            )
            return {"success": True, "message": "Shadow mode disabled"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to disable shadow mode"
            )
    
    except Exception as e:
        logger.error("Failed to disable shadow mode", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable shadow mode: {str(e)}"
        )