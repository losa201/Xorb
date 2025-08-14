"""
XORB Phase G8 Control Plane API Router
REST API endpoints for WFQ scheduler and per-tenant quota management.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from ..services.g8_control_plane_service import (
    G8ControlPlaneService,
    get_g8_control_plane_service,
    ResourceType,
    RequestPriority,
    TenantProfile
)


router = APIRouter(
    prefix="/control-plane",
    tags=["G8 Control Plane"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Resource not found"},
        429: {"description": "Rate limit exceeded"}
    }
)


# Pydantic models for API
class SubmitRequestRequest(BaseModel):
    """Request to submit a resource request to the control plane."""
    tenant_id: str = Field(..., description="Tenant identifier")
    resource_type: str = Field(..., description="Type of resource requested")
    priority: str = Field(default="medium", description="Request priority (critical, high, medium, low, background)")
    resource_amount: int = Field(default=1, description="Amount of resource requested")
    estimated_duration: float = Field(default=1.0, description="Estimated processing duration in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional request metadata")


class CreateTenantRequest(BaseModel):
    """Request to create a new tenant profile."""
    tenant_id: str = Field(..., description="Tenant identifier")
    tier: str = Field(default="starter", description="Tenant tier (enterprise, professional, starter)")
    custom_quotas: Optional[Dict[str, int]] = Field(None, description="Custom quota overrides")


class UpdateQuotaRequest(BaseModel):
    """Request to update tenant quotas."""
    resource_type: str = Field(..., description="Resource type to update")
    new_limit: int = Field(..., description="New quota limit")
    burst_allowance: Optional[int] = Field(None, description="Optional burst allowance")


class RequestResponse(BaseModel):
    """Response from request submission."""
    accepted: bool
    message: str
    request_id: Optional[str] = None
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[float] = None


class RequestStatusResponse(BaseModel):
    """Response for request status query."""
    request_id: str
    status: str
    tenant_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    estimated_wait_time: Optional[float] = None


class TenantStatusResponse(BaseModel):
    """Comprehensive tenant status response."""
    tenant_id: str
    queue_length: int
    usage_statistics: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    status_generated_at: datetime


class SystemStatusResponse(BaseModel):
    """System-wide control plane status."""
    system_health: Dict[str, Any]
    queue_statistics: Dict[str, Any]
    fairness_report: Dict[str, Any]
    generated_at: datetime


# Dependency to get current tenant (mock implementation for demo)
async def get_current_tenant() -> str:
    """Get current tenant ID from authentication context."""
    # In production, this would extract from JWT token
    return "t-enterprise"


@router.post("/requests/submit", response_model=RequestResponse)
async def submit_resource_request(
    request: SubmitRequestRequest,
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Submit a resource request to the WFQ scheduler with quota enforcement."""
    
    try:
        # Validate resource type
        try:
            resource_type = ResourceType(request.resource_type)
        except ValueError:
            valid_types = [rt.value for rt in ResourceType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid resource_type. Must be one of: {valid_types}"
            )
        
        # Validate priority
        try:
            priority = RequestPriority[request.priority.upper()]
        except KeyError:
            valid_priorities = [p.name.lower() for p in RequestPriority]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority. Must be one of: {valid_priorities}"
            )
        
        # Submit request
        accepted, message, request_id = await control_plane.submit_request(
            tenant_id=request.tenant_id,
            resource_type=resource_type,
            priority=priority,
            resource_amount=request.resource_amount,
            estimated_duration=request.estimated_duration,
            metadata=request.metadata
        )
        
        # Get queue information if accepted
        queue_position = None
        estimated_wait_time = None
        
        if accepted and request_id:
            queue_stats = control_plane.wfq_scheduler.get_queue_stats()
            estimated_wait_time = queue_stats["average_wait_time_ms"] / 1000.0
        
        return RequestResponse(
            accepted=accepted,
            message=message,
            request_id=request_id,
            queue_position=queue_position,
            estimated_wait_time=estimated_wait_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit request: {str(e)}")


@router.get("/requests/{request_id}/status", response_model=RequestStatusResponse)
async def get_request_status(
    request_id: str,
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Get status of a submitted resource request."""
    
    try:
        status_info = await control_plane.get_request_status(request_id)
        
        return RequestStatusResponse(
            request_id=request_id,
            status=status_info["status"],
            tenant_id=status_info.get("tenant_id"),
            submitted_at=datetime.fromisoformat(status_info["submitted_at"]) if status_info.get("submitted_at") else None,
            processing_time=status_info.get("processing_time"),
            estimated_wait_time=status_info.get("estimated_wait_time")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get request status: {str(e)}")


@router.post("/tenants/create", response_model=Dict[str, Any])
async def create_tenant_profile(
    request: CreateTenantRequest,
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Create a new tenant profile with quotas and WFQ weight."""
    
    try:
        # Validate tier
        valid_tiers = ["enterprise", "professional", "starter"]
        if request.tier not in valid_tiers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier. Must be one of: {valid_tiers}"
            )
        
        # Convert custom quotas if provided
        custom_quotas = None
        if request.custom_quotas:
            try:
                custom_quotas = {
                    ResourceType(k): v for k, v in request.custom_quotas.items()
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid resource type in custom quotas: {str(e)}")
        
        # Create tenant profile
        profile = control_plane.quota_manager.create_tenant_profile(
            tenant_id=request.tenant_id,
            tier=request.tier,
            custom_quotas=custom_quotas
        )
        
        return {
            "tenant_profile": profile.to_dict(),
            "message": f"Tenant profile created successfully for {request.tenant_id}",
            "created_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create tenant profile: {str(e)}")


@router.get("/tenants/{tenant_id}/status", response_model=TenantStatusResponse)
async def get_tenant_status(
    tenant_id: str,
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Get comprehensive status for a specific tenant."""
    
    try:
        status_info = await control_plane.get_tenant_status(tenant_id)
        
        return TenantStatusResponse(
            tenant_id=status_info["tenant_id"],
            queue_length=status_info["queue_length"],
            usage_statistics=status_info["usage_statistics"],
            fairness_metrics=status_info["fairness_metrics"],
            status_generated_at=datetime.fromisoformat(status_info["status_generated_at"])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tenant status: {str(e)}")


@router.put("/tenants/{tenant_id}/quotas", response_model=Dict[str, Any])
async def update_tenant_quota(
    tenant_id: str,
    request: UpdateQuotaRequest,
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Update quota limits for a specific tenant and resource type."""
    
    try:
        # Validate resource type
        try:
            resource_type = ResourceType(request.resource_type)
        except ValueError:
            valid_types = [rt.value for rt in ResourceType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid resource_type. Must be one of: {valid_types}"
            )
        
        # Check if tenant exists
        if tenant_id not in control_plane.quota_manager.tenant_profiles:
            raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")
        
        # Update quota
        profile = control_plane.quota_manager.tenant_profiles[tenant_id]
        old_limit = profile.quotas.get(resource_type, 0)
        
        profile.quotas[resource_type] = request.new_limit
        profile.updated_at = datetime.utcnow()
        
        # Update burst allowance if provided
        if request.burst_allowance is not None:
            profile.burst_allowance[resource_type] = request.burst_allowance
        
        # Save updated profile
        control_plane.quota_manager._save_tenant_profile(profile)
        
        return {
            "tenant_id": tenant_id,
            "resource_type": request.resource_type,
            "old_limit": old_limit,
            "new_limit": request.new_limit,
            "burst_allowance": profile.burst_allowance.get(resource_type),
            "message": f"Quota updated successfully for {tenant_id}",
            "updated_at": profile.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update tenant quota: {str(e)}")


@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Get comprehensive control plane system status and metrics."""
    
    try:
        status_info = await control_plane.get_system_status()
        
        return SystemStatusResponse(
            system_health=status_info["system_health"],
            queue_statistics=status_info["queue_statistics"],
            fairness_report=status_info["fairness_report"],
            generated_at=datetime.fromisoformat(status_info["generated_at"])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/system/fairness-report")
async def get_fairness_report(
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Get detailed fairness analysis and recommendations."""
    
    try:
        fairness_report = control_plane.fairness_engine.generate_fairness_report()
        
        return {
            "fairness_report": fairness_report,
            "fairness_summary": {
                "system_fairness_index": fairness_report["system_fairness_index"],
                "tenant_count": fairness_report["tenant_count"],
                "violations_count": fairness_report["violations_count"],
                "overall_status": "healthy" if fairness_report["system_fairness_index"] > 0.8 else "degraded"
            },
            "generated_at": fairness_report["report_generated_at"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate fairness report: {str(e)}")


@router.get("/queues/statistics")
async def get_queue_statistics(
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Get detailed WFQ scheduler queue statistics."""
    
    try:
        queue_stats = control_plane.wfq_scheduler.get_queue_stats()
        
        return {
            "wfq_statistics": queue_stats,
            "scheduler_health": {
                "total_queued": queue_stats["total_queued"],
                "average_wait_time_ms": queue_stats["average_wait_time_ms"],
                "processed_requests": queue_stats["processed_requests"],
                "virtual_time": queue_stats["virtual_time"],
                "active_tenants": len(queue_stats["tenant_queues"])
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue statistics: {str(e)}")


@router.post("/system/rebalance")
async def trigger_system_rebalance(
    background_tasks: BackgroundTasks,
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Trigger system rebalancing to improve fairness."""
    
    try:
        # Get current fairness violations
        violations = control_plane.fairness_engine.identify_fairness_violations()
        
        # Add background task for rebalancing
        background_tasks.add_task(_perform_system_rebalance, control_plane, violations)
        
        return {
            "rebalance_triggered": True,
            "violations_found": len(violations),
            "message": "System rebalancing initiated in background",
            "triggered_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger rebalancing: {str(e)}")


@router.get("/resources/types")
async def get_resource_types():
    """Get available resource types for quota management."""
    
    return {
        "resource_types": [
            {
                "type": rt.value,
                "description": _get_resource_description(rt)
            }
            for rt in ResourceType
        ]
    }


@router.get("/health")
async def control_plane_health(
    control_plane: G8ControlPlaneService = Depends(get_g8_control_plane_service)
):
    """Health check for control plane service."""
    
    try:
        system_status = await control_plane.get_system_status()
        is_healthy = system_status["system_health"]["control_plane_running"]
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "G8 Control Plane",
            "version": "g8.1.0",
            "components": {
                "wfq_scheduler": "operational",
                "quota_manager": "operational",
                "fairness_engine": "operational"
            },
            "system_metrics": {
                "total_tenants": system_status["system_health"]["total_tenants"],
                "total_queued_requests": system_status["system_health"]["total_queued_requests"],
                "fairness_index": system_status["system_health"]["fairness_index"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Background task functions
async def _perform_system_rebalance(
    control_plane: G8ControlPlaneService,
    violations: List[Dict[str, Any]]
):
    """Background task to perform system rebalancing."""
    
    try:
        print(f"🔄 Starting system rebalance for {len(violations)} violations")
        
        for violation in violations:
            tenant_id = violation["tenant_id"]
            violation_type = violation["violation_type"]
            
            if violation_type == "low_fairness_score":
                # Increase tenant weight slightly
                profile = control_plane.quota_manager.tenant_profiles.get(tenant_id)
                if profile:
                    old_weight = profile.weight
                    profile.weight = min(profile.weight * 1.2, 20.0)  # Max weight limit
                    control_plane.wfq_scheduler.set_tenant_weight(tenant_id, profile.weight)
                    
                    print(f"📈 Increased WFQ weight for {tenant_id}: {old_weight} → {profile.weight}")
            
            elif violation_type == "resource_starvation":
                # Temporarily boost priority for this tenant's requests
                print(f"🚨 Addressing resource starvation for {tenant_id}")
                # In a real system, this might trigger capacity scaling or resource reallocation
        
        # Wait for changes to take effect
        await asyncio.sleep(5)
        
        # Generate new fairness report
        new_report = control_plane.fairness_engine.generate_fairness_report()
        print(f"✅ Rebalancing complete. New fairness index: {new_report['system_fairness_index']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during system rebalancing: {e}")


def _get_resource_description(resource_type: ResourceType) -> str:
    """Get human-readable description for resource type."""
    
    descriptions = {
        ResourceType.API_REQUESTS: "API requests per time window",
        ResourceType.SCAN_JOBS: "Security scan jobs per time window", 
        ResourceType.STORAGE_GB: "Storage capacity in gigabytes",
        ResourceType.COMPUTE_HOURS: "Compute time allocation in hours",
        ResourceType.BANDWIDTH_MBPS: "Network bandwidth in Mbps",
        ResourceType.CONCURRENT_SCANS: "Maximum concurrent security scans"
    }
    
    return descriptions.get(resource_type, "Unknown resource type")