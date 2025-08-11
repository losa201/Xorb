"""
Metrics and monitoring endpoints for production observability
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from typing import Dict, Any, Optional
import logging

from ..container import get_container
from ..services.production_metrics_service import ProductionMetricsService
from ..services.health_service import ProductionHealthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics & Monitoring"])


async def get_metrics_service() -> ProductionMetricsService:
    """Get metrics service from container"""
    container = get_container()
    return container.get(ProductionMetricsService)


async def get_health_service() -> ProductionHealthService:
    """Get health service from container"""
    container = get_container()
    return container.get(ProductionHealthService)


@router.get("/prometheus")
async def get_prometheus_metrics(
    metrics_service: ProductionMetricsService = Depends(get_metrics_service)
):
    """Get Prometheus formatted metrics"""
    try:
        prometheus_data = metrics_service.get_prometheus_metrics()
        return Response(content=prometheus_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/api")
async def get_api_metrics(
    endpoint: Optional[str] = None,
    hours: int = 24,
    metrics_service: ProductionMetricsService = Depends(get_metrics_service)
) -> Dict[str, Any]:
    """Get API performance metrics"""
    try:
        return await metrics_service.get_api_metrics(endpoint=endpoint, hours=hours)
    except Exception as e:
        logger.error(f"Failed to get API metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve API metrics")


@router.get("/ptaas")
async def get_ptaas_metrics(
    hours: int = 24,
    metrics_service: ProductionMetricsService = Depends(get_metrics_service)
) -> Dict[str, Any]:
    """Get PTaaS performance metrics"""
    try:
        return await metrics_service.get_ptaas_metrics(hours=hours)
    except Exception as e:
        logger.error(f"Failed to get PTaaS metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve PTaaS metrics")


@router.get("/security")
async def get_security_metrics(
    hours: int = 24,
    metrics_service: ProductionMetricsService = Depends(get_metrics_service)
) -> Dict[str, Any]:
    """Get security event metrics"""
    try:
        return await metrics_service.get_security_metrics(hours=hours)
    except Exception as e:
        logger.error(f"Failed to get security metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security metrics")


@router.get("/health/detailed")
async def get_detailed_health(
    health_service: ProductionHealthService = Depends(get_health_service)
) -> Dict[str, Any]:
    """Get detailed system health information"""
    try:
        return await health_service.get_system_health()
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve health information")


@router.get("/health/{service_name}")
async def get_service_health(
    service_name: str,
    health_service: ProductionHealthService = Depends(get_health_service)
) -> Dict[str, Any]:
    """Get health status for a specific service"""
    try:
        return await health_service.check_service_health(service_name)
    except Exception as e:
        logger.error(f"Failed to get health for service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check health of {service_name}")


@router.post("/export")
async def export_metrics(
    format: str = "json",
    filepath: str = "/tmp/xorb_metrics_export",
    metrics_service: ProductionMetricsService = Depends(get_metrics_service)
) -> Dict[str, Any]:
    """Export metrics to file"""
    try:
        success = await metrics_service.export_metrics_to_file(filepath, format)
        if success:
            return {"status": "success", "filepath": filepath, "format": format}
        else:
            raise HTTPException(status_code=500, detail="Export failed")
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")


@router.get("/dashboard")
async def get_metrics_dashboard(
    hours: int = 24,
    metrics_service: ProductionMetricsService = Depends(get_metrics_service),
    health_service: ProductionHealthService = Depends(get_health_service)
) -> Dict[str, Any]:
    """Get comprehensive metrics dashboard data"""
    try:
        # Gather all metrics
        api_metrics = await metrics_service.get_api_metrics(hours=hours)
        ptaas_metrics = await metrics_service.get_ptaas_metrics(hours=hours)
        security_metrics = await metrics_service.get_security_metrics(hours=hours)
        health_status = await health_service.get_system_health()
        
        # Compile dashboard
        dashboard = {
            "overview": {
                "system_status": health_status.get("status", "unknown"),
                "total_api_requests": api_metrics.get("total_requests", 0),
                "api_error_rate": round(api_metrics.get("error_rate", 0), 2),
                "avg_response_time": round(api_metrics.get("average_response_time", 0) * 1000, 2),  # Convert to ms
                "total_scans": ptaas_metrics.get("total_scans", 0),
                "scan_success_rate": round(ptaas_metrics.get("success_rate", 0), 2),
                "security_events": security_metrics.get("total_security_events", 0),
                "failed_auth_rate": round(security_metrics.get("authentication", {}).get("failed_login_rate", 0), 2)
            },
            "detailed_metrics": {
                "api": api_metrics,
                "ptaas": ptaas_metrics,
                "security": security_metrics
            },
            "system_health": health_status,
            "timestamp": health_status.get("timestamp"),
            "timeframe_hours": hours
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")