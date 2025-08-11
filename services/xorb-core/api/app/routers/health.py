"""Health check and system status endpoints."""
from typing import Dict, Any
import asyncio
import time
from datetime import datetime

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from ..auth.dependencies import require_super_admin
from ..auth.models import UserClaims
from ..infrastructure.database import get_database_stats, check_database_connection
from ..infrastructure.performance import get_cache_manager
from ..jobs.service import JobService
import redis.asyncio as redis
import os
import structlog

logger = structlog.get_logger("health_api")
router = APIRouter(tags=["Health"])

# Initialize services for health checks
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url)


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "service": "xorb-api"
    }


@router.get("/readiness")
async def readiness_check():
    """Comprehensive readiness check for all dependencies."""
    start_time = time.time()
    health_details = {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Database connectivity check
    try:
        db_healthy = await check_database_connection()
        health_details["checks"]["database"] = {
            "status": "ok" if db_healthy else "error",
            "message": "Database connection successful" if db_healthy else "Database connection failed"
        }
        if not db_healthy:
            health_details["status"] = "not_ready"
    except Exception as e:
        health_details["checks"]["database"] = {
            "status": "error",
            "message": f"Database check failed: {str(e)}"
        }
        health_details["status"] = "not_ready"
    
    # Redis connectivity check
    try:
        pong = await redis_client.ping()
        health_details["checks"]["redis"] = {
            "status": "ok" if pong else "error",
            "message": "Redis connection successful" if pong else "Redis connection failed"
        }
        if not pong:
            health_details["status"] = "not_ready"
    except Exception as e:
        health_details["checks"]["redis"] = {
            "status": "error",
            "message": f"Redis check failed: {str(e)}"
        }
        health_details["status"] = "not_ready"
    
    # Vector store check (pgvector extension)
    try:
        from ..infrastructure.vector_store import get_vector_store
        vector_store = get_vector_store()
        # Simple check - we'll assume it's healthy if we can instantiate it
        health_details["checks"]["vector_store"] = {
            "status": "ok",
            "message": "Vector store initialized"
        }
    except Exception as e:
        health_details["checks"]["vector_store"] = {
            "status": "error",
            "message": f"Vector store check failed: {str(e)}"
        }
        health_details["status"] = "not_ready"
    
    # Check duration
    health_details["check_duration_ms"] = round((time.time() - start_time) * 1000, 2)
    
    # Return appropriate status code
    status_code = status.HTTP_200_OK if health_details["status"] == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(content=health_details, status_code=status_code)


@router.get("/status")
async def system_status(
    current_user: UserClaims = Depends(require_super_admin)
):
    """Detailed system status for administrators."""
    
    status_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time(),  # Would need to track actual uptime
        "components": {}
    }
    
    # Database statistics
    try:
        db_stats = await get_database_stats()
        status_info["components"]["database"] = {
            "status": "healthy",
            "pool_stats": db_stats,
            "connection_test": await check_database_connection()
        }
    except Exception as e:
        status_info["components"]["database"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Redis statistics
    try:
        redis_info = await redis_client.info()
        status_info["components"]["redis"] = {
            "status": "healthy",
            "connected_clients": redis_info.get("connected_clients", 0),
            "used_memory": redis_info.get("used_memory", 0),
            "total_commands_processed": redis_info.get("total_commands_processed", 0)
        }
    except Exception as e:
        status_info["components"]["redis"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Job queue statistics
    try:
        job_service = JobService(redis_client)
        queue_stats = await job_service.get_queue_stats("default")
        worker_stats = await job_service.get_worker_stats()
        
        status_info["components"]["job_system"] = {
            "status": "healthy",
            "queue_stats": queue_stats,
            "active_workers": len(worker_stats),
            "worker_stats": worker_stats
        }
    except Exception as e:
        status_info["components"]["job_system"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Application cache statistics
    try:
        cache_manager = get_cache_manager()
        status_info["components"]["cache"] = {
            "status": "healthy",
            "cache_size": cache_manager.size()
        }
    except Exception as e:
        status_info["components"]["cache"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Memory and performance metrics
    try:
        from ..infrastructure.performance import MemoryMonitor
        memory_stats = MemoryMonitor.get_memory_usage()
        status_info["components"]["memory"] = {
            "status": "healthy",
            **memory_stats
        }
    except Exception as e:
        status_info["components"]["memory"] = {
            "status": "error",
            "error": str(e)
        }
    
    return status_info


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"error": "Prometheus client not available"}


@router.get("/version")
async def version_info():
    """Version and build information."""
    return {
        "version": "2.0.0",
        "build_date": "2024-08-09",
        "commit_hash": os.getenv("GIT_COMMIT", "unknown"),
        "build_number": os.getenv("BUILD_NUMBER", "unknown"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "python_version": "3.11",
        "features": {
            "authentication": "OIDC",
            "multi_tenancy": "RLS",
            "storage": "FS+S3",
            "job_orchestration": "Redis",
            "vector_search": "pgvector",
            "observability": "OpenTelemetry",
            "rate_limiting": "Redis+SlidingWindow",
            "caching": "Redis+InMemory"
        }
    }


@router.post("/warm-up")
async def warm_up_services(
    current_user: UserClaims = Depends(require_super_admin)
):
    """Warm up all services and caches."""
    warm_up_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "operations": []
    }
    
    # Warm up database connections
    try:
        await get_database_stats()
        warm_up_results["operations"].append({
            "service": "database",
            "operation": "connection_pool_warmup",
            "status": "success"
        })
    except Exception as e:
        warm_up_results["operations"].append({
            "service": "database", 
            "operation": "connection_pool_warmup",
            "status": "error",
            "error": str(e)
        })
    
    # Warm up Redis connections
    try:
        await redis_client.ping()
        warm_up_results["operations"].append({
            "service": "redis",
            "operation": "connection_test",
            "status": "success"
        })
    except Exception as e:
        warm_up_results["operations"].append({
            "service": "redis",
            "operation": "connection_test", 
            "status": "error",
            "error": str(e)
        })
    
    # Initialize vector store
    try:
        from ..infrastructure.vector_store import get_vector_store
        get_vector_store()
        warm_up_results["operations"].append({
            "service": "vector_store",
            "operation": "initialization",
            "status": "success"
        })
    except Exception as e:
        warm_up_results["operations"].append({
            "service": "vector_store",
            "operation": "initialization",
            "status": "error", 
            "error": str(e)
        })
    
    logger.info(
        "Service warm-up completed",
        operations_count=len(warm_up_results["operations"]),
        user_id=current_user.sub
    )
    
    return warm_up_results


@router.get("/dependencies")
async def dependency_health():
    """Check health of all external dependencies."""
    dependencies = {
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": {}
    }
    
    # Database dependency
    try:
        is_healthy = await check_database_connection()
        dependencies["dependencies"]["postgresql"] = {
            "status": "healthy" if is_healthy else "unhealthy",
            "type": "database",
            "critical": True
        }
    except Exception as e:
        dependencies["dependencies"]["postgresql"] = {
            "status": "error",
            "type": "database", 
            "critical": True,
            "error": str(e)
        }
    
    # Redis dependency
    try:
        pong = await redis_client.ping()
        dependencies["dependencies"]["redis"] = {
            "status": "healthy" if pong else "unhealthy",
            "type": "cache",
            "critical": True
        }
    except Exception as e:
        dependencies["dependencies"]["redis"] = {
            "status": "error",
            "type": "cache",
            "critical": True,
            "error": str(e)
        }
    
    # OIDC Provider (would check if configured)
    oidc_endpoint = os.getenv("OIDC_ISSUER")
    if oidc_endpoint:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{oidc_endpoint}/.well-known/openid-configuration", timeout=5)
                dependencies["dependencies"]["oidc_provider"] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "type": "authentication",
                    "critical": False,
                    "endpoint": oidc_endpoint
                }
        except Exception as e:
            dependencies["dependencies"]["oidc_provider"] = {
                "status": "error",
                "type": "authentication",
                "critical": False,
                "error": str(e)
            }
    
    return dependencies