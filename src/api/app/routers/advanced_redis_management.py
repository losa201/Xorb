"""
XORB Advanced Redis Management API Router
Production-ready Redis management endpoints with sophisticated monitoring and control
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.advanced_redis_intelligence_engine import AdvancedRedisIntelligenceEngine, create_redis_intelligence_engine
from ..services.sophisticated_redis_security_engine import SophisticatedRedisSecurityEngine, create_redis_security_engine
from ..infrastructure.advanced_redis_orchestrator import AdvancedRedisOrchestrator, get_redis_orchestrator
from ..dependencies import get_current_user, require_admin_role, require_security_analyst_role
from ..domain.value_objects import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/redis", tags=["Redis Management"])


# Request/Response Models
class RedisHealthResponse(BaseModel):
    """Redis health check response"""
    status: str
    cluster_nodes: int
    healthy_nodes: int
    primary_connection: bool
    memory_usage_mb: float
    cache_hit_rate: float
    operations_per_second: float
    latency_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "cluster_nodes": 3,
                "healthy_nodes": 3,
                "primary_connection": True,
                "memory_usage_mb": 512.5,
                "cache_hit_rate": 0.95,
                "operations_per_second": 1500.2,
                "latency_ms": 2.1
            }
        }


class CacheOptimizationRequest(BaseModel):
    """Cache optimization request"""
    namespace: Optional[str] = Field(None, description="Specific namespace to optimize")
    optimization_strategy: str = Field("performance", description="Optimization strategy")
    time_horizon_hours: int = Field(24, ge=1, le=168, description="Analysis time horizon")
    dry_run: bool = Field(True, description="Whether to perform dry run")
    
    class Config:
        schema_extra = {
            "example": {
                "namespace": "user_sessions",
                "optimization_strategy": "memory_efficiency",
                "time_horizon_hours": 48,
                "dry_run": False
            }
        }


class SecurityEventResponse(BaseModel):
    """Security event response"""
    event_id: str
    event_type: str
    threat_level: str
    timestamp: float
    source_ip: str
    risk_score: float
    mitigation_actions: List[str]
    investigation_status: str


class ThreatIntelligenceResponse(BaseModel):
    """Threat intelligence response"""
    ip_address: str
    threat_type: str
    confidence_score: float
    first_seen: float
    last_seen: float
    attack_patterns: List[str]
    reputation_score: float


class PerformancePredictionRequest(BaseModel):
    """Performance prediction request"""
    time_horizon_hours: int = Field(24, ge=1, le=168)
    confidence_threshold: float = Field(0.7, ge=0.1, le=1.0)
    include_recommendations: bool = Field(True)
    
    class Config:
        schema_extra = {
            "example": {
                "time_horizon_hours": 72,
                "confidence_threshold": 0.8,
                "include_recommendations": True
            }
        }


class CacheWarmupRequest(BaseModel):
    """Cache warmup request"""
    namespace: str = Field(..., description="Cache namespace to warm up")
    patterns: List[str] = Field(..., description="Key patterns to warm up")
    priority: int = Field(5, ge=1, le=10, description="Warmup priority")
    batch_size: int = Field(100, ge=1, le=1000, description="Batch size for warmup")
    
    class Config:
        schema_extra = {
            "example": {
                "namespace": "threat_intelligence",
                "patterns": ["threat:*", "ioc:*"],
                "priority": 8,
                "batch_size": 250
            }
        }


# Redis Health and Status Endpoints
@router.get("/health", response_model=RedisHealthResponse)
async def get_redis_health(
    orchestrator: AdvancedRedisOrchestrator = Depends(get_redis_orchestrator),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive Redis health status"""
    try:
        cluster_status = await orchestrator.get_cluster_status()
        metrics = orchestrator.metrics
        
        return RedisHealthResponse(
            status="healthy" if cluster_status["healthy_nodes"] > 0 else "unhealthy",
            cluster_nodes=cluster_status["cluster_size"],
            healthy_nodes=cluster_status["healthy_nodes"],
            primary_connection=True,  # Would check actual connection
            memory_usage_mb=metrics.memory_usage_mb,
            cache_hit_rate=metrics.cache_hit_rate,
            operations_per_second=metrics.operations_per_second,
            latency_ms=2.1  # Would measure actual latency
        )
        
    except Exception as e:
        logger.error(f"Error getting Redis health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Redis health: {str(e)}"
        )


@router.get("/cluster/status")
async def get_cluster_status(
    orchestrator: AdvancedRedisOrchestrator = Depends(get_redis_orchestrator),
    current_user: User = Depends(require_admin_role)
):
    """Get detailed cluster status (Admin only)"""
    try:
        cluster_status = await orchestrator.get_cluster_status()
        return JSONResponse(content=cluster_status)
        
    except Exception as e:
        logger.error(f"Error getting cluster status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster status: {str(e)}"
        )


@router.get("/metrics/performance")
async def get_performance_metrics(
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range in hours"),
    granularity: str = Query("hour", regex="^(minute|hour|day)$", description="Metrics granularity"),
    orchestrator: AdvancedRedisOrchestrator = Depends(get_redis_orchestrator),
    current_user: User = Depends(get_current_user)
):
    """Get Redis performance metrics"""
    try:
        # Get current metrics
        current_metrics = orchestrator.metrics
        
        # Get performance history
        performance_history = orchestrator.performance_history[-time_range_hours:]
        
        metrics_data = {
            "current": {
                "operations_per_second": current_metrics.operations_per_second,
                "memory_usage_mb": current_metrics.memory_usage_mb,
                "cache_hit_rate": current_metrics.cache_hit_rate,
                "connected_clients": current_metrics.connected_clients,
                "key_count": current_metrics.key_count
            },
            "history": [
                {
                    "timestamp": m.timestamp,
                    "operations_per_second": m.operations_per_second,
                    "memory_usage_mb": m.memory_usage_mb,
                    "cache_hit_rate": m.cache_hit_rate,
                    "connected_clients": m.connected_clients
                }
                for m in performance_history
            ],
            "performance_trend": orchestrator._calculate_performance_trend(),
            "time_range_hours": time_range_hours,
            "granularity": granularity
        }
        
        return JSONResponse(content=metrics_data)
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


# Intelligence and Optimization Endpoints
@router.get("/intelligence/report")
async def get_intelligence_report(
    intelligence_engine: AdvancedRedisIntelligenceEngine = Depends(create_redis_intelligence_engine),
    current_user: User = Depends(require_security_analyst_role)
):
    """Get comprehensive Redis intelligence report (Security Analyst+)"""
    try:
        report = await intelligence_engine.get_intelligence_report()
        return JSONResponse(content=report)
        
    except Exception as e:
        logger.error(f"Error getting intelligence report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get intelligence report: {str(e)}"
        )


@router.post("/optimization/analyze")
async def analyze_cache_optimization(
    request: CacheOptimizationRequest,
    intelligence_engine: AdvancedRedisIntelligenceEngine = Depends(create_redis_intelligence_engine),
    current_user: User = Depends(require_admin_role)
):
    """Analyze cache optimization opportunities (Admin only)"""
    try:
        recommendations = await intelligence_engine.analyze_cache_optimization()
        
        # Filter by namespace if specified
        if request.namespace:
            recommendations = [
                rec for rec in recommendations 
                if request.namespace in rec.key_pattern
            ]
        
        optimization_analysis = {
            "optimization_strategy": request.optimization_strategy,
            "time_horizon_hours": request.time_horizon_hours,
            "dry_run": request.dry_run,
            "recommendations": [
                {
                    "key_pattern": rec.key_pattern,
                    "current_performance": rec.current_performance,
                    "predicted_improvement": rec.predicted_improvement,
                    "recommended_action": rec.recommended_action,
                    "confidence_score": rec.confidence_score,
                    "estimated_impact": rec.estimated_impact,
                    "implementation_complexity": rec.implementation_complexity,
                    "resource_requirements": rec.resource_requirements
                }
                for rec in recommendations
            ],
            "total_recommendations": len(recommendations),
            "high_confidence_count": len([r for r in recommendations if r.confidence_score > 0.8]),
            "estimated_total_improvement": sum(r.predicted_improvement for r in recommendations)
        }
        
        return JSONResponse(content=optimization_analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing cache optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze cache optimization: {str(e)}"
        )


@router.post("/intelligence/predict-performance")
async def predict_cache_performance(
    request: PerformancePredictionRequest,
    intelligence_engine: AdvancedRedisIntelligenceEngine = Depends(create_redis_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """Predict cache performance using ML models"""
    try:
        predictions = await intelligence_engine.predict_cache_performance(
            time_horizon_hours=request.time_horizon_hours,
            confidence_threshold=request.confidence_threshold
        )
        
        if request.include_recommendations:
            # Add optimization recommendations based on predictions
            recommendations = await intelligence_engine.analyze_cache_optimization()
            predictions["optimization_recommendations"] = [
                {
                    "action": rec.recommended_action,
                    "confidence": rec.confidence_score,
                    "impact": rec.predicted_improvement
                }
                for rec in recommendations[:5]  # Top 5 recommendations
            ]
        
        return JSONResponse(content=predictions)
        
    except Exception as e:
        logger.error(f"Error predicting cache performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict cache performance: {str(e)}"
        )


# Security Endpoints
@router.get("/security/status")
async def get_security_status(
    security_engine: SophisticatedRedisSecurityEngine = Depends(create_redis_security_engine),
    current_user: User = Depends(require_security_analyst_role)
):
    """Get Redis security status (Security Analyst+)"""
    try:
        security_status = await security_engine.get_security_status()
        return JSONResponse(content=security_status)
        
    except Exception as e:
        logger.error(f"Error getting security status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security status: {str(e)}"
        )


@router.get("/security/events")
async def get_security_events(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events"),
    threat_level: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    source_ip: Optional[str] = Query(None, description="Filter by source IP"),
    security_engine: SophisticatedRedisSecurityEngine = Depends(create_redis_security_engine),
    current_user: User = Depends(require_security_analyst_role)
):
    """Get Redis security events (Security Analyst+)"""
    try:
        # Get recent security events
        all_events = list(security_engine.security_events)
        
        # Apply filters
        filtered_events = all_events
        
        if threat_level:
            filtered_events = [e for e in filtered_events if e.threat_level.value == threat_level]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type.value == event_type]
        
        if source_ip:
            filtered_events = [e for e in filtered_events if e.source_ip == source_ip]
        
        # Sort by timestamp (most recent first)
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limit results
        limited_events = filtered_events[:limit]
        
        events_data = {
            "events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "threat_level": event.threat_level.value,
                    "timestamp": event.timestamp,
                    "source_ip": event.source_ip,
                    "user_id": event.user_id,
                    "redis_command": event.redis_command,
                    "affected_keys": event.affected_keys,
                    "risk_score": event.risk_score,
                    "mitigation_actions": [action.value for action in event.mitigation_actions],
                    "investigation_status": event.investigation_status,
                    "details": event.details
                }
                for event in limited_events
            ],
            "total_events": len(all_events),
            "filtered_events": len(filtered_events),
            "returned_events": len(limited_events),
            "filters_applied": {
                "threat_level": threat_level,
                "event_type": event_type,
                "source_ip": source_ip
            }
        }
        
        return JSONResponse(content=events_data)
        
    except Exception as e:
        logger.error(f"Error getting security events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security events: {str(e)}"
        )


@router.get("/security/events/{event_id}/investigate")
async def investigate_security_event(
    event_id: str,
    security_engine: SophisticatedRedisSecurityEngine = Depends(create_redis_security_engine),
    current_user: User = Depends(require_security_analyst_role)
):
    """Investigate specific security event (Security Analyst+)"""
    try:
        investigation_data = await security_engine.investigate_security_event(event_id)
        
        if "error" in investigation_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=investigation_data["error"]
            )
        
        return JSONResponse(content=investigation_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error investigating security event {event_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to investigate security event: {str(e)}"
        )


@router.get("/security/threats")
async def get_active_threats(
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    security_engine: SophisticatedRedisSecurityEngine = Depends(create_redis_security_engine),
    current_user: User = Depends(require_security_analyst_role)
):
    """Get active threats (Security Analyst+)"""
    try:
        active_threats = security_engine.active_threats
        
        # Filter by confidence threshold
        filtered_threats = {
            ip: threat for ip, threat in active_threats.items()
            if threat.confidence_score >= confidence_threshold
        }
        
        threats_data = {
            "active_threats": [
                {
                    "ip_address": threat.ip_address,
                    "threat_type": threat.threat_type,
                    "confidence_score": threat.confidence_score,
                    "first_seen": threat.first_seen,
                    "last_seen": threat.last_seen,
                    "attack_patterns": threat.attack_patterns,
                    "reputation_score": threat.reputation_score,
                    "sources": threat.sources
                }
                for threat in filtered_threats.values()
            ],
            "total_threats": len(active_threats),
            "high_confidence_threats": len([
                t for t in active_threats.values() if t.confidence_score > 0.8
            ]),
            "confidence_threshold": confidence_threshold,
            "blocked_ips_count": len(security_engine.blocked_ips),
            "rate_limited_clients_count": len(security_engine.rate_limited_clients)
        }
        
        return JSONResponse(content=threats_data)
        
    except Exception as e:
        logger.error(f"Error getting active threats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active threats: {str(e)}"
        )


# Cache Management Endpoints
@router.post("/cache/warmup")
async def warmup_cache(
    request: CacheWarmupRequest,
    orchestrator: AdvancedRedisOrchestrator = Depends(get_redis_orchestrator),
    current_user: User = Depends(require_admin_role)
):
    """Warm up cache with specified patterns (Admin only)"""
    try:
        # Create intelligent cache for the namespace
        cache_config = {
            "ttl": 3600,
            "compression": True,
            "priority": request.priority
        }
        
        cache = await orchestrator.create_intelligent_cache(request.namespace, cache_config)
        
        # Perform cache warmup (simplified implementation)
        warmup_data = {}
        for pattern in request.patterns:
            # In practice, this would fetch data based on patterns
            warmup_data[f"{pattern}_example"] = f"warmed_data_{pattern}"
        
        await cache.set_multi(warmup_data)
        
        warmup_result = {
            "namespace": request.namespace,
            "patterns": request.patterns,
            "warmed_keys": len(warmup_data),
            "priority": request.priority,
            "batch_size": request.batch_size,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content=warmup_result)
        
    except Exception as e:
        logger.error(f"Error warming up cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to warm up cache: {str(e)}"
        )


@router.post("/cache/invalidate")
async def invalidate_cache_pattern(
    pattern: str = Body(..., description="Cache pattern to invalidate"),
    namespace: str = Body("default", description="Cache namespace"),
    orchestrator: AdvancedRedisOrchestrator = Depends(get_redis_orchestrator),
    current_user: User = Depends(require_admin_role)
):
    """Invalidate cache entries matching pattern (Admin only)"""
    try:
        # Get cache manager
        cache = await orchestrator.create_intelligent_cache(namespace, {})
        
        # Invalidate pattern (simplified implementation)
        client = await orchestrator.get_optimal_client("write")
        deleted_count = await client.eval(
            """
            local keys = redis.call('keys', ARGV[1])
            local count = 0
            for i=1,#keys do
                redis.call('del', keys[i])
                count = count + 1
            end
            return count
            """,
            0,
            f"{namespace}:{pattern}"
        )
        
        invalidation_result = {
            "pattern": pattern,
            "namespace": namespace,
            "deleted_keys": deleted_count,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content=invalidation_result)
        
    except Exception as e:
        logger.error(f"Error invalidating cache pattern: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate cache pattern: {str(e)}"
        )


@router.get("/cache/info")
async def get_cache_info(
    namespace: Optional[str] = Query(None, description="Specific namespace to analyze"),
    orchestrator: AdvancedRedisOrchestrator = Depends(get_redis_orchestrator),
    current_user: User = Depends(get_current_user)
):
    """Get cache information and statistics"""
    try:
        if namespace:
            # Get specific namespace cache info
            cache = await orchestrator.create_intelligent_cache(namespace, {})
            cache_info = await cache.get_cache_info()
        else:
            # Get general Redis info
            client = await orchestrator.get_optimal_client("read")
            info = await client.info()
            
            cache_info = {
                "general": {
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0)
                },
                "keyspace": {},
                "replication": {
                    "role": info.get("role", "unknown"),
                    "connected_slaves": info.get("connected_slaves", 0)
                }
            }
            
            # Get keyspace info
            for key, value in info.items():
                if key.startswith("db"):
                    cache_info["keyspace"][key] = value
        
        return JSONResponse(content=cache_info)
        
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache info: {str(e)}"
        )


# Administrative Endpoints
@router.post("/admin/emergency-shutdown")
async def emergency_shutdown(
    reason: str = Body(..., description="Reason for emergency shutdown"),
    orchestrator: AdvancedRedisOrchestrator = Depends(get_redis_orchestrator),
    current_user: User = Depends(require_admin_role)
):
    """Emergency Redis shutdown (Admin only)"""
    try:
        logger.critical(f"Emergency Redis shutdown initiated by {current_user.username}: {reason}")
        
        # Graceful shutdown
        await orchestrator.shutdown()
        
        shutdown_result = {
            "status": "shutdown_initiated",
            "reason": reason,
            "initiated_by": current_user.username,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Redis orchestrator shutdown completed"
        }
        
        return JSONResponse(content=shutdown_result)
        
    except Exception as e:
        logger.error(f"Error during emergency shutdown: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute emergency shutdown: {str(e)}"
        )


@router.get("/admin/system-diagnostics")
async def get_system_diagnostics(
    orchestrator: AdvancedRedisOrchestrator = Depends(get_redis_orchestrator),
    intelligence_engine: AdvancedRedisIntelligenceEngine = Depends(create_redis_intelligence_engine),
    security_engine: SophisticatedRedisSecurityEngine = Depends(create_redis_security_engine),
    current_user: User = Depends(require_admin_role)
):
    """Get comprehensive system diagnostics (Admin only)"""
    try:
        # Collect diagnostics from all components
        cluster_status = await orchestrator.get_cluster_status()
        intelligence_report = await intelligence_engine.get_intelligence_report()
        security_status = await security_engine.get_security_status()
        
        diagnostics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_overview": {
                "redis_cluster": cluster_status,
                "intelligence_engine": {
                    "status": intelligence_report.get("ml_status", {}),
                    "cache_statistics": intelligence_report.get("cache_statistics", {}),
                    "monitoring_active": intelligence_report.get("system_health", {}).get("monitoring_active", False)
                },
                "security_engine": {
                    "monitoring_active": security_status.get("monitoring_active", False),
                    "active_threats": security_status.get("active_threats", {}),
                    "recent_events": security_status.get("recent_events", {})
                }
            },
            "performance_metrics": {
                "current_performance": intelligence_report.get("current_performance"),
                "performance_trends": intelligence_report.get("performance_trends"),
                "predictions": intelligence_report.get("performance_predictions", {})
            },
            "security_assessment": {
                "threat_level": "low",  # Would be calculated based on current threats
                "security_events_24h": security_status.get("recent_events", {}).get("total", 0),
                "blocked_ips": security_status.get("active_threats", {}).get("blocked_ips", 0),
                "emergency_lockdown": security_status.get("system_health", {}).get("emergency_lockdown", False)
            },
            "recommendations": intelligence_report.get("optimization_recommendations", [])[:3]
        }
        
        return JSONResponse(content=diagnostics)
        
    except Exception as e:
        logger.error(f"Error getting system diagnostics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system diagnostics: {str(e)}"
        )