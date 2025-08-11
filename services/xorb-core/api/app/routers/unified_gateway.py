"""Unified API Gateway for all XORB Platform Services"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from datetime import datetime
import logging

from ..auth.dependencies import require_auth, UserClaims
# Remove tenant context dependency for now - will be handled by middleware
from ..infrastructure.service_orchestrator import get_service_orchestrator, ServiceType
from ..infrastructure.observability import add_trace_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/platform", tags=["Platform Gateway"])


@router.get("/services", summary="List all platform services")
async def list_services(
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """List all registered platform services with their current status"""
    try:
        service_status = orchestrator.get_all_service_status()
        
        services = []
        for service_id, instance in service_status.items():
            services.append({
                "service_id": service_id,
                "name": instance.definition.name,
                "type": instance.definition.service_type.value,
                "status": instance.status.value,
                "uptime_seconds": (datetime.utcnow() - instance.start_time).total_seconds() if instance.start_time else 0,
                "restart_count": instance.restart_count,
                "last_health_check": instance.last_health_check.isoformat() if instance.last_health_check else None
            })
        
        return {
            "services": services,
            "total_count": len(services),
            "by_type": {
                stype.value: len([s for s in services if s["type"] == stype.value])
                for stype in ServiceType
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve service list")


@router.get("/services/{service_id}/status", summary="Get service status")
async def get_service_status(
    service_id: str,
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Get detailed status information for a specific service"""
    try:
        instance = orchestrator.get_service_status(service_id)
        
        if not instance:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")
        
        # Get health check
        health_result = await orchestrator.health_check(service_id)
        
        # Get metrics
        metrics = await orchestrator.get_service_metrics(service_id)
        
        return {
            "service_id": service_id,
            "name": instance.definition.name,
            "type": instance.definition.service_type.value,
            "status": instance.status.value,
            "instance_id": instance.instance_id,
            "start_time": instance.start_time.isoformat() if instance.start_time else None,
            "restart_count": instance.restart_count,
            "error_message": instance.error_message,
            "health": health_result,
            "metrics": metrics,
            "dependencies": instance.definition.dependencies
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service status for {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve service status")


@router.post("/services/{service_id}/start", summary="Start a service")
async def start_service(
    service_id: str,
    background_tasks: BackgroundTasks,
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Start a specific service"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Start service in background
        def start_task():
            import asyncio
            asyncio.create_task(orchestrator.start_service(service_id))
        
        background_tasks.add_task(start_task)
        
        add_trace_context(
            operation="service_start_request",
            service_id=service_id,
            requested_by=user.user_id
        )
        
        return {"message": f"Starting service {service_id}", "service_id": service_id}
        
    except Exception as e:
        logger.error(f"Failed to start service {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start service")


@router.post("/services/{service_id}/stop", summary="Stop a service")
async def stop_service(
    service_id: str,
    background_tasks: BackgroundTasks,
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Stop a specific service"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Stop service in background
        def stop_task():
            import asyncio
            asyncio.create_task(orchestrator.stop_service(service_id))
        
        background_tasks.add_task(stop_task)
        
        add_trace_context(
            operation="service_stop_request", 
            service_id=service_id,
            requested_by=user.user_id
        )
        
        return {"message": f"Stopping service {service_id}", "service_id": service_id}
        
    except Exception as e:
        logger.error(f"Failed to stop service {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop service")


@router.post("/services/{service_id}/restart", summary="Restart a service")
async def restart_service(
    service_id: str,
    background_tasks: BackgroundTasks,
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Restart a specific service"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Restart service in background
        def restart_task():
            import asyncio
            asyncio.create_task(orchestrator.restart_service(service_id))
        
        background_tasks.add_task(restart_task)
        
        add_trace_context(
            operation="service_restart_request",
            service_id=service_id, 
            requested_by=user.user_id
        )
        
        return {"message": f"Restarting service {service_id}", "service_id": service_id}
        
    except Exception as e:
        logger.error(f"Failed to restart service {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to restart service")


# Analytics Service Endpoints
@router.post("/analytics/behavioral/profile", summary="Create behavioral profile")
async def create_behavioral_profile(
    profile_data: Dict[str, Any],
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Create a new behavioral analytics profile"""
    try:
        # Get behavioral analytics service
        instance = orchestrator.get_service_status("behavioral_analytics")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Behavioral analytics service not available")
        
        service = instance.instance_object
        
        # Create profile
        profile_id = profile_data.get("profile_id")
        profile_type = profile_data.get("profile_type", "user")
        
        profile = service.create_profile(profile_id, profile_type)
        
        return {
            "message": "Behavioral profile created",
            "profile": profile.get_profile_summary()
        }
        
    except Exception as e:
        logger.error(f"Failed to create behavioral profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to create behavioral profile")


@router.post("/analytics/behavioral/update", summary="Update behavioral profile")
async def update_behavioral_profile(
    update_data: Dict[str, Any],
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Update behavioral profile with new features"""
    try:
        # Get behavioral analytics service
        instance = orchestrator.get_service_status("behavioral_analytics")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Behavioral analytics service not available")
        
        service = instance.instance_object
        
        profile_id = update_data.get("profile_id")
        features = update_data.get("features", {})
        
        result = service.update_profile(profile_id, features)
        
        return {
            "message": "Behavioral profile updated",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Failed to update behavioral profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update behavioral profile")


@router.get("/analytics/behavioral/dashboard", summary="Get behavioral analytics dashboard")
async def get_behavioral_dashboard(
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Get behavioral analytics risk dashboard"""
    try:
        # Get behavioral analytics service
        instance = orchestrator.get_service_status("behavioral_analytics")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Behavioral analytics service not available")
        
        service = instance.instance_object
        dashboard = service.get_risk_dashboard()
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Failed to get behavioral dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve behavioral dashboard")


# Threat Hunting Service Endpoints
@router.post("/threat-hunting/query", summary="Execute threat hunting query")
async def execute_threat_hunting_query(
    query_data: Dict[str, Any],
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Execute a threat hunting query"""
    try:
        # Get threat hunting service
        instance = orchestrator.get_service_status("threat_hunting")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Threat hunting service not available")
        
        service = instance.instance_object
        
        query = query_data.get("query")
        data_source = query_data.get("data_source", "default")
        time_range = query_data.get("time_range")
        
        result = service.execute_query(query, data_source, time_range)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to execute threat hunting query: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute query")


@router.get("/threat-hunting/queries", summary="List saved queries")
async def list_saved_queries(
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """List all saved threat hunting queries"""
    try:
        # Get threat hunting service
        instance = orchestrator.get_service_status("threat_hunting")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Threat hunting service not available")
        
        service = instance.instance_object
        queries = service.list_saved_queries()
        
        return {"queries": queries}
        
    except Exception as e:
        logger.error(f"Failed to list saved queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve saved queries")


@router.post("/threat-hunting/queries", summary="Save threat hunting query")
async def save_threat_hunting_query(
    query_data: Dict[str, Any],
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Save a threat hunting query"""
    try:
        # Get threat hunting service
        instance = orchestrator.get_service_status("threat_hunting")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Threat hunting service not available")
        
        service = instance.instance_object
        
        name = query_data.get("name")
        query = query_data.get("query")
        description = query_data.get("description", "")
        
        service.add_saved_query(name, query, description)
        
        return {"message": f"Query '{name}' saved successfully"}
        
    except Exception as e:
        logger.error(f"Failed to save query: {e}")
        raise HTTPException(status_code=500, detail="Failed to save query")


# Forensics Service Endpoints
@router.post("/forensics/evidence", summary="Collect forensic evidence")
async def collect_forensic_evidence(
    evidence_data: Dict[str, Any],
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Collect new forensic evidence"""
    try:
        # Get forensics service
        instance = orchestrator.get_service_status("forensics")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Forensics service not available")
        
        service = instance.instance_object
        
        # Import required classes
        from ptaas.forensics_engine import EvidenceMetadata
        
        # Create metadata
        metadata = EvidenceMetadata(
            case_id=evidence_data.get("case_id"),
            evidence_type=evidence_data.get("evidence_type"),
            source=evidence_data.get("source"),
            collection_method=evidence_data.get("collection_method"),
            collector=user.user_id
        )
        
        data = evidence_data.get("data", {})
        evidence_id = service.collect_evidence(metadata, data)
        
        return {
            "message": "Evidence collected successfully",
            "evidence_id": evidence_id
        }
        
    except Exception as e:
        logger.error(f"Failed to collect evidence: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect evidence")


@router.get("/forensics/evidence/{evidence_id}", summary="Get forensic evidence")
async def get_forensic_evidence(
    evidence_id: str,
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Retrieve forensic evidence by ID"""
    try:
        # Get forensics service
        instance = orchestrator.get_service_status("forensics")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Forensics service not available")
        
        service = instance.instance_object
        
        evidence = service.get_evidence(evidence_id)
        if not evidence:
            raise HTTPException(status_code=404, detail="Evidence not found")
        
        # Get chain of custody
        chain = service.get_chain_of_custody(evidence_id)
        
        return {
            "evidence": evidence,
            "chain_of_custody": chain
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evidence {evidence_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve evidence")


@router.post("/forensics/evidence/{evidence_id}/chain", summary="Add to chain of custody")
async def add_to_chain_of_custody(
    evidence_id: str,
    chain_data: Dict[str, Any],
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Add entry to chain of custody"""
    try:
        # Get forensics service
        instance = orchestrator.get_service_status("forensics")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Forensics service not available")
        
        service = instance.instance_object
        
        # Import required class
        from ptaas.forensics_engine import ChainOfCustodyEntry
        
        # Create chain entry
        entry = ChainOfCustodyEntry(
            evidence_id=evidence_id,
            action=chain_data.get("action"),
            actor=user.user_id,
            location=chain_data.get("location"),
            next_custodian=chain_data.get("next_custodian")
        )
        
        success = service.add_to_chain_of_custody(evidence_id, entry)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add chain entry")
        
        return {"message": "Chain of custody entry added successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add chain entry: {e}")
        raise HTTPException(status_code=500, detail="Failed to add chain entry")


# Network Security Service Endpoints
@router.post("/network/segments", summary="Create network segment")
async def create_network_segment(
    segment_data: Dict[str, Any],
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Create a new network segment"""
    try:
        # Get network microsegmentation service
        instance = orchestrator.get_service_status("network_microsegmentation")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Network microsegmentation service not available")
        
        service = instance.instance_object
        
        # Import required class
        from ptaas.network_microsegmentation import NetworkSegment
        
        # Create segment
        segment = NetworkSegment(
            segment_id=segment_data.get("segment_id"),
            name=segment_data.get("name"),
            description=segment_data.get("description", ""),
            assets=segment_data.get("assets", []),
            policies=[],
            metadata=segment_data.get("metadata", {})
        )
        
        service.create_segment(segment)
        
        return {"message": f"Network segment '{segment.name}' created successfully"}
        
    except Exception as e:
        logger.error(f"Failed to create network segment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create network segment")


@router.post("/network/segments/{segment_id}/evaluate", summary="Evaluate network access")
async def evaluate_network_access(
    segment_id: str,
    access_data: Dict[str, Any],
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Evaluate network access request"""
    try:
        # Get network microsegmentation service
        instance = orchestrator.get_service_status("network_microsegmentation")
        if not instance or not instance.instance_object:
            raise HTTPException(status_code=503, detail="Network microsegmentation service not available")
        
        service = instance.instance_object
        
        context = access_data.get("context", {})
        result = service.evaluate_access(segment_id, context)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to evaluate network access: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate network access")


# Platform Health Endpoint
@router.get("/health", summary="Platform health check")
async def platform_health(
    orchestrator = Depends(get_service_orchestrator)
):
    """Get overall platform health status"""
    try:
        service_status = orchestrator.get_all_service_status()
        
        health_summary = {
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        unhealthy_count = 0
        
        for service_id, instance in service_status.items():
            health_result = await orchestrator.health_check(service_id)
            
            health_summary["services"][service_id] = {
                "status": instance.status.value,
                "healthy": health_result.get("healthy", False),
                "uptime_seconds": (datetime.utcnow() - instance.start_time).total_seconds() if instance.start_time else 0
            }
            
            if not health_result.get("healthy", False):
                unhealthy_count += 1
        
        # Determine overall status
        if unhealthy_count == 0:
            health_summary["overall_status"] = "healthy"
        elif unhealthy_count < len(service_status) / 2:
            health_summary["overall_status"] = "degraded"
        else:
            health_summary["overall_status"] = "unhealthy"
        
        health_summary["total_services"] = len(service_status)
        health_summary["unhealthy_services"] = unhealthy_count
        
        return health_summary
        
    except Exception as e:
        logger.error(f"Failed to get platform health: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# Platform Metrics Endpoint
@router.get("/metrics", summary="Platform metrics")
async def platform_metrics(
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Get platform-wide metrics"""
    try:
        service_status = orchestrator.get_all_service_status()
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "platform": {
                "total_services": len(service_status),
                "running_services": len([s for s in service_status.values() if s.status.value == "running"]),
                "service_types": {
                    stype.value: len([
                        s for s in service_status.values() 
                        if s.definition.service_type == stype
                    ])
                    for stype in ServiceType
                }
            },
            "services": {}
        }
        
        # Get metrics for each service
        for service_id, instance in service_status.items():
            service_metrics = await orchestrator.get_service_metrics(service_id)
            metrics["services"][service_id] = service_metrics
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get platform metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve platform metrics")


# Platform Dashboard Endpoint
@router.get("/dashboard", summary="Platform dashboard")
async def platform_dashboard(
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Get comprehensive platform dashboard data"""
    try:
        service_status = orchestrator.get_all_service_status()
        
        # Calculate service health summary
        healthy_services = 0
        total_uptime = 0
        total_restarts = 0
        
        service_health = {}
        for service_id, instance in service_status.items():
            health_result = await orchestrator.health_check(service_id)
            service_health[service_id] = health_result
            
            if health_result.get("healthy", False):
                healthy_services += 1
            
            if instance.start_time:
                uptime = (datetime.utcnow() - instance.start_time).total_seconds()
                total_uptime += uptime
            
            total_restarts += instance.restart_count
        
        # Service type distribution
        service_by_type = {}
        for stype in ServiceType:
            services_of_type = [
                s for s in service_status.values() 
                if s.definition.service_type == stype
            ]
            service_by_type[stype.value] = {
                "count": len(services_of_type),
                "running": len([s for s in services_of_type if s.status.value == "running"]),
                "services": [s.service_id for s in services_of_type]
            }
        
        # Recent activity (services started/restarted recently)
        recent_activity = []
        for service_id, instance in service_status.items():
            if instance.start_time:
                age_seconds = (datetime.utcnow() - instance.start_time).total_seconds()
                if age_seconds < 3600:  # Last hour
                    recent_activity.append({
                        "service_id": service_id,
                        "action": "restart" if instance.restart_count > 0 else "start",
                        "timestamp": instance.start_time.isoformat(),
                        "restart_count": instance.restart_count
                    })
        
        # Sort recent activity by timestamp
        recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_services": len(service_status),
                "healthy_services": healthy_services,
                "unhealthy_services": len(service_status) - healthy_services,
                "health_percentage": (healthy_services / len(service_status) * 100) if service_status else 0,
                "average_uptime_hours": (total_uptime / len(service_status) / 3600) if service_status else 0,
                "total_restarts": total_restarts
            },
            "service_types": service_by_type,
            "service_health": service_health,
            "recent_activity": recent_activity[:10],  # Last 10 activities
            "alerts": []
        }
        
        # Generate alerts for unhealthy services
        for service_id, health in service_health.items():
            if not health.get("healthy", False):
                dashboard["alerts"].append({
                    "level": "warning",
                    "service_id": service_id,
                    "message": f"Service {service_id} is unhealthy: {health.get('reason', 'Unknown')}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Generate alerts for high restart counts
        for service_id, instance in service_status.items():
            if instance.restart_count >= 3:
                dashboard["alerts"].append({
                    "level": "error",
                    "service_id": service_id,
                    "message": f"Service {service_id} has restarted {instance.restart_count} times",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Failed to get platform dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve platform dashboard")


# Bulk Service Operations
@router.post("/services/bulk-action", summary="Perform bulk service operations")
async def bulk_service_action(
    action_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user: UserClaims = Depends(require_auth),
    orchestrator = Depends(get_service_orchestrator)
):
    """Perform bulk operations on multiple services"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        action = action_data.get("action")  # "start", "stop", "restart"
        service_ids = action_data.get("service_ids", [])
        
        if not action or not service_ids:
            raise HTTPException(status_code=400, detail="Action and service_ids required")
        
        if action not in ["start", "stop", "restart"]:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        # Execute bulk action in background
        def bulk_action_task():
            import asyncio
            async def run_bulk_action():
                results = {}
                for service_id in service_ids:
                    if action == "start":
                        results[service_id] = await orchestrator.start_service(service_id)
                    elif action == "stop":
                        results[service_id] = await orchestrator.stop_service(service_id)
                    elif action == "restart":
                        results[service_id] = await orchestrator.restart_service(service_id)
                    
                    # Brief pause between operations
                    await asyncio.sleep(0.5)
                
                logger.info(f"Bulk {action} completed for services {service_ids}: {results}")
            
            asyncio.create_task(run_bulk_action())
        
        background_tasks.add_task(bulk_action_task)
        
        add_trace_context(
            operation=f"bulk_service_{action}",
            service_count=len(service_ids),
            requested_by=user.user_id
        )
        
        return {
            "message": f"Bulk {action} initiated for {len(service_ids)} services",
            "action": action,
            "service_ids": service_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute bulk service action: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute bulk action")