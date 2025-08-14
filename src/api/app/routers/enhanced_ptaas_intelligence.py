"""
Enhanced PTaaS Intelligence Router
Principal Auditor Implementation: Strategic Intelligence-Driven Security Testing

This router provides enhanced PTaaS capabilities with integrated threat intelligence,
providing intelligence-driven scanning, contextual threat analysis, and automated
threat hunting capabilities.

Key Features:
- Intelligence-enhanced security scanning
- Real-time threat context integration
- Automated threat hunting query generation
- Risk assessment with global threat intelligence
- Predictive threat analysis and recommendations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.enhanced_threat_intelligence_service import (
    get_enhanced_threat_intelligence_service,
    EnhancedThreatIntelligenceService
)
from ..services.ptaas_orchestrator_service import (
    get_ptaas_orchestrator,
    PTaaSOrchestrator,
    PTaaSTarget
)
from ..middleware.tenant_context import get_current_tenant_id
from ..infrastructure.observability import add_trace_context, get_metrics_collector

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Enhanced PTaaS Intelligence"])

# Enhanced Request/Response Models
class IntelligenceEnhancedScanRequest(BaseModel):
    """Request model for intelligence-enhanced security scanning"""
    targets: List[Dict[str, Any]] = Field(..., description="Scan targets with intelligence preferences")
    scan_profile: str = Field(default="intelligence_driven", description="Intelligence-enhanced scan profile")
    intelligence_scope: str = Field(default="comprehensive", description="Scope of threat intelligence integration")
    threat_hunting_enabled: bool = Field(default=True, description="Enable threat hunting query generation")
    risk_assessment_enabled: bool = Field(default=True, description="Enable enhanced risk assessment")
    correlation_timeframe: int = Field(default=24, description="Intelligence correlation timeframe in hours")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional scan metadata")

class ThreatIntelligenceContextRequest(BaseModel):
    """Request model for threat intelligence context analysis"""
    targets: List[str] = Field(..., description="Target hosts/IPs for intelligence analysis")
    context_depth: str = Field(default="detailed", description="Depth of context analysis")
    include_predictions: bool = Field(default=True, description="Include predictive threat analysis")
    include_attribution: bool = Field(default=True, description="Include threat actor attribution")

class ThreatHuntingQueryRequest(BaseModel):
    """Request model for threat hunting query generation"""
    scan_session_id: str = Field(..., description="Scan session ID for query generation")
    query_types: List[str] = Field(default=["network", "dns", "file"], description="Types of queries to generate")
    siem_platforms: List[str] = Field(default=["elasticsearch", "splunk"], description="Target SIEM platforms")
    timeframe: str = Field(default="7d", description="Recommended query timeframe")

class EnhancedScanResponse(BaseModel):
    """Response model for enhanced security scan"""
    session_id: str
    status: str
    scan_type: str
    intelligence_enhancement: Dict[str, Any]
    targets_count: int
    created_at: str
    estimated_completion: Optional[str] = None
    threat_landscape_snapshot: Optional[Dict[str, Any]] = None

class ThreatIntelligenceResponse(BaseModel):
    """Response model for threat intelligence analysis"""
    analysis_id: str
    timestamp: str
    targets_analyzed: int
    threat_indicators_found: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    threat_landscape_context: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float

class ThreatHuntingResponse(BaseModel):
    """Response model for threat hunting queries"""
    query_set_id: str
    session_id: str
    queries_generated: int
    query_types: List[str]
    siem_queries: Dict[str, List[Dict[str, Any]]]
    recommended_execution_order: List[str]
    estimated_coverage: float

class ThreatLandscapeResponse(BaseModel):
    """Response model for current threat landscape"""
    analysis_id: str
    timestamp: str
    overall_risk_score: float
    top_threats: List[Dict[str, Any]]
    emerging_threats: List[Dict[str, Any]]
    geographic_distribution: Dict[str, Any]
    actor_activity: Dict[str, Any]
    predictive_indicators: List[Dict[str, Any]]


@router.post("/intelligence-enhanced-scan", response_model=EnhancedScanResponse)
async def create_intelligence_enhanced_scan(
    request: IntelligenceEnhancedScanRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator),
    intelligence_service: EnhancedThreatIntelligenceService = Depends(get_enhanced_threat_intelligence_service)
):
    """
    Create an intelligence-enhanced security scan
    
    This endpoint creates a security scan enhanced with real-time threat intelligence,
    providing contextual threat analysis, risk assessment, and automated threat hunting.
    """
    try:
        # Validate targets and extract intelligence context
        ptaas_targets = []
        intelligence_context = {}
        
        for target_data in request.targets:
            target = PTaaSTarget(
                target_id=f"intel_target_{target_data.get('host', '')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                host=target_data.get("host", ""),
                ports=target_data.get("ports", []),
                scan_profile=request.scan_profile,
                constraints=[],
                authorized=target_data.get("authorized", True)
            )
            ptaas_targets.append(target)
            
            # Pre-scan intelligence gathering
            if request.intelligence_scope in ["comprehensive", "detailed"]:
                target_intel = await intelligence_service._analyze_target_intelligence(target)
                intelligence_context[target.host] = target_intel
        
        # Create enhanced scan session
        session_metadata = request.metadata or {}
        session_metadata.update({
            "intelligence_enhanced": True,
            "intelligence_scope": request.intelligence_scope,
            "threat_hunting_enabled": request.threat_hunting_enabled,
            "risk_assessment_enabled": request.risk_assessment_enabled,
            "correlation_timeframe_hours": request.correlation_timeframe,
            "pre_scan_intelligence": intelligence_context
        })
        
        session_id = await orchestrator.create_scan_session(
            targets=ptaas_targets,
            scan_type="intelligence_enhanced",
            tenant_id=tenant_id,
            metadata=session_metadata
        )
        
        # Start enhanced scan with intelligence integration
        background_tasks.add_task(
            _execute_intelligence_enhanced_scan,
            session_id,
            ptaas_targets,
            request,
            intelligence_service,
            orchestrator
        )
        
        # Get current threat landscape snapshot
        threat_landscape = None
        if intelligence_service.fusion_engine and intelligence_service.fusion_engine.threat_landscape:
            landscape = intelligence_service.fusion_engine.threat_landscape
            threat_landscape = {
                "overall_risk_score": landscape.risk_score,
                "top_threat_categories": [t["category"] for t in landscape.top_threats[:3]],
                "emerging_threat_count": len(landscape.emerging_threats),
                "confidence": landscape.confidence
            }
        
        # Calculate estimated completion time
        base_time = len(ptaas_targets) * 30  # 30 minutes per target base
        intelligence_overhead = 10 if request.intelligence_scope == "comprehensive" else 5
        estimated_completion = datetime.utcnow() + timedelta(minutes=base_time + intelligence_overhead)
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("intelligence_enhanced_scan_created", 1)
        
        # Add tracing context
        add_trace_context(
            operation="intelligence_enhanced_scan",
            session_id=session_id,
            tenant_id=str(tenant_id),
            targets_count=len(ptaas_targets),
            intelligence_scope=request.intelligence_scope
        )
        
        logger.info(f"Created intelligence-enhanced scan {session_id} for {len(ptaas_targets)} targets")
        
        return EnhancedScanResponse(
            session_id=session_id,
            status="initializing",
            scan_type="intelligence_enhanced",
            intelligence_enhancement={
                "scope": request.intelligence_scope,
                "threat_hunting_enabled": request.threat_hunting_enabled,
                "risk_assessment_enabled": request.risk_assessment_enabled,
                "pre_scan_indicators": sum(len(intel) for intel in intelligence_context.values())
            },
            targets_count=len(ptaas_targets),
            created_at=datetime.utcnow().isoformat(),
            estimated_completion=estimated_completion.isoformat(),
            threat_landscape_snapshot=threat_landscape
        )
        
    except ValueError as e:
        logger.error(f"Invalid request for intelligence-enhanced scan: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create intelligence-enhanced scan: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/threat-intelligence-context", response_model=ThreatIntelligenceResponse)
async def analyze_threat_intelligence_context(
    request: ThreatIntelligenceContextRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    intelligence_service: EnhancedThreatIntelligenceService = Depends(get_enhanced_threat_intelligence_service)
):
    """
    Analyze threat intelligence context for specified targets
    
    Provides comprehensive threat intelligence analysis including indicators,
    risk assessment, threat landscape context, and actionable recommendations.
    """
    try:
        analysis_id = f"intel_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze each target for threat intelligence
        all_indicators = []
        target_risks = {}
        
        for target in request.targets:
            # Create a mock scan target for analysis
            from ..domain.tenant_entities import ScanTarget
            scan_target = ScanTarget(
                host=target,
                ports=[],
                scan_profile="intelligence_only"
            )
            
            # Get threat intelligence for target
            target_intel = await intelligence_service._analyze_target_intelligence(scan_target)
            all_indicators.extend(target_intel)
            
            # Calculate target-specific risk
            target_risks[target] = await _calculate_target_risk(target_intel)
        
        # Perform overall risk assessment
        overall_risk = await _calculate_overall_risk_assessment(all_indicators, target_risks)
        
        # Get threat landscape context
        landscape_context = await intelligence_service._get_threat_landscape_context()
        
        # Generate context-aware recommendations
        recommendations = await _generate_context_recommendations(
            request.targets,
            all_indicators,
            overall_risk,
            landscape_context,
            request.include_predictions,
            request.include_attribution
        )
        
        # Calculate confidence score
        confidence_score = min(1.0, len(all_indicators) / 20.0 + 0.5)
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("threat_intelligence_context_analyzed", 1)
        
        logger.info(f"Analyzed threat intelligence context for {len(request.targets)} targets: "
                   f"{len(all_indicators)} indicators found")
        
        return ThreatIntelligenceResponse(
            analysis_id=analysis_id,
            timestamp=datetime.utcnow().isoformat(),
            targets_analyzed=len(request.targets),
            threat_indicators_found=all_indicators,
            risk_assessment=overall_risk,
            threat_landscape_context=landscape_context,
            recommendations=recommendations,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze threat intelligence context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate-threat-hunting-queries", response_model=ThreatHuntingResponse)
async def generate_threat_hunting_queries(
    request: ThreatHuntingQueryRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator),
    intelligence_service: EnhancedThreatIntelligenceService = Depends(get_enhanced_threat_intelligence_service)
):
    """
    Generate threat hunting queries based on scan results and intelligence
    
    Creates SIEM-specific threat hunting queries based on scan findings and
    correlated threat intelligence for proactive threat detection.
    """
    try:
        # Get scan session results
        session_status = await orchestrator.get_scan_session_status(request.scan_session_id)
        if not session_status:
            raise HTTPException(status_code=404, detail="Scan session not found")
        
        # Extract scan results for query generation
        scan_results = session_status.get("results", {})
        if not scan_results:
            raise HTTPException(status_code=400, detail="No scan results available for query generation")
        
        # Generate hunting queries for each SIEM platform
        siem_queries = {}
        total_queries = 0
        
        for platform in request.siem_platforms:
            platform_queries = await _generate_platform_specific_queries(
                request.scan_session_id,
                scan_results,
                platform,
                request.query_types,
                request.timeframe,
                intelligence_service
            )
            siem_queries[platform] = platform_queries
            total_queries += len(platform_queries)
        
        # Generate recommended execution order
        execution_order = await _generate_query_execution_order(siem_queries, scan_results)
        
        # Calculate estimated coverage
        estimated_coverage = await _calculate_query_coverage(siem_queries, scan_results)
        
        query_set_id = f"hunting_queries_{request.scan_session_id}_{datetime.utcnow().strftime('%H%M%S')}"
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("threat_hunting_queries_generated", 1)
        
        logger.info(f"Generated {total_queries} threat hunting queries for session {request.scan_session_id}")
        
        return ThreatHuntingResponse(
            query_set_id=query_set_id,
            session_id=request.scan_session_id,
            queries_generated=total_queries,
            query_types=request.query_types,
            siem_queries=siem_queries,
            recommended_execution_order=execution_order,
            estimated_coverage=estimated_coverage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate threat hunting queries: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/threat-landscape", response_model=ThreatLandscapeResponse)
async def get_current_threat_landscape(
    include_predictions: bool = Query(True, description="Include predictive threat indicators"),
    include_geography: bool = Query(True, description="Include geographic threat distribution"),
    confidence_threshold: float = Query(0.7, description="Minimum confidence threshold for inclusion"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    intelligence_service: EnhancedThreatIntelligenceService = Depends(get_enhanced_threat_intelligence_service)
):
    """
    Get current global threat landscape analysis
    
    Provides real-time threat landscape information including top threats,
    emerging threats, geographic distribution, and predictive indicators.
    """
    try:
        if not intelligence_service.fusion_engine or not intelligence_service.fusion_engine.threat_landscape:
            # Trigger fresh analysis if no current landscape
            await intelligence_service.fusion_engine.fuse_intelligence()
        
        landscape = intelligence_service.fusion_engine.threat_landscape
        if not landscape:
            raise HTTPException(status_code=503, detail="Threat landscape analysis unavailable")
        
        # Filter data based on confidence threshold
        filtered_top_threats = [
            threat for threat in landscape.top_threats
            if threat.get("average_severity", 0) * threat.get("count", 0) / 100.0 >= confidence_threshold
        ]
        
        filtered_emerging_threats = [
            threat for threat in landscape.emerging_threats
            if threat.get("confidence", 0) >= confidence_threshold
        ]
        
        # Include predictive indicators if requested
        predictive_data = landscape.predictive_indicators if include_predictions else []
        
        # Include geographic data if requested
        geographic_data = landscape.geographic_distribution if include_geography else {}
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("threat_landscape_retrieved", 1)
        
        return ThreatLandscapeResponse(
            analysis_id=landscape.analysis_id,
            timestamp=landscape.timestamp.isoformat(),
            overall_risk_score=landscape.risk_score,
            top_threats=filtered_top_threats[:10],  # Top 10 threats
            emerging_threats=filtered_emerging_threats[:15],  # Top 15 emerging threats
            geographic_distribution=geographic_data,
            actor_activity=landscape.actor_activity,
            predictive_indicators=predictive_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get threat landscape: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions/{session_id}/intelligence-enhancement")
async def get_session_intelligence_enhancement(
    session_id: str,
    tenant_id: UUID = Depends(get_current_tenant_id),
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator),
    intelligence_service: EnhancedThreatIntelligenceService = Depends(get_enhanced_threat_intelligence_service)
):
    """
    Get intelligence enhancement details for a scan session
    
    Returns detailed threat intelligence analysis, risk assessment,
    and recommendations for a completed scan session.
    """
    try:
        # Get scan session status and results
        session_status = await orchestrator.get_scan_session_status(session_id)
        if not session_status:
            raise HTTPException(status_code=404, detail="Scan session not found")
        
        # Check if this was an intelligence-enhanced scan
        metadata = session_status.get("metadata", {})
        if not metadata.get("intelligence_enhanced", False):
            raise HTTPException(status_code=400, detail="Session was not intelligence-enhanced")
        
        # Get scan results
        scan_results = session_status.get("results", {})
        if not scan_results:
            return {"status": "pending", "message": "Scan results not yet available"}
        
        # Retrieve cached intelligence enhancement or regenerate
        enhancement_data = await _get_or_generate_enhancement_data(
            session_id,
            scan_results,
            metadata,
            intelligence_service
        )
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("intelligence_enhancement_retrieved", 1)
        
        return {
            "session_id": session_id,
            "enhancement_status": "complete",
            "intelligence_data": enhancement_data,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get intelligence enhancement for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/bulk-intelligence-analysis")
async def bulk_intelligence_analysis(
    targets: List[str],
    analysis_depth: str = "standard",
    include_related_indicators: bool = True,
    max_results_per_target: int = 50,
    tenant_id: UUID = Depends(get_current_tenant_id),
    intelligence_service: EnhancedThreatIntelligenceService = Depends(get_enhanced_threat_intelligence_service)
):
    """
    Perform bulk threat intelligence analysis on multiple targets
    
    Efficiently analyzes multiple targets for threat intelligence indicators,
    providing batch processing for large-scale intelligence operations.
    """
    try:
        if len(targets) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 targets allowed per bulk analysis")
        
        analysis_results = {}
        total_indicators = 0
        
        # Process targets in batches for efficiency
        batch_size = 10
        for i in range(0, len(targets), batch_size):
            batch = targets[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = []
            for target in batch:
                task = _analyze_single_target_intelligence(
                    target,
                    analysis_depth,
                    include_related_indicators,
                    max_results_per_target,
                    intelligence_service
                )
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for target, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    analysis_results[target] = {"error": str(result), "indicators": []}
                else:
                    analysis_results[target] = result
                    total_indicators += len(result.get("indicators", []))
        
        # Generate bulk analysis summary
        summary = await _generate_bulk_analysis_summary(analysis_results, targets)
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("bulk_intelligence_analysis_completed", 1)
        
        logger.info(f"Completed bulk intelligence analysis for {len(targets)} targets: {total_indicators} total indicators")
        
        return {
            "analysis_id": f"bulk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "targets_analyzed": len(targets),
            "total_indicators_found": total_indicators,
            "analysis_summary": summary,
            "results": analysis_results,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to perform bulk intelligence analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/fusion-metrics")
async def get_fusion_engine_metrics(
    tenant_id: UUID = Depends(get_current_tenant_id),
    intelligence_service: EnhancedThreatIntelligenceService = Depends(get_enhanced_threat_intelligence_service)
):
    """
    Get threat intelligence fusion engine metrics and statistics
    
    Provides operational metrics for the threat intelligence fusion engine
    including feed status, indicator counts, and performance statistics.
    """
    try:
        if not intelligence_service.fusion_engine:
            raise HTTPException(status_code=503, detail="Fusion engine not available")
        
        # Get fusion engine metrics
        fusion_metrics = await intelligence_service.fusion_engine.get_fusion_metrics()
        
        # Add service-specific metrics
        service_metrics = {
            "intelligence_cache_size": len(intelligence_service.intelligence_cache),
            "generated_hunting_queries": len(intelligence_service.generated_queries),
            "service_uptime": "operational",  # Would calculate actual uptime
            "last_intelligence_update": datetime.utcnow().isoformat()
        }
        
        # Combine metrics
        combined_metrics = {
            "fusion_engine": fusion_metrics,
            "intelligence_service": service_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("fusion_metrics_retrieved", 1)
        
        return combined_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fusion engine metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Helper functions for request processing
async def _execute_intelligence_enhanced_scan(
    session_id: str,
    targets: List,
    request: IntelligenceEnhancedScanRequest,
    intelligence_service: EnhancedThreatIntelligenceService,
    orchestrator: PTaaSOrchestrator
):
    """Execute intelligence-enhanced scan with full integration"""
    try:
        # Start the actual scan
        await orchestrator.start_scan_session(session_id)
        
        # Wait for scan completion (simplified - would use proper async monitoring)
        await asyncio.sleep(5)  # Placeholder for scan execution
        
        # Get scan results for enhancement
        session_status = await orchestrator.get_scan_session_status(session_id)
        scan_results = session_status.get("results", {})
        
        if scan_results:
            # Enhance scan results with intelligence
            for target in targets:
                # Create mock scan result for demonstration
                from ..domain.tenant_entities import ScanResult
                mock_scan_result = ScanResult(
                    scan_id=session_id,
                    target=target.host,
                    scan_type="intelligence_enhanced",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    status="completed",
                    open_ports=[],
                    services=[],
                    vulnerabilities=[],
                    os_fingerprint={},
                    scan_statistics={},
                    raw_output={},
                    findings=[],
                    recommendations=[]
                )
                
                # Apply intelligence enhancement
                enhancement = await intelligence_service.enhance_scan_with_intelligence(target, mock_scan_result)
                
                # Store enhancement results (would persist to database in production)
                logger.info(f"Enhanced scan results for {target.host} with {len(enhancement.get('intelligence_enhancement', {}).get('threat_indicators_found', []))} threat indicators")
        
    except Exception as e:
        logger.error(f"Intelligence-enhanced scan execution failed: {e}")


async def _calculate_target_risk(threat_indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate risk assessment for a single target"""
    if not threat_indicators:
        return {"risk_score": 0.0, "risk_level": "low", "factors": []}
    
    # Calculate risk based on indicators
    total_risk = 0.0
    risk_factors = []
    
    for indicator in threat_indicators:
        severity = indicator.get("severity", "low")
        confidence = indicator.get("confidence", 0.5)
        
        severity_weight = {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3,
            "info": 0.1
        }.get(severity, 0.3)
        
        indicator_risk = severity_weight * confidence
        total_risk += indicator_risk
        
        if severity in ["critical", "high"]:
            risk_factors.append(f"{severity} indicator: {indicator.get('category', 'unknown')}")
    
    # Normalize risk score
    risk_score = min(1.0, total_risk / len(threat_indicators))
    
    # Determine risk level
    if risk_score >= 0.8:
        risk_level = "critical"
    elif risk_score >= 0.6:
        risk_level = "high"
    elif risk_score >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "factors": risk_factors,
        "indicator_count": len(threat_indicators)
    }


async def _calculate_overall_risk_assessment(
    all_indicators: List[Dict[str, Any]], 
    target_risks: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate overall risk assessment across all targets"""
    if not target_risks:
        return {"overall_risk": 0.0, "risk_level": "low", "summary": "No threats detected"}
    
    # Calculate weighted average risk
    total_risk = sum(risk["risk_score"] for risk in target_risks.values())
    avg_risk = total_risk / len(target_risks)
    
    # Count high-risk targets
    high_risk_targets = len([risk for risk in target_risks.values() if risk["risk_score"] >= 0.6])
    
    # Determine overall risk level
    if avg_risk >= 0.8 or high_risk_targets > len(target_risks) * 0.5:
        risk_level = "critical"
    elif avg_risk >= 0.6 or high_risk_targets > 0:
        risk_level = "high"
    elif avg_risk >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "overall_risk": avg_risk,
        "risk_level": risk_level,
        "high_risk_targets": high_risk_targets,
        "total_targets": len(target_risks),
        "total_indicators": len(all_indicators),
        "summary": f"{len(all_indicators)} threat indicators across {len(target_risks)} targets"
    }


async def _generate_context_recommendations(
    targets: List[str],
    indicators: List[Dict[str, Any]],
    risk_assessment: Dict[str, Any],
    landscape_context: Dict[str, Any],
    include_predictions: bool,
    include_attribution: bool
) -> List[str]:
    """Generate context-aware recommendations"""
    recommendations = []
    
    risk_level = risk_assessment.get("risk_level", "low")
    
    # Risk-based recommendations
    if risk_level == "critical":
        recommendations.extend([
            "🚨 CRITICAL: Immediate security response required",
            "🔥 Isolate high-risk targets immediately",
            "📞 Escalate to security leadership"
        ])
    elif risk_level == "high":
        recommendations.extend([
            "⚠️ HIGH: Enhanced monitoring required",
            "🎯 Deploy targeted threat hunting"
        ])
    
    # Indicator-based recommendations
    critical_indicators = [i for i in indicators if i.get("severity") == "critical"]
    if critical_indicators:
        recommendations.append(f"🎯 {len(critical_indicators)} critical indicators require immediate blocking")
    
    # Attribution-based recommendations
    if include_attribution:
        nation_state_indicators = [i for i in indicators if "NATION_STATE" in i.get("attributed_actors", [])]
        if nation_state_indicators:
            recommendations.append("🏛️ Nation-state attribution detected - escalate to senior leadership")
    
    # Predictive recommendations
    if include_predictions and landscape_context.get("predictive_alerts", 0) > 0:
        recommendations.append("🔮 Predictive threats identified - implement proactive controls")
    
    # General recommendations
    recommendations.extend([
        "📊 Implement continuous threat monitoring",
        "🔍 Execute generated threat hunting queries",
        "📈 Maintain threat intelligence integration"
    ])
    
    return recommendations[:15]


async def _generate_platform_specific_queries(
    session_id: str,
    scan_results: Dict[str, Any],
    platform: str,
    query_types: List[str],
    timeframe: str,
    intelligence_service: EnhancedThreatIntelligenceService
) -> List[Dict[str, Any]]:
    """Generate SIEM platform-specific queries"""
    queries = []
    
    # Mock implementation - in production would be more sophisticated
    if "network" in query_types:
        if platform == "elasticsearch":
            query = {
                "query_id": f"net_activity_{session_id[:8]}",
                "name": "Network Activity Hunt",
                "query": {
                    "bool": {
                        "must": [
                            {"range": {"@timestamp": {"gte": f"now-{timeframe}"}}},
                            {"exists": {"field": "source.ip"}}
                        ]
                    }
                },
                "description": "Hunt for network activity patterns",
                "timeframe": timeframe
            }
            queries.append(query)
        
        elif platform == "splunk":
            query = {
                "query_id": f"net_activity_{session_id[:8]}",
                "name": "Network Activity Hunt",
                "query": f"index=* earliest=-{timeframe} (src_ip=* OR dest_ip=*)",
                "description": "Hunt for network activity patterns",
                "timeframe": timeframe
            }
            queries.append(query)
    
    return queries


async def _generate_query_execution_order(
    siem_queries: Dict[str, List[Dict[str, Any]]], 
    scan_results: Dict[str, Any]
) -> List[str]:
    """Generate recommended query execution order"""
    # Prioritize by criticality and dependencies
    execution_order = []
    
    for platform, queries in siem_queries.items():
        for query in queries:
            query_id = query.get("query_id", "")
            if query_id:
                execution_order.append(query_id)
    
    return execution_order


async def _calculate_query_coverage(
    siem_queries: Dict[str, List[Dict[str, Any]]], 
    scan_results: Dict[str, Any]
) -> float:
    """Calculate estimated coverage of generated queries"""
    # Mock implementation - would calculate actual coverage metrics
    total_queries = sum(len(queries) for queries in siem_queries.values())
    
    # Coverage based on query diversity and scan findings
    base_coverage = min(0.8, total_queries / 10.0)
    
    return base_coverage


async def _get_or_generate_enhancement_data(
    session_id: str,
    scan_results: Dict[str, Any],
    metadata: Dict[str, Any],
    intelligence_service: EnhancedThreatIntelligenceService
) -> Dict[str, Any]:
    """Get cached or generate intelligence enhancement data"""
    # In production, would check cache first
    cache_key = f"enhancement_{session_id}"
    
    # Generate fresh enhancement data
    enhancement_data = {
        "session_id": session_id,
        "intelligence_scope": metadata.get("intelligence_scope", "standard"),
        "threat_indicators": [],  # Would be populated from actual analysis
        "risk_assessment": {"overall_risk": 0.5, "risk_level": "medium"},
        "threat_hunting_queries": [],  # Would be populated from query generation
        "recommendations": [
            "📊 Review threat intelligence findings",
            "🔍 Execute recommended hunting queries",
            "📈 Monitor for related threats"
        ],
        "generated_at": datetime.utcnow().isoformat()
    }
    
    return enhancement_data


async def _analyze_single_target_intelligence(
    target: str,
    analysis_depth: str,
    include_related: bool,
    max_results: int,
    intelligence_service: EnhancedThreatIntelligenceService
) -> Dict[str, Any]:
    """Analyze intelligence for a single target"""
    try:
        # Create scan target for analysis
        from ..domain.tenant_entities import ScanTarget
        scan_target = ScanTarget(
            host=target,
            ports=[],
            scan_profile="intelligence_only"
        )
        
        # Get threat intelligence
        indicators = await intelligence_service._analyze_target_intelligence(scan_target)
        
        # Limit results
        limited_indicators = indicators[:max_results]
        
        # Calculate target risk
        risk_data = await _calculate_target_risk(limited_indicators)
        
        return {
            "target": target,
            "indicators": limited_indicators,
            "risk_assessment": risk_data,
            "analysis_depth": analysis_depth,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Analysis failed for target {target}: {e}")


async def _generate_bulk_analysis_summary(
    analysis_results: Dict[str, Dict[str, Any]], 
    targets: List[str]
) -> Dict[str, Any]:
    """Generate summary for bulk analysis results"""
    total_indicators = 0
    high_risk_targets = 0
    errors = 0
    
    for target, result in analysis_results.items():
        if "error" in result:
            errors += 1
        else:
            total_indicators += len(result.get("indicators", []))
            risk_level = result.get("risk_assessment", {}).get("risk_level", "low")
            if risk_level in ["high", "critical"]:
                high_risk_targets += 1
    
    return {
        "targets_processed": len(targets),
        "successful_analyses": len(targets) - errors,
        "failed_analyses": errors,
        "total_indicators_found": total_indicators,
        "high_risk_targets": high_risk_targets,
        "average_indicators_per_target": total_indicators / max(len(targets) - errors, 1),
        "analysis_success_rate": (len(targets) - errors) / len(targets) if targets else 0.0
    }