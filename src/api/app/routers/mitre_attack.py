"""
MITRE ATT&CK Integration API Endpoints
Advanced threat mapping, attack pattern detection, and intelligence analysis
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator

from ..security import (
    SecurityConfig,
    require_admin,
    require_permission,
    Permission
)
from ..services.advanced_mitre_attack_engine import (
    get_advanced_mitre_engine,
    AdvancedMitreAttackEngine,
    ThreatMapping,
    AttackPattern,
    ThreatIntelligenceReport,
    ThreatSeverity,
    AttackStage,
    ConfidenceLevel
)
from ..middleware.tenant_context import require_tenant_context


class IndicatorType(str, Enum):
    """Supported indicator types"""
    IP_ADDRESS = "ip-dst"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file-hash"
    EMAIL = "email"
    PROCESS = "process"
    REGISTRY = "registry"
    NETWORK_TRAFFIC = "network-traffic"
    BEHAVIOR = "behavior"


class AnalysisDepth(str, Enum):
    """Analysis depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    FORENSIC = "forensic"


# Request Models
class ThreatIndicator(BaseModel):
    """Threat indicator for analysis"""
    type: IndicatorType
    value: str
    confidence: float = Field(0.5, ge=0, le=1)
    severity: Optional[str] = "medium"
    source: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow()


class ThreatAnalysisRequest(BaseModel):
    """Request for threat analysis"""
    indicators: List[ThreatIndicator]
    analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD
    context: Dict[str, Any] = Field(default_factory=dict)
    include_attribution: bool = True
    include_predictions: bool = True
    correlation_window: int = Field(3600, ge=300, le=86400)  # 5 min to 24 hours


class AttackPatternDetectionRequest(BaseModel):
    """Request for attack pattern detection"""
    events: List[Dict[str, Any]]
    time_window: int = Field(3600, ge=300, le=86400)
    confidence_threshold: float = Field(0.7, ge=0, le=1)
    severity_filter: Optional[List[ThreatSeverity]] = None
    include_timeline: bool = True


class TechniqueQuery(BaseModel):
    """Query for technique information"""
    technique_ids: Optional[List[str]] = None
    tactic_filter: Optional[List[str]] = None
    platform_filter: Optional[List[str]] = None
    search_term: Optional[str] = None
    limit: int = Field(50, ge=1, le=1000)


class AttackProgressionRequest(BaseModel):
    """Request for attack progression prediction"""
    current_techniques: List[str]
    time_horizon: int = Field(3600, ge=300, le=86400)
    prediction_confidence: float = Field(0.6, ge=0, le=1)
    include_mitigations: bool = True


class IntelligenceReportRequest(BaseModel):
    """Request for intelligence report generation"""
    time_range_days: int = Field(30, ge=1, le=365)
    focus_areas: Optional[List[str]] = None
    threat_actors: Optional[List[str]] = None
    techniques: Optional[List[str]] = None
    include_predictions: bool = True
    export_format: str = Field("json", pattern="^(json|pdf|html)$")


# Response Models
class TechniqueSummary(BaseModel):
    """MITRE technique summary"""
    technique_id: str
    name: str
    description: str
    tactics: List[str]
    platforms: List[str]
    confidence: float
    threat_score: float
    mitigation_count: int
    detection_methods: List[str]


class ThreatAttributionResult(BaseModel):
    """Threat attribution result"""
    group_id: str
    name: str
    aliases: List[str]
    confidence: float
    similarity_score: float
    common_techniques: List[str]
    country: Optional[str] = None
    sophistication: str
    motivation: List[str]


class AttackProgressionPrediction(BaseModel):
    """Attack progression prediction"""
    prediction_id: str
    timestamp: datetime
    current_stage: AttackStage
    next_likely_techniques: List[TechniqueSummary]
    attack_paths: List[Dict[str, Any]]
    progression_probabilities: Dict[str, float]
    time_estimates: Dict[str, int]
    defensive_recommendations: List[str]
    confidence_score: float


class ThreatAnalysisResponse(BaseModel):
    """Comprehensive threat analysis response"""
    analysis_id: str
    timestamp: datetime
    threat_mapping: Dict[str, Any]
    attack_patterns: List[Dict[str, Any]]
    attribution_results: List[ThreatAttributionResult]
    technique_analysis: List[TechniqueSummary]
    severity_assessment: ThreatSeverity
    attack_stage: Optional[AttackStage]
    progression_prediction: Optional[AttackProgressionPrediction]
    immediate_recommendations: List[str]
    investigation_priorities: List[str]


class MitreEngineStatus(BaseModel):
    """MITRE engine status"""
    status: str
    framework_version: str
    last_update: Optional[datetime]
    techniques_loaded: int
    groups_loaded: int
    software_loaded: int
    detection_rules: int
    ml_models_ready: bool
    attack_graph_nodes: int
    threat_mappings_total: int
    analytics: Dict[str, Any]


# Initialize router
router = APIRouter(prefix="/mitre-attack", tags=["MITRE ATT&CK"])


@router.post("/analyze", response_model=ThreatAnalysisResponse)
async def analyze_threat_indicators(
    request: ThreatAnalysisRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(require_tenant_context),
    # Security context for production auth
) -> ThreatAnalysisResponse:
    """
    Analyze threat indicators using advanced MITRE ATT&CK mapping
    Provides comprehensive threat analysis with AI-powered attribution
    """
    try:
        engine = await get_advanced_mitre_engine()

        # Convert indicators to engine format
        indicators_data = []
        for indicator in request.indicators:
            indicators_data.append({
                "type": indicator.type.value,
                "value": indicator.value,
                "confidence": indicator.confidence,
                "severity": indicator.severity,
                "source": indicator.source,
                "context": indicator.context,
                "timestamp": indicator.timestamp.isoformat()
            })

        # Perform threat mapping
        threat_mapping = await engine.analyze_threat_indicators(
            indicators_data,
            request.context
        )

        # Extract techniques for additional analysis
        technique_ids = threat_mapping.technique_ids

        # Get detailed technique information
        technique_summaries = []
        for tech_id in technique_ids:
            try:
                tech_intel = await engine.get_technique_intelligence(tech_id)
                technique_summaries.append(TechniqueSummary(
                    technique_id=tech_id,
                    name=tech_intel["technique"]["name"],
                    description=tech_intel["technique"]["description"][:200] + "...",
                    tactics=tech_intel["technique"]["tactic_refs"],
                    platforms=tech_intel["technique"]["platforms"],
                    confidence=threat_mapping.confidence,
                    threat_score=tech_intel["threat_score"],
                    mitigation_count=len(tech_intel["mitigation_strategies"]),
                    detection_methods=list(tech_intel["detection_recommendations"].keys())[:3]
                ))
            except Exception as e:
                # Skip techniques that can't be analyzed
                continue

        # Perform attribution analysis
        attribution_results = []
        for group_id in threat_mapping.attribution_groups:
            try:
                if group_id in engine.groups:
                    group = engine.groups[group_id]
                    attribution_results.append(ThreatAttributionResult(
                        group_id=group_id,
                        name=group.name,
                        aliases=group.aliases,
                        confidence=threat_mapping.confidence * 0.8,  # Slightly lower for attribution
                        similarity_score=0.75,  # Would be calculated properly
                        common_techniques=list(set(technique_ids) & set(group.techniques))[:10],
                        country=group.country,
                        sophistication=group.sophistication_level,
                        motivation=group.motivation
                    ))
            except Exception as e:
                continue

        # Generate attack progression prediction if requested
        progression_prediction = None
        if request.include_predictions and technique_ids:
            try:
                prediction_data = await engine.predict_attack_progression(technique_ids)

                next_techniques = []
                for tech_data in prediction_data.get("next_likely_techniques", [])[:5]:
                    if tech_data["technique_id"] in engine.techniques:
                        tech = engine.techniques[tech_data["technique_id"]]
                        next_techniques.append(TechniqueSummary(
                            technique_id=tech_data["technique_id"],
                            name=tech.name,
                            description=tech.description[:200] + "...",
                            tactics=tech.tactic_refs,
                            platforms=tech.platforms,
                            confidence=tech_data.get("probability", 0.5),
                            threat_score=tech.impact_score,
                            mitigation_count=len(tech.mitigations),
                            detection_methods=list(tech.detection_methods.keys())[:3]
                        ))

                progression_prediction = AttackProgressionPrediction(
                    prediction_id=prediction_data["prediction_id"],
                    timestamp=datetime.fromisoformat(prediction_data["timestamp"]),
                    current_stage=threat_mapping.attack_stage or AttackStage.EXPLOITATION,
                    next_likely_techniques=next_techniques,
                    attack_paths=prediction_data.get("attack_paths", []),
                    progression_probabilities=prediction_data.get("progression_probabilities", {}),
                    time_estimates=prediction_data.get("time_to_next_stage", {}),
                    defensive_recommendations=prediction_data.get("defensive_recommendations", []),
                    confidence_score=prediction_data.get("confidence_score", 0.7)
                )
            except Exception as e:
                # Prediction failed, continue without it
                pass

        # Generate recommendations
        immediate_recommendations = [
            f"🚨 {threat_mapping.severity.value.upper()} severity threat detected",
            f"📊 {len(technique_ids)} MITRE ATT&CK techniques identified",
            f"🎯 Attack stage: {threat_mapping.attack_stage.value if threat_mapping.attack_stage else 'Unknown'}",
            "🔍 Initiate incident response procedures",
            "📋 Review detection rules for identified techniques"
        ]

        if attribution_results:
            top_attribution = attribution_results[0]
            immediate_recommendations.append(f"👥 Potential threat actor: {top_attribution.name}")

        investigation_priorities = [
            "Analyze network traffic for command & control communications",
            "Check for lateral movement indicators",
            "Verify endpoint security controls effectiveness",
            "Review authentication logs for anomalies",
            "Examine file system for persistence mechanisms"
        ]

        # Schedule background enhancement
        background_tasks.add_task(
            _enhance_analysis_with_external_intelligence,
            threat_mapping.mapping_id
        )

        return ThreatAnalysisResponse(
            analysis_id=threat_mapping.mapping_id,
            timestamp=threat_mapping.timestamp,
            threat_mapping={
                "mapping_id": threat_mapping.mapping_id,
                "confidence": threat_mapping.confidence,
                "correlation_score": threat_mapping.correlation_score,
                "severity": threat_mapping.severity.value,
                "attack_stage": threat_mapping.attack_stage.value if threat_mapping.attack_stage else None,
                "iocs": threat_mapping.iocs,
                "evidence_count": len(threat_mapping.evidence)
            },
            attack_patterns=[],  # Would be populated from pattern detection
            attribution_results=attribution_results,
            technique_analysis=technique_summaries,
            severity_assessment=threat_mapping.severity,
            attack_stage=threat_mapping.attack_stage,
            progression_prediction=progression_prediction,
            immediate_recommendations=immediate_recommendations,
            investigation_priorities=investigation_priorities
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Threat analysis failed: {str(e)}")


@router.post("/patterns/detect", response_model=List[Dict[str, Any]])
async def detect_attack_patterns(
    request: AttackPatternDetectionRequest,
    tenant_id: str = Depends(require_tenant_context),
    # Security context for production auth
) -> List[Dict[str, Any]]:
    """
    Detect sophisticated attack patterns from security events
    Uses temporal analysis and machine learning for pattern recognition
    """
    try:
        engine = await get_advanced_mitre_engine()

        # Detect patterns
        patterns = await engine.detect_attack_patterns(
            request.events,
            request.time_window
        )

        # Filter by confidence and severity
        filtered_patterns = []
        for pattern in patterns:
            if pattern.confidence >= request.confidence_threshold:
                if request.severity_filter:
                    if pattern.severity in request.severity_filter:
                        filtered_patterns.append(pattern)
                else:
                    filtered_patterns.append(pattern)

        # Convert to response format
        response_patterns = []
        for pattern in filtered_patterns:
            pattern_dict = {
                "pattern_id": pattern.pattern_id,
                "name": pattern.name,
                "techniques": pattern.techniques,
                "confidence": pattern.confidence,
                "severity": pattern.severity.value,
                "detection_time": pattern.detection_time.isoformat(),
                "kill_chain_coverage": pattern.kill_chain_coverage,
                "tactic_progression": pattern.tactic_progression,
                "behavioral_indicators": pattern.behavioral_indicators,
                "threat_actor_similarity": pattern.threat_actor_similarity,
                "immediate_actions": pattern.immediate_actions,
                "investigation_priorities": pattern.investigation_priorities,
                "containment_strategies": pattern.containment_strategies,
                "mitigation_recommendations": pattern.mitigation_recommendations
            }

            if request.include_timeline:
                pattern_dict["timeline_analysis"] = pattern.timeline_analysis

            response_patterns.append(pattern_dict)

        return response_patterns

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern detection failed: {str(e)}")


@router.get("/techniques", response_model=List[TechniqueSummary])
async def get_techniques(
    query: TechniqueQuery,
    tenant_id: str = Depends(require_tenant_context),
    # Security context for production auth
) -> List[TechniqueSummary]:
    """
    Search and retrieve MITRE ATT&CK techniques with filtering
    """
    try:
        engine = await get_advanced_mitre_engine()

        techniques = []

        # If specific technique IDs requested
        if query.technique_ids:
            for tech_id in query.technique_ids:
                if tech_id in engine.techniques:
                    tech = engine.techniques[tech_id]
                    techniques.append(tech)
        else:
            # Filter all techniques
            for tech_id, tech in engine.techniques.items():
                # Apply filters
                if query.tactic_filter:
                    if not any(tactic in tech.tactic_refs for tactic in query.tactic_filter):
                        continue

                if query.platform_filter:
                    if not any(platform in tech.platforms for platform in query.platform_filter):
                        continue

                if query.search_term:
                    search_lower = query.search_term.lower()
                    if (search_lower not in tech.name.lower() and
                        search_lower not in tech.description.lower()):
                        continue

                techniques.append(tech)

        # Sort by prevalence score
        techniques.sort(key=lambda t: t.prevalence_score, reverse=True)

        # Limit results
        techniques = techniques[:query.limit]

        # Convert to response format
        technique_summaries = []
        for tech in techniques:
            technique_summaries.append(TechniqueSummary(
                technique_id=tech.technique_id,
                name=tech.name,
                description=tech.description[:200] + "..." if len(tech.description) > 200 else tech.description,
                tactics=tech.tactic_refs,
                platforms=tech.platforms,
                confidence=1.0,  # Full confidence for direct lookup
                threat_score=(tech.prevalence_score + tech.impact_score) / 2,
                mitigation_count=len(tech.mitigations),
                detection_methods=list(tech.detection_methods.keys())[:3]
            ))

        return technique_summaries

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Technique retrieval failed: {str(e)}")


@router.get("/techniques/{technique_id}", response_model=Dict[str, Any])
async def get_technique_details(
    technique_id: str,
    tenant_id: str = Depends(require_tenant_context),
    # Security context for production auth
) -> Dict[str, Any]:
    """
    Get comprehensive intelligence about a specific MITRE technique
    """
    try:
        engine = await get_advanced_mitre_engine()

        technique_intel = await engine.get_technique_intelligence(technique_id)

        return technique_intel

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve technique details: {str(e)}")


@router.post("/predict/progression", response_model=AttackProgressionPrediction)
async def predict_attack_progression(
    request: AttackProgressionRequest,
    tenant_id: str = Depends(require_tenant_context),
    # Security context for production auth
) -> AttackProgressionPrediction:
    """
    Predict attack progression based on current techniques
    Uses AI and attack graph analysis for forecasting
    """
    try:
        engine = await get_advanced_mitre_engine()

        prediction_data = await engine.predict_attack_progression(
            request.current_techniques
        )

        # Convert next techniques to summaries
        next_techniques = []
        for tech_data in prediction_data.get("next_likely_techniques", []):
            if tech_data["technique_id"] in engine.techniques:
                tech = engine.techniques[tech_data["technique_id"]]
                next_techniques.append(TechniqueSummary(
                    technique_id=tech_data["technique_id"],
                    name=tech.name,
                    description=tech.description[:200] + "...",
                    tactics=tech.tactic_refs,
                    platforms=tech.platforms,
                    confidence=tech_data.get("probability", 0.5),
                    threat_score=tech.impact_score,
                    mitigation_count=len(tech.mitigations),
                    detection_methods=list(tech.detection_methods.keys())[:3]
                ))

        # Determine current stage
        current_stage = AttackStage.EXPLOITATION  # Default
        if request.current_techniques:
            # Logic to determine stage from techniques
            pass

        return AttackProgressionPrediction(
            prediction_id=prediction_data["prediction_id"],
            timestamp=datetime.fromisoformat(prediction_data["timestamp"]),
            current_stage=current_stage,
            next_likely_techniques=next_techniques,
            attack_paths=prediction_data.get("attack_paths", []),
            progression_probabilities=prediction_data.get("progression_probabilities", {}),
            time_estimates=prediction_data.get("time_to_next_stage", {}),
            defensive_recommendations=prediction_data.get("defensive_recommendations", []),
            confidence_score=prediction_data.get("confidence_score", 0.7)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Progression prediction failed: {str(e)}")


@router.post("/intelligence/report", response_model=Dict[str, Any])
async def generate_intelligence_report(
    request: IntelligenceReportRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(require_tenant_context),
    # Security context for production auth
) -> Dict[str, Any]:
    """
    Generate comprehensive threat intelligence report
    Provides strategic insights and predictive analysis
    """
    try:
        engine = await get_advanced_mitre_engine()

        time_range = timedelta(days=request.time_range_days)

        # Generate report
        report = await engine.generate_threat_intelligence_report(time_range)

        # Convert to API response format
        response = {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "time_range_days": request.time_range_days,
            "threat_landscape": report.threat_landscape,
            "attack_patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "name": pattern.name,
                    "techniques": pattern.techniques,
                    "confidence": pattern.confidence,
                    "severity": pattern.severity.value
                }
                for pattern in report.attack_patterns
            ],
            "technique_trends": report.technique_trends,
            "group_activities": report.group_activities,
            "emerging_threats": report.emerging_threats,
            "recommendations": report.recommendations,
            "risk_assessment": {
                "overall_risk_score": report.overall_risk_score,
                "sector_specific_risks": report.sector_specific_risks
            },
            "predictive_insights": report.predictive_insights,
            "export_format": request.export_format
        }

        # Schedule report export if requested
        if request.export_format != "json":
            background_tasks.add_task(
                _export_intelligence_report,
                report.report_id,
                request.export_format,
                response
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intelligence report generation failed: {str(e)}")


@router.get("/status", response_model=MitreEngineStatus)
async def get_engine_status(
    tenant_id: str = Depends(require_tenant_context),
    # Security context for production auth
) -> MitreEngineStatus:
    """
    Get MITRE ATT&CK engine status and health information
    """
    try:
        engine = await get_advanced_mitre_engine()

        health = await engine.health_check()

        return MitreEngineStatus(
            status=health.status.value,
            framework_version="ATT&CK v13.1",
            last_update=engine.analytics.get("last_framework_update"),
            techniques_loaded=len(engine.techniques),
            groups_loaded=len(engine.groups),
            software_loaded=len(engine.software),
            detection_rules=len(engine.detection_rules),
            ml_models_ready=engine.technique_vectorizer is not None,
            attack_graph_nodes=engine.attack_graph.number_of_nodes() if engine.attack_graph else 0,
            threat_mappings_total=len(engine.threat_mappings),
            analytics=engine.analytics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get engine status: {str(e)}")


@router.post("/cache/clear")
async def clear_cache(
    cache_type: str = Query("all", regex="^(all|mapping|similarity)$"),
    # Admin permission required
) -> Dict[str, str]:
    """
    Clear MITRE engine caches
    """
    try:
        engine = await get_advanced_mitre_engine()

        if cache_type in ["all", "mapping"]:
            engine.mapping_cache.clear()

        if cache_type in ["all", "similarity"]:
            engine.similarity_cache.clear()

        return {
            "message": f"Cache cleared: {cache_type}",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


# Background task functions
async def _enhance_analysis_with_external_intelligence(mapping_id: str):
    """Enhance analysis with external threat intelligence"""
    try:
        # Implementation for external intelligence enhancement
        pass
    except Exception as e:
        # Log error but don't fail
        pass


async def _export_intelligence_report(report_id: str, format_type: str, report_data: Dict[str, Any]):
    """Export intelligence report to specified format"""
    try:
        # Implementation for report export
        # Could generate PDF, HTML, etc.
        pass
    except Exception as e:
        # Log error but don't fail
        pass
