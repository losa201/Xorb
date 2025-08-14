#!/usr/bin/env python3
"""
Advanced PTaaS API Router
Production-grade API endpoints for sophisticated penetration testing services

Features:
- Advanced African market penetration testing
- AI-powered vulnerability assessment
- Automated exploit development
- Compliance automation for African frameworks
- Threat hunting and incident response
- Real-time security monitoring
- Comprehensive reporting and analytics
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import json
import logging
from pathlib import Path as PathLib

async def _check_service_health_default(service_name: str) -> Dict[str, Any]:
    """Default health check implementation for services without specific health checks"""
    try:
        import psutil
        import time

        start_time = time.time()

        # Basic system checks
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        response_time = int((time.time() - start_time) * 1000)

        # Determine status based on resource utilization
        status = "healthy"
        checks = []

        if cpu_percent > 90:
            status = "degraded"
            checks.append("High CPU usage")

        if memory.percent > 90:
            status = "degraded"
            checks.append("High memory usage")

        if disk.percent > 95:
            status = "degraded"
            checks.append("Low disk space")

        if response_time > 1000:
            status = "degraded"
            checks.append("Slow response time")

        return {
            "status": status,
            "message": f"Default health check for {service_name}",
            "response_time_ms": response_time,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Health check failed for {service_name}: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

from ..services.advanced_africa_pentesting_suite import AdvancedAfricaPentestingSuite
from ..services.advanced_exploit_framework import AdvancedExploitFramework
try:
    from ..services.ai_vulnerability_assessment_engine import AIVulnerabilityAssessmentEngine
except ImportError:
    # Fallback when AI vulnerability assessment is not available
    AIVulnerabilityAssessmentEngine = None
from ..services.african_compliance_automation_engine import AfricanComplianceAutomationEngine
from ..services.advanced_threat_hunting_system import AdvancedThreatHuntingSystem
from ..dependencies import get_current_user, get_current_org, verify_api_key
from ..models.ptaas_models import *

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/advanced-ptaas", tags=["Advanced PTaaS"])
security = HTTPBearer()

# Service instances (would be dependency injected in production)
africa_pentesting_suite: Optional[AdvancedAfricaPentestingSuite] = None
exploit_framework: Optional[AdvancedExploitFramework] = None
ai_vulnerability_engine: Optional[AIVulnerabilityAssessmentEngine] = None
compliance_engine: Optional[AfricanComplianceAutomationEngine] = None
threat_hunting_system: Optional[AdvancedThreatHuntingSystem] = None

async def get_services():
    """Initialize and get service instances"""
    global africa_pentesting_suite, exploit_framework, ai_vulnerability_engine, compliance_engine, threat_hunting_system

    if not africa_pentesting_suite:
        africa_pentesting_suite = AdvancedAfricaPentestingSuite()
        await africa_pentesting_suite.initialize()

    if not exploit_framework:
        exploit_framework = AdvancedExploitFramework()
        await exploit_framework.initialize()

    if not ai_vulnerability_engine:
        ai_vulnerability_engine = AIVulnerabilityAssessmentEngine()
        await ai_vulnerability_engine.initialize()

    if not compliance_engine:
        compliance_engine = AfricanComplianceAutomationEngine()
        await compliance_engine.initialize()

    if not threat_hunting_system:
        threat_hunting_system = AdvancedThreatHuntingSystem()
        await threat_hunting_system.initialize()

    return {
        "africa_pentesting": africa_pentesting_suite,
        "exploit_framework": exploit_framework,
        "ai_vulnerability": ai_vulnerability_engine,
        "compliance": compliance_engine,
        "threat_hunting": threat_hunting_system
    }

# ====================== ADVANCED AFRICAN PENETRATION TESTING ======================

@router.post("/african-pentest/comprehensive",
             summary="Comprehensive African Market Penetration Test",
             description="Conduct comprehensive penetration test with African regional focus")
async def conduct_comprehensive_african_pentest(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    Conduct comprehensive penetration test optimized for African markets

    Features:
    - Regional threat intelligence for African cybersecurity landscape
    - Mobile money and fintech security testing
    - Telecommunications infrastructure assessment
    - Government and public sector penetration testing
    - Advanced social engineering with cultural context
    """
    try:
        services = await get_services()
        africa_service = services["africa_pentesting"]

        # Extract target information
        target_host = request.get("target_host")
        if not target_host:
            raise HTTPException(status_code=400, detail="Target host is required")

        # Create scan target
        from ..domain.tenant_entities import ScanTarget
        target = ScanTarget(
            host=target_host,
            ports=request.get("ports", []),
            scan_profile=request.get("scan_profile", "comprehensive"),
            stealth_mode=request.get("stealth_mode", False)
        )

        # Execute comprehensive African pentest
        pentest_result = await africa_service.conduct_comprehensive_african_pentest(
            target=target,
            test_profile=request.get("test_profile", "comprehensive")
        )

        return {
            "status": "success",
            "scan_id": pentest_result.scan_id,
            "message": "Comprehensive African penetration test completed",
            "results": {
                "regional_context": {
                    "country": pentest_result.regional_context.country,
                    "threat_level": pentest_result.regional_context.threat_level,
                    "primary_threats": pentest_result.regional_context.primary_threats
                },
                "attack_surface": {
                    "domains_discovered": len(pentest_result.attack_surface_analysis.get("domains", [])),
                    "mobile_apps_found": len(pentest_result.attack_surface_analysis.get("mobile_apps", [])),
                    "cloud_assets": len(pentest_result.attack_surface_analysis.get("cloud_assets", []))
                },
                "vulnerabilities_found": len(pentest_result.infrastructure_assessment.get("vulnerabilities", [])),
                "zero_day_indicators": len(pentest_result.zero_day_indicators),
                "social_engineering_vectors": len(pentest_result.social_engineering_vectors),
                "compliance_gaps": len(pentest_result.regulatory_compliance.get("violations", [])),
                "overall_risk_score": pentest_result.business_impact_assessment.get("risk_score", 0),
                "executive_summary": pentest_result.executive_summary
            }
        }

    except Exception as e:
        logger.error(f"Comprehensive African pentest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Penetration test failed: {str(e)}")

@router.post("/mobile-money/security-assessment",
             summary="Mobile Money Security Assessment",
             description="Specialized security testing for African mobile money services")
async def mobile_money_security_assessment(
    request: Dict[str, Any],
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    Comprehensive mobile money security assessment

    Supports:
    - M-Pesa (Kenya)
    - Airtel Money
    - MTN Mobile Money
    - Orange Money
    - Vodacom M-Pesa
    """
    try:
        services = await get_services()
        africa_service = services["africa_pentesting"]

        target_host = request.get("target_host")
        mobile_service = request.get("mobile_service", "mpesa")

        # Create target with mobile money context
        from ..domain.tenant_entities import ScanTarget
        target = ScanTarget(
            host=target_host,
            ports=request.get("ports", [443, 80, 8080, 8443]),
            scan_profile="mobile_money",
            stealth_mode=request.get("stealth_mode", True)
        )

        # Get regional context
        regional_context = await africa_service._gather_regional_threat_intelligence(target)

        # Assess mobile financial security
        mobile_assessment = await africa_service._assess_mobile_financial_security(
            target, regional_context
        )

        return {
            "status": "success",
            "assessment_id": f"mobile_money_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "mobile_service": mobile_service,
            "results": {
                "mobile_money_services": mobile_assessment.get("mobile_money_services", []),
                "api_security": mobile_assessment.get("api_security", []),
                "authentication_analysis": mobile_assessment.get("authentication_mechanisms", []),
                "encryption_analysis": mobile_assessment.get("encryption_analysis", {}),
                "vulnerabilities": mobile_assessment.get("vulnerabilities", []),
                "fraud_vectors": mobile_assessment.get("fraud_vectors", []),
                "compliance_status": mobile_assessment.get("compliance_gaps", []),
                "risk_assessment": {
                    "overall_risk": "medium",  # Would be calculated
                    "financial_impact": "high",
                    "regulatory_risk": "medium"
                }
            }
        }

    except Exception as e:
        logger.error(f"Mobile money security assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

# ====================== AI-POWERED VULNERABILITY ASSESSMENT ======================

@router.post("/ai/vulnerability-prediction",
             summary="AI-Powered Vulnerability Prediction",
             description="Use machine learning to predict potential vulnerabilities")
async def ai_vulnerability_prediction(
    request: Dict[str, Any],
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    Advanced AI-powered vulnerability prediction using machine learning
    """
    try:
        services = await get_services()
        ai_service = services["ai_vulnerability"]

        target_data = request.get("target_data", {})
        analysis_depth = request.get("analysis_depth", "comprehensive")

        # Predict vulnerabilities using AI
        predictions = await ai_service.predict_vulnerabilities(target_data, analysis_depth)

        # Format response
        formatted_predictions = []
        for prediction in predictions:
            formatted_predictions.append({
                "prediction_id": prediction.prediction_id,
                "vulnerability_type": prediction.vulnerability_type,
                "confidence_score": prediction.confidence_score,
                "severity_prediction": prediction.severity_prediction,
                "exploitability_score": prediction.exploitability_score,
                "attack_vectors": prediction.attack_vectors,
                "mitigation_suggestions": prediction.mitigation_suggestions,
                "false_positive_probability": prediction.false_positive_probability
            })

        return {
            "status": "success",
            "predictions_count": len(formatted_predictions),
            "predictions": formatted_predictions,
            "analysis_metadata": {
                "analysis_depth": analysis_depth,
                "ai_models_used": ["vulnerability_prediction_nn", "threat_classifier"],
                "confidence_threshold": 0.7,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"AI vulnerability prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI prediction failed: {str(e)}")

@router.post("/ai/code-analysis",
             summary="AI-Powered Code Analysis",
             description="Advanced code analysis using AI and machine learning")
async def ai_code_analysis(
    request: Dict[str, Any],
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    AI-powered code analysis for vulnerability detection
    """
    try:
        services = await get_services()
        ai_service = services["ai_vulnerability"]

        code_data = {
            "code": request.get("code", ""),
            "language": request.get("language", "python"),
            "context": request.get("context", {})
        }

        if not code_data["code"]:
            raise HTTPException(status_code=400, detail="Code content is required")

        # Analyze code with AI
        analysis_result = await ai_service.analyze_code_with_ai(code_data)

        return {
            "status": "success",
            "analysis_id": analysis_result.analysis_id,
            "results": {
                "language": analysis_result.language,
                "security_score": analysis_result.security_score,
                "vulnerability_patterns": analysis_result.vulnerability_patterns,
                "suggested_fixes": analysis_result.suggested_fixes,
                "ai_confidence": analysis_result.ai_confidence,
                "complexity_metrics": analysis_result.complexity_metrics,
                "control_flow_analysis": analysis_result.control_flow_analysis,
                "data_flow_analysis": analysis_result.data_flow_analysis
            },
            "timestamp": analysis_result.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"AI code analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {str(e)}")

# ====================== ADVANCED EXPLOIT FRAMEWORK ======================

@router.post("/exploit/custom-development",
             summary="Custom Exploit Development",
             description="AI-assisted custom exploit development")
async def custom_exploit_development(
    request: Dict[str, Any],
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    Develop custom exploits using advanced AI assistance
    """
    try:
        services = await get_services()
        exploit_service = services["exploit_framework"]

        vulnerability_info = request.get("vulnerability_info", {})
        if not vulnerability_info:
            raise HTTPException(status_code=400, detail="Vulnerability information is required")

        # Develop custom exploit
        exploit_module = await exploit_service.develop_custom_exploit(vulnerability_info)

        return {
            "status": "success",
            "exploit_module_id": exploit_module.module_id,
            "exploit_info": {
                "module_name": exploit_module.module_name,
                "exploit_type": exploit_module.exploit_type,
                "target_platforms": exploit_module.target_platforms,
                "cvss_score": exploit_module.cvss_score,
                "stealth_rating": exploit_module.stealth_rating,
                "reliability_rating": exploit_module.reliability_rating,
                "payload_templates_count": len(exploit_module.payload_templates),
                "success_indicators": exploit_module.success_indicators
            },
            "creation_date": exploit_module.creation_date.isoformat()
        }

    except Exception as e:
        logger.error(f"Custom exploit development failed: {e}")
        raise HTTPException(status_code=500, detail=f"Exploit development failed: {str(e)}")

@router.post("/exploit/payload-generation",
             summary="Advanced Payload Generation",
             description="Generate advanced payloads with evasion techniques")
async def advanced_payload_generation(
    request: Dict[str, Any],
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    Generate advanced payloads with sophisticated evasion techniques
    """
    try:
        services = await get_services()
        exploit_service = services["exploit_framework"]

        payload_type = request.get("payload_type", "reverse_shell")
        target_info = request.get("target_info", {})
        options = request.get("options", {})

        # Generate advanced payload
        payload = await exploit_service.generate_advanced_payload(
            payload_type, target_info, options
        )

        return {
            "status": "success",
            "payload_id": payload.payload_id,
            "payload_info": {
                "payload_type": payload.payload_type,
                "architecture": payload.architecture,
                "platform": payload.platform,
                "encoding_method": payload.encoding_method,
                "evasion_techniques": payload.evasion_techniques,
                "size_bytes": payload.size_bytes,
                "obfuscation_level": payload.obfuscation_level,
                "success_probability": payload.success_probability,
                "anti_forensics": payload.anti_forensics
            },
            # Note: Not returning actual payload data for security
            "creation_timestamp": payload.creation_timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Advanced payload generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Payload generation failed: {str(e)}")

# ====================== AFRICAN COMPLIANCE AUTOMATION ======================

@router.post("/compliance/african-assessment",
             summary="African Compliance Assessment",
             description="Comprehensive compliance assessment for African regulatory frameworks")
async def african_compliance_assessment(
    request: Dict[str, Any],
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    Comprehensive compliance assessment for African regulatory frameworks

    Supported frameworks:
    - POPIA (South Africa)
    - NDPR (Nigeria)
    - DPA (Kenya, Ghana)
    - CBN Guidelines (Nigeria)
    - SARB Regulations (South Africa)
    """
    try:
        services = await get_services()
        compliance_service = services["compliance"]

        organization_data = request.get("organization_data", {})
        assessment_scope = request.get("assessment_scope", "full")

        if not organization_data:
            raise HTTPException(status_code=400, detail="Organization data is required")

        # Conduct comprehensive compliance assessment
        compliance_report = await compliance_service.conduct_comprehensive_compliance_assessment(
            organization_data, assessment_scope
        )

        return {
            "status": "success",
            "report_id": compliance_report.report_id,
            "assessment_results": {
                "organization_name": compliance_report.organization_name,
                "overall_compliance_score": compliance_report.overall_compliance_score,
                "overall_risk_level": compliance_report.overall_risk_level.value,
                "frameworks_assessed": [f.value for f in compliance_report.frameworks_assessed],
                "assessment_summary": {
                    "total_requirements": len(compliance_report.assessment_results),
                    "compliant": len([r for r in compliance_report.assessment_results
                                   if r.status.value == "compliant"]),
                    "non_compliant": len([r for r in compliance_report.assessment_results
                                        if r.status.value == "non_compliant"]),
                    "partially_compliant": len([r for r in compliance_report.assessment_results
                                              if r.status.value == "partially_compliant"])
                },
                "executive_summary": compliance_report.executive_summary,
                "regulatory_implications": compliance_report.regulatory_implications,
                "remediation_roadmap_items": len(compliance_report.remediation_roadmap),
                "estimated_remediation_cost": compliance_report.cost_analysis.get("total_cost", 0)
            },
            "next_assessment_date": compliance_report.next_assessment_date.isoformat(),
            "report_generation_date": compliance_report.report_generation_date.isoformat()
        }

    except Exception as e:
        logger.error(f"African compliance assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance assessment failed: {str(e)}")

@router.get("/compliance/frameworks",
            summary="Get Available Compliance Frameworks",
            description="Get list of supported African compliance frameworks")
async def get_compliance_frameworks(
    current_user = Depends(get_current_user)
):
    """
    Get list of supported African compliance frameworks
    """
    try:
        services = await get_services()
        compliance_service = services["compliance"]

        frameworks = []
        for framework, details in compliance_service.african_frameworks.items():
            frameworks.append({
                "framework_id": framework.value,
                "name": details["name"],
                "country": details["country"],
                "authority": details["authority"],
                "effective_date": details["effective_date"].isoformat(),
                "scope": details["scope"],
                "update_frequency": details["update_frequency"]
            })

        return {
            "status": "success",
            "frameworks_count": len(frameworks),
            "frameworks": frameworks
        }

    except Exception as e:
        logger.error(f"Get compliance frameworks failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get frameworks: {str(e)}")

# ====================== THREAT HUNTING AND INCIDENT RESPONSE ======================

@router.post("/threat-hunting/execute-hunt",
             summary="Execute Threat Hunt",
             description="Execute proactive threat hunting with custom queries")
async def execute_threat_hunt(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    Execute proactive threat hunting with advanced analytics
    """
    try:
        services = await get_services()
        hunting_service = services["threat_hunting"]

        hunt_config = {
            "hunt_name": request.get("hunt_name", "Custom Hunt"),
            "hypothesis": request.get("hypothesis", ""),
            "hunt_query": request.get("hunt_query", ""),
            "data_sources": request.get("data_sources", ["logs", "network", "endpoints"]),
            "start_time": datetime.fromisoformat(request.get("start_time",
                                               (datetime.now() - timedelta(hours=24)).isoformat())),
            "end_time": datetime.fromisoformat(request.get("end_time", datetime.now().isoformat())),
            "analyst": current_user.username if hasattr(current_user, 'username') else "api_user"
        }

        # Execute threat hunt
        threat_hunt = await hunting_service.execute_threat_hunt(hunt_config)

        return {
            "status": "success",
            "hunt_id": threat_hunt.hunt_id,
            "hunt_results": {
                "hunt_name": threat_hunt.hunt_name,
                "hunt_status": threat_hunt.hunt_status,
                "findings_count": len(threat_hunt.findings),
                "findings": [
                    {
                        "detection_id": f.detection_id,
                        "title": f.title,
                        "severity": f.severity.value,
                        "confidence_score": f.confidence_score,
                        "affected_entities": f.affected_entities,
                        "mitre_techniques": f.mitre_techniques
                    } for f in threat_hunt.findings[:10]  # Limit to first 10
                ],
                "data_sources_queried": threat_hunt.data_sources,
                "time_range": {
                    "start": threat_hunt.time_range["start"].isoformat(),
                    "end": threat_hunt.time_range["end"].isoformat()
                },
                "analyst": threat_hunt.analyst,
                "created_time": threat_hunt.created_time.isoformat(),
                "completed_time": threat_hunt.completed_time.isoformat() if threat_hunt.completed_time else None
            }
        }

    except Exception as e:
        logger.error(f"Threat hunt execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Threat hunt failed: {str(e)}")

@router.post("/incident-response/automated",
             summary="Automated Incident Response",
             description="Trigger automated incident response workflow")
async def automated_incident_response(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    current_org = Depends(get_current_org)
):
    """
    Trigger automated incident response workflow
    """
    try:
        services = await get_services()
        hunting_service = services["threat_hunting"]

        # Create detection object from request
        from ..services.advanced_threat_hunting_system import ThreatDetection, ThreatSeverity, DetectionType

        detection = ThreatDetection(
            detection_id=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=request.get("title", "Security Event"),
            description=request.get("description", ""),
            severity=ThreatSeverity(request.get("severity", "medium")),
            detection_type=DetectionType(request.get("detection_type", "custom_rule")),
            confidence_score=request.get("confidence_score", 0.8),
            raw_data=request.get("raw_data", {}),
            affected_entities=request.get("affected_entities", []),
            indicators_of_compromise=request.get("iocs", []),
            mitre_techniques=request.get("mitre_techniques", []),
            threat_actor_attribution=request.get("threat_actor"),
            detection_time=datetime.now(),
            source_system=request.get("source_system", "api"),
            rule_id=request.get("rule_id"),
            false_positive_probability=request.get("false_positive_probability", 0.1)
        )

        # Execute automated incident response
        incident = await hunting_service.automated_incident_response(detection)

        return {
            "status": "success",
            "incident_id": incident.incident_id,
            "incident_details": {
                "title": incident.title,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "affected_systems": incident.affected_systems,
                "containment_actions_taken": len(incident.containment_actions),
                "evidence_collected": len(incident.evidence_collected),
                "assigned_analyst": incident.assigned_analyst,
                "created_time": incident.created_time.isoformat(),
                "attack_timeline_events": len(incident.attack_timeline)
            },
            "response_actions": {
                "containment_executed": len(incident.containment_actions) > 0,
                "evidence_collection_started": len(incident.evidence_collected) > 0,
                "automated_analysis_completed": True
            }
        }

    except Exception as e:
        logger.error(f"Automated incident response failed: {e}")
        raise HTTPException(status_code=500, detail=f"Incident response failed: {str(e)}")

# ====================== SYSTEM STATUS AND HEALTH ======================

@router.get("/health",
            summary="Advanced PTaaS Health Check",
            description="Get health status of all advanced PTaaS services")
async def advanced_ptaas_health():
    """
    Get comprehensive health status of all advanced PTaaS services
    """
    try:
        services = await get_services()

        health_status = {}

        # Check each service
        for service_name, service in services.items():
            if hasattr(service, 'health_check'):
                service_health = await service.health_check()
                health_status[service_name] = {
                    "status": service_health.status.value,
                    "checks": service_health.checks,
                    "timestamp": service_health.timestamp.isoformat()
                }
            else:
                # Implement default health check for unknown services
                health_status[service_name] = await _check_service_health_default(service_name)

        # Overall system status
        all_statuses = [status["status"] for status in health_status.values()]
        overall_status = "healthy" if all(s == "healthy" for s in all_statuses) else "degraded"

        return {
            "overall_status": overall_status,
            "services": health_status,
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "african_pentesting": True,
                "ai_vulnerability_assessment": True,
                "advanced_exploit_framework": True,
                "african_compliance_automation": True,
                "threat_hunting_and_response": True
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/capabilities",
            summary="Get Advanced PTaaS Capabilities",
            description="Get detailed information about advanced PTaaS capabilities")
async def get_advanced_capabilities():
    """
    Get detailed information about advanced PTaaS capabilities
    """
    try:
        return {
            "status": "success",
            "advanced_capabilities": {
                "african_penetration_testing": {
                    "description": "Specialized penetration testing for African markets",
                    "features": [
                        "Regional threat intelligence",
                        "Mobile money security testing",
                        "Telecommunications assessment",
                        "Government sector testing",
                        "Cultural context social engineering",
                        "Mining sector security",
                        "Cross-border financial testing"
                    ],
                    "supported_countries": [
                        "South Africa", "Nigeria", "Kenya", "Ghana", "Egypt",
                        "Morocco", "Tanzania", "Uganda", "Ethiopia"
                    ]
                },
                "ai_vulnerability_assessment": {
                    "description": "AI-powered vulnerability discovery and analysis",
                    "features": [
                        "Machine learning vulnerability prediction",
                        "Deep learning code analysis",
                        "AI threat correlation",
                        "Natural language processing",
                        "Computer vision for GUI testing",
                        "Reinforcement learning optimization",
                        "Behavioral analysis",
                        "Zero-day discovery assistance"
                    ]
                },
                "advanced_exploit_framework": {
                    "description": "Sophisticated exploit development and execution",
                    "features": [
                        "Custom exploit development",
                        "Advanced payload generation",
                        "Memory corruption exploitation",
                        "Post-exploitation frameworks",
                        "Advanced persistence mechanisms",
                        "Lateral movement techniques",
                        "Anti-forensics capabilities",
                        "Stealth and evasion techniques"
                    ]
                },
                "african_compliance_automation": {
                    "description": "Automated compliance for African regulatory frameworks",
                    "features": [
                        "POPIA compliance (South Africa)",
                        "NDPR compliance (Nigeria)",
                        "DPA compliance (Kenya, Ghana)",
                        "CBN Guidelines (Nigeria)",
                        "SARB Regulations (South Africa)",
                        "Automated evidence collection",
                        "Multi-format reporting",
                        "Real-time monitoring"
                    ]
                },
                "threat_hunting_and_response": {
                    "description": "Advanced threat hunting and incident response",
                    "features": [
                        "Custom query language",
                        "Machine learning anomaly detection",
                        "Automated incident response",
                        "Digital forensics",
                        "Behavioral analytics",
                        "Threat intelligence correlation",
                        "Attack path reconstruction",
                        "Timeline analysis"
                    ]
                }
            },
            "integration_capabilities": [
                "RESTful API integration",
                "Webhook notifications",
                "SIEM integration",
                "EDR platform integration",
                "Threat intelligence feeds",
                "Custom dashboard integration"
            ],
            "supported_formats": [
                "JSON", "XML", "PDF", "Excel", "CSV", "STIX/TAXII"
            ]
        }

    except Exception as e:
        logger.error(f"Get capabilities failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

# Export the router
__all__ = ["router"]
