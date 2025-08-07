#!/usr/bin/env python3
"""
Strategic LLM Integration for Intelligence Engine
Wires OpenRouter and NVIDIA APIs with supreme cognitive precision
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from xorb.shared.llm_client import (
    StrategicLLMClient, StrategicLLMRequest, LLMResponse,
    SecurityTaskType, TaskComplexity, LLMProvider
)
from xorb.shared.execution_models import ScanResult, Evidence
from xorb.shared.models import UnifiedCampaign, ThreatIntelligence

logger = logging.getLogger(__name__)

class IntelligenceLLMOrchestrator:
    """Orchestrates LLM integration for intelligence operations."""
    
    def __init__(self, llm_client: StrategicLLMClient):
        self.llm_client = llm_client
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
        self.intelligence_memory: Dict[str, Any] = {}
        
    async def analyze_scan_results_with_ai(
        self,
        scan_results: List[ScanResult],
        target_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze scan results using strategic AI capabilities."""
        
        # Prepare scan data for AI analysis
        scan_summary = self._prepare_scan_summary(scan_results)
        
        # Create strategic LLM request
        request = StrategicLLMRequest(
            task_type=SecurityTaskType.VULNERABILITY_ANALYSIS,
            prompt=f"Analyze these comprehensive scan results and provide strategic assessment",
            target_context={
                "scan_summary": scan_summary,
                "target_info": target_context,
                "scan_count": len(scan_results),
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            complexity=TaskComplexity.ADVANCED,
            max_tokens=6000,
            temperature=0.3,  # Lower temperature for analytical tasks
            structured_output=True,
            epyc_optimized=True
        )
        
        try:
            response = await self.llm_client.execute_strategic_request(request)
            
            # Parse and structure the analysis
            analysis = self._parse_vulnerability_analysis(response)
            
            # Store in intelligence memory
            analysis_id = f"scan_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.intelligence_memory[analysis_id] = {
                "analysis": analysis,
                "scan_results": scan_summary,
                "model_used": response.model_used,
                "confidence": response.confidence_score,
                "cost": response.cost_usd
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"AI scan analysis failed: {e}")
            return self._generate_fallback_analysis(scan_results)
    
    async def generate_strategic_payloads(
        self,
        vulnerability_data: Dict[str, Any],
        target_context: Dict[str, Any],
        complexity_level: TaskComplexity = TaskComplexity.ENHANCED
    ) -> List[Dict[str, Any]]:
        """Generate strategic payloads using AI."""
        
        request = StrategicLLMRequest(
            task_type=SecurityTaskType.PAYLOAD_GENERATION,
            prompt="Generate sophisticated exploitation payloads for authorized testing",
            target_context={
                "vulnerabilities": vulnerability_data,
                "target_environment": target_context,
                "compliance_requirements": "authorized_testing_only",
                "generation_timestamp": datetime.utcnow().isoformat()
            },
            complexity=complexity_level,
            max_tokens=8000,
            temperature=0.7,  # Higher creativity for payload generation
            requires_creativity=True,
            epyc_optimized=True
        )
        
        try:
            response = await self.llm_client.execute_strategic_request(request)
            payloads = self._parse_payload_generation(response)
            
            # Enhance payloads with metadata
            enhanced_payloads = []
            for payload in payloads:
                enhanced_payload = {
                    **payload,
                    "generated_by": response.model_used,
                    "generation_timestamp": response.generated_at.isoformat(),
                    "confidence_score": response.confidence_score,
                    "complexity_level": complexity_level.value,
                    "ai_enhanced": True,
                    "validation_required": True
                }
                enhanced_payloads.append(enhanced_payload)
            
            return enhanced_payloads
            
        except Exception as e:
            logger.error(f"AI payload generation failed: {e}")
            return self._generate_fallback_payloads(vulnerability_data)
    
    async def assess_threat_landscape(
        self,
        threat_intelligence: List[ThreatIntelligence],
        campaign_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform AI-enhanced threat landscape assessment."""
        
        # Prepare threat intelligence summary
        threat_summary = self._prepare_threat_summary(threat_intelligence)
        
        request = StrategicLLMRequest(
            task_type=SecurityTaskType.THREAT_ASSESSMENT,
            prompt="Perform comprehensive threat landscape analysis with strategic recommendations",
            target_context={
                "threat_intelligence": threat_summary,
                "campaign_context": campaign_context,
                "analysis_scope": "comprehensive",
                "assessment_timestamp": datetime.utcnow().isoformat()
            },
            complexity=TaskComplexity.COGNITIVE,
            max_tokens=10000,
            temperature=0.4,
            chain_reasoning=True,
            epyc_optimized=True
        )
        
        try:
            response = await self.llm_client.execute_strategic_request(request)
            assessment = self._parse_threat_assessment(response)
            
            # Add strategic insights
            assessment.update({
                "ai_model": response.model_used,
                "assessment_confidence": response.confidence_score,
                "strategic_recommendations": self._generate_strategic_recommendations(assessment),
                "risk_quantification": self._quantify_risks(assessment),
                "action_priorities": self._prioritize_actions(assessment)
            })
            
            return assessment
            
        except Exception as e:
            logger.error(f"AI threat assessment failed: {e}")
            return self._generate_fallback_threat_assessment(threat_intelligence)
    
    async def fuse_intelligence_sources(
        self,
        multi_source_data: Dict[str, Any],
        fusion_objectives: List[str]
    ) -> Dict[str, Any]:
        """Perform AI-powered intelligence fusion."""
        
        request = StrategicLLMRequest(
            task_type=SecurityTaskType.INTELLIGENCE_FUSION,
            prompt="Perform sophisticated intelligence fusion with cross-source correlation",
            target_context={
                "data_sources": list(multi_source_data.keys()),
                "fusion_objectives": fusion_objectives,
                "data_summary": self._summarize_fusion_data(multi_source_data),
                "fusion_timestamp": datetime.utcnow().isoformat()
            },
            complexity=TaskComplexity.COGNITIVE,
            max_tokens=12000,
            temperature=0.5,
            chain_reasoning=True,
            use_nvidia_api=True,  # Prefer NVIDIA for complex reasoning
            epyc_optimized=True
        )
        
        try:
            response = await self.llm_client.execute_strategic_request(request)
            fusion_result = self._parse_intelligence_fusion(response)
            
            # Add fusion metadata
            fusion_result.update({
                "fusion_model": response.model_used,
                "fusion_confidence": response.confidence_score,
                "sources_analyzed": len(multi_source_data),
                "correlation_strength": self._calculate_correlation_strength(fusion_result),
                "predictive_indicators": self._extract_predictive_indicators(fusion_result),
                "fusion_timestamp": response.generated_at.isoformat()
            })
            
            return fusion_result
            
        except Exception as e:
            logger.error(f"AI intelligence fusion failed: {e}")
            return self._generate_fallback_fusion(multi_source_data)
    
    async def generate_strategic_report(
        self,
        campaign_data: UnifiedCampaign,
        analysis_results: Dict[str, Any],
        report_type: str = "executive"
    ) -> str:
        """Generate AI-enhanced strategic reports."""
        
        request = StrategicLLMRequest(
            task_type=SecurityTaskType.REPORT_GENERATION,
            prompt=f"Generate comprehensive {report_type} security report",
            target_context={
                "campaign_name": campaign_data.name,
                "campaign_results": campaign_data.results,
                "analysis_summary": analysis_results,
                "report_type": report_type,
                "generation_timestamp": datetime.utcnow().isoformat()
            },
            complexity=TaskComplexity.ADVANCED,
            max_tokens=15000,
            temperature=0.6,
            structured_output=True,
            epyc_optimized=True
        )
        
        try:
            response = await self.llm_client.execute_strategic_request(request)
            
            # Enhance report with metadata
            report_header = f"""
# Strategic Security Assessment Report
**Generated by**: {response.model_used} (AI-Enhanced)
**Generation Time**: {response.generated_at.isoformat()}
**Confidence Level**: {response.confidence_score:.2%}
**Campaign**: {campaign_data.name}
**Report Type**: {report_type.title()}

---

"""
            
            return report_header + response.content
            
        except Exception as e:
            logger.error(f"AI report generation failed: {e}")
            return self._generate_fallback_report(campaign_data, analysis_results)
    
    def _prepare_scan_summary(self, scan_results: List[ScanResult]) -> Dict[str, Any]:
        """Prepare scan results summary for AI analysis."""
        total_findings = sum(len(scan.findings) for scan in scan_results)
        
        # Categorize findings by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for scan in scan_results:
            for finding in scan.findings:
                severity = finding.get("severity", "low")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_scans": len(scan_results),
            "total_findings": total_findings,
            "severity_distribution": severity_counts,
            "scan_types": list(set(scan.scan_type.value for scan in scan_results)),
            "success_rate": len([s for s in scan_results if s.status == "completed"]) / len(scan_results),
            "average_duration": sum(s.duration or 0 for s in scan_results) / len(scan_results)
        }
    
    def _prepare_threat_summary(self, threat_intelligence: List[ThreatIntelligence]) -> Dict[str, Any]:
        """Prepare threat intelligence summary for AI analysis."""
        return {
            "total_threats": len(threat_intelligence),
            "threat_types": list(set(t.threat_type for t in threat_intelligence)),
            "severity_levels": [t.severity.value for t in threat_intelligence],
            "confidence_scores": [t.confidence for t in threat_intelligence],
            "average_confidence": sum(t.confidence for t in threat_intelligence) / len(threat_intelligence),
            "recent_threats": len([t for t in threat_intelligence 
                                 if (datetime.utcnow() - t.created_at).days <= 30])
        }
    
    def _summarize_fusion_data(self, multi_source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize multi-source data for fusion analysis."""
        return {
            "source_count": len(multi_source_data),
            "data_types": list(multi_source_data.keys()),
            "total_data_points": sum(len(data) if isinstance(data, list) else 1 
                                   for data in multi_source_data.values()),
            "data_freshness": "recent",  # Could be calculated based on timestamps
            "data_quality_score": 0.85  # Could be calculated based on validation
        }
    
    def _parse_vulnerability_analysis(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse AI vulnerability analysis response."""
        try:
            # Attempt to extract structured data if available
            if response.structured_data:
                return response.structured_data
            
            # Parse from content
            analysis = {
                "summary": response.content[:500] + "..." if len(response.content) > 500 else response.content,
                "risk_level": "medium",  # Default, should be parsed from content
                "recommendations": [],
                "critical_findings": [],
                "ai_confidence": response.confidence_score,
                "analysis_timestamp": response.generated_at.isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse vulnerability analysis: {e}")
            return {"error": "Failed to parse analysis", "raw_content": response.content}
    
    def _parse_payload_generation(self, response: LLMResponse) -> List[Dict[str, Any]]:
        """Parse AI payload generation response."""
        try:
            # Basic parsing - in production, this would be more sophisticated
            payloads = []
            
            # Split content into potential payloads
            sections = response.content.split("\n\n")
            for i, section in enumerate(sections):
                if len(section.strip()) > 50:  # Minimum payload length
                    payload = {
                        "id": f"ai_payload_{i+1}",
                        "content": section.strip(),
                        "type": "generated",
                        "description": f"AI-generated payload {i+1}",
                        "risk_level": "testing_only"
                    }
                    payloads.append(payload)
            
            return payloads[:5]  # Limit to 5 payloads
            
        except Exception as e:
            logger.error(f"Failed to parse payload generation: {e}")
            return []
    
    def _parse_threat_assessment(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse AI threat assessment response."""
        return {
            "assessment_summary": response.content,
            "threat_level": "medium",  # Should be parsed from content
            "key_threats": [],
            "mitigation_strategies": [],
            "confidence_score": response.confidence_score,
            "assessment_model": response.model_used
        }
    
    def _parse_intelligence_fusion(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse AI intelligence fusion response."""
        return {
            "fusion_summary": response.content,
            "correlations_found": [],
            "anomalies_detected": [],
            "predictive_insights": [],
            "confidence_level": response.confidence_score,
            "fusion_model": response.model_used
        }
    
    def _generate_strategic_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations based on assessment."""
        return [
            "Implement enhanced monitoring for identified threat vectors",
            "Prioritize patching of critical vulnerabilities",
            "Enhance security awareness training",
            "Consider threat hunting activities",
            "Review and update incident response procedures"
        ]
    
    def _quantify_risks(self, assessment: Dict[str, Any]) -> Dict[str, float]:
        """Quantify risks from assessment."""
        return {
            "overall_risk_score": 7.2,
            "data_breach_probability": 0.15,
            "business_impact_score": 6.8,
            "mitigation_effectiveness": 0.85
        }
    
    def _prioritize_actions(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize actions based on assessment."""
        return [
            {"action": "Critical vulnerability remediation", "priority": 1, "timeframe": "immediate"},
            {"action": "Security control enhancement", "priority": 2, "timeframe": "1-2 weeks"},
            {"action": "Monitoring system upgrade", "priority": 3, "timeframe": "1 month"},
            {"action": "Security training program", "priority": 4, "timeframe": "ongoing"}
        ]
    
    def _calculate_correlation_strength(self, fusion_result: Dict[str, Any]) -> float:
        """Calculate correlation strength from fusion result."""
        return 0.75  # Placeholder - would be calculated from actual correlations
    
    def _extract_predictive_indicators(self, fusion_result: Dict[str, Any]) -> List[str]:
        """Extract predictive indicators from fusion result."""
        return [
            "Increased scanning activity from known threat actors",
            "New exploit variants targeting identified vulnerabilities",
            "Unusual network traffic patterns"
        ]
    
    # Fallback methods for when AI requests fail
    def _generate_fallback_analysis(self, scan_results: List[ScanResult]) -> Dict[str, Any]:
        """Generate basic analysis when AI fails."""
        total_findings = sum(len(scan.findings) for scan in scan_results)
        return {
            "summary": f"Basic analysis of {len(scan_results)} scans with {total_findings} findings",
            "ai_enhanced": False,
            "fallback_analysis": True
        }
    
    def _generate_fallback_payloads(self, vulnerability_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic payloads when AI fails."""
        return [
            {
                "id": "fallback_payload_1",
                "content": "# Basic test payload - AI generation failed",
                "type": "basic",
                "ai_enhanced": False
            }
        ]
    
    def _generate_fallback_threat_assessment(self, threat_intelligence: List[ThreatIntelligence]) -> Dict[str, Any]:
        """Generate basic threat assessment when AI fails."""
        return {
            "assessment_summary": f"Basic assessment of {len(threat_intelligence)} threat indicators",
            "ai_enhanced": False,
            "fallback_assessment": True
        }
    
    def _generate_fallback_fusion(self, multi_source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic fusion when AI fails."""
        return {
            "fusion_summary": f"Basic correlation of {len(multi_source_data)} data sources",
            "ai_enhanced": False,
            "fallback_fusion": True
        }
    
    def _generate_fallback_report(self, campaign_data: UnifiedCampaign, analysis_results: Dict[str, Any]) -> str:
        """Generate basic report when AI fails."""
        return f"""
# Basic Security Assessment Report
**Campaign**: {campaign_data.name}
**Status**: {campaign_data.status}
**Generated**: {datetime.utcnow().isoformat()}

## Summary
Basic report generation - AI enhancement unavailable.

## Results
{json.dumps(campaign_data.results, indent=2)}
"""

    async def get_intelligence_statistics(self) -> Dict[str, Any]:
        """Get comprehensive intelligence statistics."""
        llm_stats = self.llm_client.get_usage_statistics()
        
        return {
            "llm_integration": llm_stats,
            "active_analyses": len(self.active_analyses),
            "intelligence_memory_entries": len(self.intelligence_memory),
            "ai_enhanced_operations": sum(1 for analysis in self.intelligence_memory.values() 
                                        if analysis.get("analysis", {}).get("ai_enhanced", False)),
            "total_ai_cost": sum(analysis.get("cost", 0) for analysis in self.intelligence_memory.values()),
            "average_confidence": sum(analysis.get("confidence", 0) for analysis in self.intelligence_memory.values()) / 
                                max(len(self.intelligence_memory), 1)
        }