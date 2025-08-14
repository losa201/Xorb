"""
Enhanced Threat Intelligence Service
Principal Auditor Implementation: Strategic Integration Layer

This service integrates the Advanced Threat Intelligence Fusion Engine
with the existing PTaaS platform, providing enhanced threat context
and intelligence-driven security scanning capabilities.

Key Features:
- Real-time threat intelligence integration with PTaaS scans
- Intelligence-driven scan prioritization and targeting
- Contextual threat analysis for scan results
- Automated threat hunting query generation
- Enhanced vulnerability assessment with threat context
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import Depends

# Internal imports
from .base_service import SecurityService, ServiceHealth, ServiceStatus
from .interfaces import IntelligenceService, PTaaSService
from .ptaas_scanner_service import SecurityScannerService, get_scanner_service
from ..domain.tenant_entities import ScanTarget, ScanResult, SecurityFinding
from ...xorb.intelligence.advanced_threat_intelligence_fusion_engine import (
    get_threat_intelligence_fusion, 
    AdvancedThreatIntelligenceFusion,
    GlobalThreatIndicator,
    ThreatSeverity,
    ThreatCategory
)

logger = logging.getLogger(__name__)


class EnhancedThreatIntelligenceService(SecurityService):
    """Enhanced threat intelligence service with fusion engine integration"""
    
    def __init__(self, **kwargs):
        super().__init__(
            service_id="enhanced_threat_intelligence",
            dependencies=["threat_fusion_engine", "ptaas_scanner"],
            config=kwargs.get("config", {})
        )
        
        self.fusion_engine: Optional[AdvancedThreatIntelligenceFusion] = None
        self.scanner_service: Optional[SecurityScannerService] = None
        
        # Intelligence cache for performance
        self.intelligence_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # Threat hunting queries
        self.generated_queries: List[Dict[str, Any]] = []
        
    async def initialize(self) -> bool:
        """Initialize the enhanced threat intelligence service"""
        try:
            logger.info("Initializing Enhanced Threat Intelligence Service...")
            
            # Get fusion engine instance
            self.fusion_engine = await get_threat_intelligence_fusion()
            
            # Get scanner service instance
            self.scanner_service = await get_scanner_service()
            
            # Start background tasks
            asyncio.create_task(self._intelligence_update_loop())
            asyncio.create_task(self._cache_cleanup_loop())
            
            logger.info("Enhanced Threat Intelligence Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced threat intelligence service: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the service gracefully"""
        try:
            # Clear cache
            self.intelligence_cache.clear()
            
            logger.info("Enhanced Threat Intelligence Service shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown enhanced threat intelligence service: {e}")
            return False
    
    async def enhance_scan_with_intelligence(self, 
                                           scan_target: ScanTarget,
                                           scan_result: ScanResult) -> Dict[str, Any]:
        """Enhance scan results with threat intelligence context"""
        enhancement_result = {
            "scan_id": scan_result.scan_id,
            "target": scan_target.host,
            "intelligence_enhancement": {
                "threat_indicators_found": [],
                "contextual_threats": [],
                "risk_assessment": {},
                "recommendations": [],
                "threat_hunting_queries": []
            },
            "enhanced_vulnerabilities": [],
            "threat_landscape_context": {}
        }
        
        try:
            # Analyze scan target against threat intelligence
            target_intelligence = await self._analyze_target_intelligence(scan_target)
            enhancement_result["intelligence_enhancement"]["threat_indicators_found"] = target_intelligence
            
            # Enhance vulnerabilities with threat context
            enhanced_vulns = await self._enhance_vulnerabilities_with_intelligence(
                scan_result.vulnerabilities, 
                scan_target.host
            )
            enhancement_result["enhanced_vulnerabilities"] = enhanced_vulns
            
            # Generate contextual threat assessment
            threat_context = await self._generate_threat_context(scan_target, scan_result)
            enhancement_result["intelligence_enhancement"]["contextual_threats"] = threat_context
            
            # Perform risk assessment with intelligence
            risk_assessment = await self._assess_risk_with_intelligence(
                scan_target, 
                scan_result, 
                target_intelligence
            )
            enhancement_result["intelligence_enhancement"]["risk_assessment"] = risk_assessment
            
            # Generate threat hunting queries
            hunting_queries = await self._generate_threat_hunting_queries(
                scan_target, 
                scan_result, 
                target_intelligence
            )
            enhancement_result["intelligence_enhancement"]["threat_hunting_queries"] = hunting_queries
            
            # Get current threat landscape context
            landscape_context = await self._get_threat_landscape_context()
            enhancement_result["threat_landscape_context"] = landscape_context
            
            # Generate enhanced recommendations
            recommendations = await self._generate_intelligence_recommendations(
                scan_target,
                scan_result,
                target_intelligence,
                threat_context,
                risk_assessment
            )
            enhancement_result["intelligence_enhancement"]["recommendations"] = recommendations
            
            logger.info(f"Enhanced scan {scan_result.scan_id} with threat intelligence: "
                       f"{len(target_intelligence)} indicators, {len(enhanced_vulns)} enhanced vulnerabilities")
            
        except Exception as e:
            logger.error(f"Failed to enhance scan with intelligence: {e}")
            enhancement_result["error"] = str(e)
        
        return enhancement_result
    
    async def _analyze_target_intelligence(self, scan_target: ScanTarget) -> List[Dict[str, Any]]:
        """Analyze scan target against threat intelligence indicators"""
        threat_indicators = []
        
        try:
            if not self.fusion_engine:
                return threat_indicators
            
            # Check cache first
            cache_key = f"target_intel_{scan_target.host}"
            if cache_key in self.intelligence_cache:
                cached_data = self.intelligence_cache[cache_key]
                if datetime.utcnow() - cached_data["timestamp"] < self.cache_ttl:
                    return cached_data["indicators"]
            
            # Search for indicators related to target
            search_queries = [
                {"value": scan_target.host},
                {"type": "ip"},
                {"type": "domain"}
            ]
            
            for query in search_queries:
                indicators = await self.fusion_engine.search_indicators(query, limit=50)
                
                for indicator in indicators:
                    # Check if indicator matches target
                    if self._indicator_matches_target(indicator, scan_target):
                        threat_data = {
                            "indicator_id": indicator.indicator_id,
                            "indicator_type": indicator.indicator_type,
                            "value": indicator.value,
                            "severity": indicator.severity.value,
                            "category": indicator.category.value,
                            "confidence": indicator.confidence,
                            "first_seen": indicator.first_seen.isoformat(),
                            "last_seen": indicator.last_seen.isoformat(),
                            "sources": indicator.sources,
                            "attributed_actors": [actor.value for actor in indicator.attributed_actors],
                            "mitre_techniques": indicator.mitre_techniques,
                            "tags": indicator.tags,
                            "context": self._extract_relevant_context(indicator, scan_target)
                        }
                        threat_indicators.append(threat_data)
            
            # Cache results
            self.intelligence_cache[cache_key] = {
                "timestamp": datetime.utcnow(),
                "indicators": threat_indicators
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze target intelligence: {e}")
        
        return threat_indicators
    
    async def _enhance_vulnerabilities_with_intelligence(self, 
                                                       vulnerabilities: List[Dict[str, Any]],
                                                       target_host: str) -> List[Dict[str, Any]]:
        """Enhance vulnerabilities with threat intelligence context"""
        enhanced_vulnerabilities = []
        
        try:
            for vuln in vulnerabilities:
                enhanced_vuln = vuln.copy()
                
                # Add threat intelligence context
                intel_context = await self._get_vulnerability_intelligence_context(vuln, target_host)
                enhanced_vuln["threat_intelligence"] = intel_context
                
                # Calculate enhanced risk score
                enhanced_risk = await self._calculate_enhanced_risk_score(vuln, intel_context)
                enhanced_vuln["enhanced_risk_score"] = enhanced_risk
                
                # Add exploitation probability
                exploit_prob = await self._calculate_exploitation_probability(vuln, intel_context)
                enhanced_vuln["exploitation_probability"] = exploit_prob
                
                # Add threat actor attribution
                actor_attribution = await self._get_threat_actor_attribution(vuln, intel_context)
                enhanced_vuln["threat_actor_attribution"] = actor_attribution
                
                enhanced_vulnerabilities.append(enhanced_vuln)
            
        except Exception as e:
            logger.error(f"Failed to enhance vulnerabilities with intelligence: {e}")
        
        return enhanced_vulnerabilities
    
    async def _generate_threat_context(self, 
                                     scan_target: ScanTarget, 
                                     scan_result: ScanResult) -> List[Dict[str, Any]]:
        """Generate contextual threat information for the scan"""
        contextual_threats = []
        
        try:
            # Get current threat landscape
            if self.fusion_engine and self.fusion_engine.threat_landscape:
                landscape = self.fusion_engine.threat_landscape
                
                # Analyze target against top threats
                for threat in landscape.top_threats:
                    threat_category = threat.get("category", "")
                    
                    # Check if scan results indicate this threat category
                    if self._scan_indicates_threat_category(scan_result, threat_category):
                        contextual_threat = {
                            "threat_category": threat_category,
                            "threat_score": threat.get("threat_score", 0),
                            "indicators_in_scan": self._count_threat_indicators_in_scan(scan_result, threat_category),
                            "relevance": "high" if threat.get("threat_score", 0) > 50 else "medium",
                            "description": f"Current threat landscape shows elevated {threat_category} activity",
                            "mitigation_priority": "high" if threat.get("threat_score", 0) > 80 else "medium"
                        }
                        contextual_threats.append(contextual_threat)
                
                # Analyze against emerging threats
                for emerging_threat in landscape.emerging_threats:
                    if self._scan_relates_to_emerging_threat(scan_result, emerging_threat):
                        contextual_threat = {
                            "threat_type": "emerging",
                            "threat_category": emerging_threat.get("category", ""),
                            "severity": emerging_threat.get("severity", ""),
                            "confidence": emerging_threat.get("confidence", 0),
                            "first_seen": emerging_threat.get("first_seen", ""),
                            "description": f"Emerging threat activity detected in current threat landscape",
                            "mitigation_priority": "critical" if emerging_threat.get("severity") == "critical" else "high"
                        }
                        contextual_threats.append(contextual_threat)
            
        except Exception as e:
            logger.error(f"Failed to generate threat context: {e}")
        
        return contextual_threats
    
    async def _assess_risk_with_intelligence(self, 
                                           scan_target: ScanTarget,
                                           scan_result: ScanResult, 
                                           threat_indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk with threat intelligence enhancement"""
        risk_assessment = {
            "overall_risk_score": 0.0,
            "risk_level": "low",
            "risk_factors": [],
            "threat_probability": 0.0,
            "impact_assessment": {},
            "confidence": 0.0
        }
        
        try:
            risk_score = 0.0
            risk_factors = []
            
            # Base risk from vulnerabilities
            vuln_count = len(scan_result.vulnerabilities)
            critical_vulns = len([v for v in scan_result.vulnerabilities if v.get("severity") == "critical"])
            high_vulns = len([v for v in scan_result.vulnerabilities if v.get("severity") == "high"])
            
            base_vuln_score = min(0.4, (critical_vulns * 0.1 + high_vulns * 0.05))
            risk_score += base_vuln_score
            
            if critical_vulns > 0:
                risk_factors.append(f"{critical_vulns} critical vulnerabilities identified")
            if high_vulns > 0:
                risk_factors.append(f"{high_vulns} high-severity vulnerabilities identified")
            
            # Intelligence-based risk enhancement
            intel_risk_boost = 0.0
            
            for indicator in threat_indicators:
                indicator_severity = indicator.get("severity", "low")
                indicator_confidence = indicator.get("confidence", 0.5)
                
                # Risk boost based on severity and confidence
                severity_multiplier = {
                    "critical": 0.3,
                    "high": 0.2,
                    "medium": 0.1,
                    "low": 0.05,
                    "info": 0.02
                }.get(indicator_severity, 0.05)
                
                indicator_risk = severity_multiplier * indicator_confidence
                intel_risk_boost += indicator_risk
                
                # Add specific risk factors
                if indicator_severity in ["critical", "high"]:
                    risk_factors.append(f"Target matches {indicator_severity} threat indicator: {indicator.get('category', 'unknown')}")
                
                # Actor attribution risk
                if indicator.get("attributed_actors"):
                    for actor in indicator["attributed_actors"]:
                        if actor in ["NATION_STATE", "ORGANIZED_CRIME", "APT_GROUP"]:
                            intel_risk_boost += 0.1
                            risk_factors.append(f"Target associated with {actor} threat actor")
            
            risk_score += min(0.4, intel_risk_boost)
            
            # Network exposure risk
            open_ports = len(scan_result.open_ports)
            if open_ports > 20:
                exposure_risk = min(0.1, open_ports / 100.0)
                risk_score += exposure_risk
                risk_factors.append(f"High network exposure: {open_ports} open ports")
            
            # Service risk assessment
            risky_services = [s for s in scan_result.services if s.get("name", "").lower() in ["telnet", "ftp", "rsh"]]
            if risky_services:
                risk_score += len(risky_services) * 0.05
                risk_factors.append(f"Insecure services detected: {', '.join([s.get('name', '') for s in risky_services])}")
            
            # Normalize risk score
            risk_score = min(1.0, risk_score)
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = "critical"
            elif risk_score >= 0.6:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Calculate threat probability
            threat_probability = min(1.0, risk_score + len(threat_indicators) * 0.1)
            
            # Impact assessment
            impact_assessment = {
                "data_exposure_risk": "high" if critical_vulns > 0 or len(threat_indicators) > 5 else "medium",
                "service_disruption_risk": "high" if open_ports > 30 or risky_services else "low",
                "lateral_movement_risk": "high" if any("lateral" in v.get("description", "").lower() for v in scan_result.vulnerabilities) else "medium",
                "reputation_risk": "high" if any(indicator.get("category") == "malware" for indicator in threat_indicators) else "low"
            }
            
            # Confidence calculation
            confidence = min(1.0, (vuln_count / 20.0) + (len(threat_indicators) / 10.0) + 0.3)
            
            risk_assessment.update({
                "overall_risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "threat_probability": threat_probability,
                "impact_assessment": impact_assessment,
                "confidence": confidence
            })
            
        except Exception as e:
            logger.error(f"Failed to assess risk with intelligence: {e}")
            risk_assessment["error"] = str(e)
        
        return risk_assessment
    
    async def _generate_threat_hunting_queries(self, 
                                             scan_target: ScanTarget,
                                             scan_result: ScanResult, 
                                             threat_indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate threat hunting queries based on scan results and intelligence"""
        hunting_queries = []
        
        try:
            # Generate queries for known threat indicators
            for indicator in threat_indicators:
                indicator_value = indicator.get("value", "")
                indicator_type = indicator.get("indicator_type", "")
                
                if indicator_type == "ip":
                    query = {
                        "query_id": f"hunt_ip_{indicator['indicator_id'][:8]}",
                        "query_type": "network_activity",
                        "query_name": f"Hunt for IP {indicator_value} activity",
                        "siem_query": f"src_ip:{indicator_value} OR dst_ip:{indicator_value}",
                        "elasticsearch_query": {
                            "bool": {
                                "should": [
                                    {"term": {"source.ip": indicator_value}},
                                    {"term": {"destination.ip": indicator_value}}
                                ]
                            }
                        },
                        "splunk_query": f"index=* (src_ip=\"{indicator_value}\" OR dest_ip=\"{indicator_value}\")",
                        "severity": indicator.get("severity", "medium"),
                        "confidence": indicator.get("confidence", 0.5),
                        "description": f"Hunt for network activity involving threat indicator {indicator_value}",
                        "mitre_techniques": indicator.get("mitre_techniques", []),
                        "recommended_timeframe": "7d"
                    }
                    hunting_queries.append(query)
                
                elif indicator_type == "domain":
                    query = {
                        "query_id": f"hunt_domain_{indicator['indicator_id'][:8]}",
                        "query_type": "dns_activity",
                        "query_name": f"Hunt for domain {indicator_value} activity",
                        "siem_query": f"dns_query:{indicator_value} OR http_host:{indicator_value}",
                        "elasticsearch_query": {
                            "bool": {
                                "should": [
                                    {"term": {"dns.question.name": indicator_value}},
                                    {"term": {"http.request.headers.host": indicator_value}}
                                ]
                            }
                        },
                        "splunk_query": f"index=* (dns_query=\"{indicator_value}\" OR http_host=\"{indicator_value}\")",
                        "severity": indicator.get("severity", "medium"),
                        "confidence": indicator.get("confidence", 0.5),
                        "description": f"Hunt for DNS and HTTP activity involving domain {indicator_value}",
                        "mitre_techniques": indicator.get("mitre_techniques", []),
                        "recommended_timeframe": "30d"
                    }
                    hunting_queries.append(query)
                
                elif indicator_type in ["hash", "md5", "sha1", "sha256"]:
                    query = {
                        "query_id": f"hunt_hash_{indicator['indicator_id'][:8]}",
                        "query_type": "file_activity",
                        "query_name": f"Hunt for file hash {indicator_value} activity",
                        "siem_query": f"file_hash:{indicator_value}",
                        "elasticsearch_query": {
                            "bool": {
                                "should": [
                                    {"term": {"file.hash.md5": indicator_value}},
                                    {"term": {"file.hash.sha1": indicator_value}},
                                    {"term": {"file.hash.sha256": indicator_value}}
                                ]
                            }
                        },
                        "splunk_query": f"index=* file_hash=\"{indicator_value}\"",
                        "severity": indicator.get("severity", "medium"),
                        "confidence": indicator.get("confidence", 0.5),
                        "description": f"Hunt for file activity involving hash {indicator_value}",
                        "mitre_techniques": indicator.get("mitre_techniques", []),
                        "recommended_timeframe": "90d"
                    }
                    hunting_queries.append(query)
            
            # Generate queries based on scan results
            # Query for suspicious port activity
            unusual_ports = [p.get("port") for p in scan_result.open_ports if p.get("port", 0) > 10000]
            if unusual_ports:
                query = {
                    "query_id": f"hunt_unusual_ports_{scan_target.host.replace('.', '_')}",
                    "query_type": "network_activity",
                    "query_name": f"Hunt for unusual port activity on {scan_target.host}",
                    "siem_query": f"dst_ip:{scan_target.host} AND dst_port:({' OR '.join(map(str, unusual_ports))})",
                    "elasticsearch_query": {
                        "bool": {
                            "must": [
                                {"term": {"destination.ip": scan_target.host}},
                                {"terms": {"destination.port": unusual_ports}}
                            ]
                        }
                    },
                    "splunk_query": f"index=* dest_ip=\"{scan_target.host}\" (dest_port={' OR dest_port='.join(map(str, unusual_ports))})",
                    "severity": "medium",
                    "confidence": 0.7,
                    "description": f"Hunt for connections to unusual ports ({', '.join(map(str, unusual_ports))}) on target",
                    "recommended_timeframe": "14d"
                }
                hunting_queries.append(query)
            
            # Query for vulnerability exploitation attempts
            critical_vulns = [v for v in scan_result.vulnerabilities if v.get("severity") == "critical"]
            if critical_vulns:
                vuln_signatures = []
                for vuln in critical_vulns:
                    if "CVE" in vuln.get("description", ""):
                        cve_match = __import__("re").search(r"CVE-\d{4}-\d+", vuln.get("description", ""))
                        if cve_match:
                            vuln_signatures.append(cve_match.group())
                
                if vuln_signatures:
                    query = {
                        "query_id": f"hunt_vuln_exploitation_{scan_target.host.replace('.', '_')}",
                        "query_type": "exploitation_attempts",
                        "query_name": f"Hunt for exploitation attempts against {scan_target.host}",
                        "siem_query": f"dst_ip:{scan_target.host} AND ({' OR '.join(vuln_signatures)})",
                        "elasticsearch_query": {
                            "bool": {
                                "must": [
                                    {"term": {"destination.ip": scan_target.host}},
                                    {"bool": {"should": [{"match": {"message": sig}} for sig in vuln_signatures]}}
                                ]
                            }
                        },
                        "splunk_query": f"index=* dest_ip=\"{scan_target.host}\" ({' OR '.join(vuln_signatures)})",
                        "severity": "high",
                        "confidence": 0.8,
                        "description": f"Hunt for exploitation attempts against identified vulnerabilities",
                        "cve_references": vuln_signatures,
                        "recommended_timeframe": "30d"
                    }
                    hunting_queries.append(query)
            
            # Store generated queries for future reference
            self.generated_queries.extend(hunting_queries)
            
        except Exception as e:
            logger.error(f"Failed to generate threat hunting queries: {e}")
        
        return hunting_queries
    
    async def _get_threat_landscape_context(self) -> Dict[str, Any]:
        """Get current threat landscape context"""
        context = {}
        
        try:
            if self.fusion_engine and self.fusion_engine.threat_landscape:
                landscape = self.fusion_engine.threat_landscape
                
                context = {
                    "analysis_timestamp": landscape.timestamp.isoformat(),
                    "overall_risk_score": landscape.risk_score,
                    "top_threat_categories": [threat["category"] for threat in landscape.top_threats[:5]],
                    "emerging_threat_count": len(landscape.emerging_threats),
                    "high_severity_emerging": len([t for t in landscape.emerging_threats if t.get("severity") == "high"]),
                    "critical_emerging": len([t for t in landscape.emerging_threats if t.get("severity") == "critical"]),
                    "geographic_hotspots": list(landscape.geographic_distribution.keys())[:5],
                    "active_campaigns": len([actor for actor, activity in landscape.actor_activity.items() 
                                           if activity.get("activity_level") == "high"]),
                    "predictive_alerts": len(landscape.predictive_indicators),
                    "confidence_level": landscape.confidence
                }
            
        except Exception as e:
            logger.error(f"Failed to get threat landscape context: {e}")
        
        return context
    
    async def _generate_intelligence_recommendations(self, 
                                                   scan_target: ScanTarget,
                                                   scan_result: ScanResult,
                                                   threat_indicators: List[Dict[str, Any]],
                                                   threat_context: List[Dict[str, Any]], 
                                                   risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate intelligence-enhanced recommendations"""
        recommendations = []
        
        try:
            risk_level = risk_assessment.get("risk_level", "low")
            
            # Risk-based recommendations
            if risk_level == "critical":
                recommendations.extend([
                    "🚨 CRITICAL: Immediate isolation and investigation required",
                    "🔥 Deploy advanced threat hunting teams immediately",
                    "🛡️ Implement emergency network segmentation",
                    "📋 Activate incident response procedures"
                ])
            elif risk_level == "high":
                recommendations.extend([
                    "⚠️ HIGH RISK: Prioritize remediation within 24 hours",
                    "🔍 Conduct enhanced monitoring and logging",
                    "🎯 Deploy targeted threat hunting activities"
                ])
            
            # Threat indicator-based recommendations
            critical_indicators = [i for i in threat_indicators if i.get("severity") == "critical"]
            high_indicators = [i for i in threat_indicators if i.get("severity") == "high"]
            
            if critical_indicators:
                recommendations.append(f"🎯 {len(critical_indicators)} critical threat indicators detected - implement immediate blocking")
            
            if high_indicators:
                recommendations.append(f"⚡ {len(high_indicators)} high-severity indicators require enhanced monitoring")
            
            # Actor attribution recommendations
            nation_state_indicators = [i for i in threat_indicators 
                                     if any(actor in ["NATION_STATE", "APT_GROUP"] for actor in i.get("attributed_actors", []))]
            if nation_state_indicators:
                recommendations.extend([
                    "🏛️ Nation-state actor attribution detected - escalate to senior security leadership",
                    "🔒 Implement enhanced data protection measures",
                    "📊 Increase security monitoring frequency and depth"
                ])
            
            # MITRE technique-based recommendations
            all_techniques = []
            for indicator in threat_indicators:
                all_techniques.extend(indicator.get("mitre_techniques", []))
            
            common_techniques = set(all_techniques)
            if "T1566" in common_techniques:  # Phishing
                recommendations.append("📧 Implement enhanced email security and user awareness training")
            if "T1190" in common_techniques:  # Exploit Public-Facing Application
                recommendations.append("🌐 Prioritize public-facing application security hardening")
            if "T1078" in common_techniques:  # Valid Accounts
                recommendations.append("🔐 Implement enhanced authentication and access controls")
            
            # Contextual threat recommendations
            for context in threat_context:
                if context.get("mitigation_priority") == "critical":
                    recommendations.append(f"🎯 Critical threat context: {context.get('description', 'Unknown threat')}")
            
            # Vulnerability-based intelligence recommendations
            rce_vulns = [v for v in scan_result.vulnerabilities 
                        if "remote" in v.get("description", "").lower() and "code" in v.get("description", "").lower()]
            if rce_vulns:
                recommendations.append("💥 Remote code execution vulnerabilities detected - patch immediately")
            
            # Network exposure recommendations
            if len(scan_result.open_ports) > 30:
                recommendations.append("🌐 High network exposure detected - implement network segmentation")
            
            # Hunting query recommendations
            if len(self.generated_queries) > 0:
                recommendations.append(f"🔍 {len(self.generated_queries)} threat hunting queries generated - execute within 48 hours")
            
            # General intelligence-enhanced recommendations
            recommendations.extend([
                "📊 Implement continuous threat intelligence monitoring",
                "🤖 Deploy automated indicator matching and alerting",
                "📈 Establish threat intelligence fusion workflows",
                "👥 Conduct threat intelligence-driven security briefings"
            ])
            
        except Exception as e:
            logger.error(f"Failed to generate intelligence recommendations: {e}")
            recommendations.append("⚠️ Review threat intelligence analysis manually")
        
        return recommendations[:20]  # Limit to most important recommendations
    
    # Helper methods for intelligence analysis
    def _indicator_matches_target(self, indicator: GlobalThreatIndicator, target: ScanTarget) -> bool:
        """Check if threat indicator matches scan target"""
        try:
            # Direct value match
            if indicator.value == target.host:
                return True
            
            # Subnet/CIDR match for IP indicators
            if indicator.indicator_type in ["ip", "ip-src", "ip-dst"]:
                try:
                    import ipaddress
                    target_ip = ipaddress.ip_address(target.host)
                    
                    if "/" in indicator.value:  # CIDR notation
                        indicator_network = ipaddress.ip_network(indicator.value, strict=False)
                        return target_ip in indicator_network
                    else:
                        indicator_ip = ipaddress.ip_address(indicator.value)
                        return target_ip == indicator_ip
                except:
                    pass
            
            # Domain/subdomain match
            if indicator.indicator_type in ["domain", "hostname"]:
                return (indicator.value in target.host or 
                        target.host.endswith(f".{indicator.value}"))
            
            return False
            
        except Exception as e:
            logger.error(f"Error matching indicator to target: {e}")
            return False
    
    def _extract_relevant_context(self, indicator: GlobalThreatIndicator, target: ScanTarget) -> Dict[str, Any]:
        """Extract relevant context from threat indicator for target"""
        context = {}
        
        try:
            context["match_reason"] = "direct_match"
            context["indicator_age_days"] = (datetime.utcnow() - indicator.first_seen).days
            context["source_count"] = len(indicator.sources)
            context["technique_count"] = len(indicator.mitre_techniques)
            context["tag_count"] = len(indicator.tags)
            context["tlp_marking"] = indicator.tlp_marking
            
            # Add specific context based on indicator type
            if indicator.indicator_type in ["ip", "ip-src", "ip-dst"]:
                context["network_indicator"] = True
                context["geolocation"] = indicator.geolocation
            elif indicator.indicator_type in ["domain", "hostname"]:
                context["dns_indicator"] = True
            elif indicator.indicator_type in ["hash", "md5", "sha1", "sha256"]:
                context["file_indicator"] = True
            
        except Exception as e:
            logger.error(f"Error extracting indicator context: {e}")
        
        return context
    
    async def _get_vulnerability_intelligence_context(self, 
                                                    vulnerability: Dict[str, Any], 
                                                    target_host: str) -> Dict[str, Any]:
        """Get threat intelligence context for a specific vulnerability"""
        context = {
            "exploitation_indicators": [],
            "related_campaigns": [],
            "threat_actor_interest": [],
            "exploit_availability": "unknown",
            "active_exploitation": False
        }
        
        try:
            # Search for CVE-related threat intelligence
            vuln_description = vulnerability.get("description", "")
            cve_pattern = __import__("re").findall(r"CVE-\d{4}-\d+", vuln_description)
            
            if cve_pattern and self.fusion_engine:
                for cve in cve_pattern:
                    # Search for indicators related to this CVE
                    cve_indicators = await self.fusion_engine.search_indicators(
                        {"tags": cve}, 
                        limit=20
                    )
                    
                    for indicator in cve_indicators:
                        context["exploitation_indicators"].append({
                            "indicator_id": indicator.indicator_id,
                            "indicator_type": indicator.indicator_type,
                            "value": indicator.value,
                            "severity": indicator.severity.value,
                            "confidence": indicator.confidence,
                            "sources": indicator.sources
                        })
                        
                        # Check for active exploitation
                        if (indicator.category == ThreatCategory.VULNERABILITY_EXPLOITATION and
                            (datetime.utcnow() - indicator.last_seen).days < 30):
                            context["active_exploitation"] = True
            
            # Analyze port/service specific threats
            vuln_port = vulnerability.get("port")
            if vuln_port and self.fusion_engine:
                port_indicators = await self.fusion_engine.search_indicators(
                    {"value": str(vuln_port)}, 
                    limit=10
                )
                
                for indicator in port_indicators:
                    if indicator.category in [ThreatCategory.NETWORK_INTRUSION, ThreatCategory.VULNERABILITY_EXPLOITATION]:
                        context["exploitation_indicators"].append({
                            "indicator_id": indicator.indicator_id,
                            "indicator_type": indicator.indicator_type,
                            "port_specific": True,
                            "severity": indicator.severity.value,
                            "attributed_actors": [actor.value for actor in indicator.attributed_actors]
                        })
            
        except Exception as e:
            logger.error(f"Error getting vulnerability intelligence context: {e}")
        
        return context
    
    async def _calculate_enhanced_risk_score(self, 
                                           vulnerability: Dict[str, Any], 
                                           intel_context: Dict[str, Any]) -> float:
        """Calculate enhanced risk score with threat intelligence"""
        try:
            # Base score from vulnerability severity
            base_score = {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.5,
                "low": 0.3,
                "info": 0.1
            }.get(vulnerability.get("severity", "medium"), 0.5)
            
            # Intelligence enhancement
            intel_boost = 0.0
            
            # Active exploitation boost
            if intel_context.get("active_exploitation", False):
                intel_boost += 0.2
            
            # Exploitation indicators boost
            exploitation_indicators = intel_context.get("exploitation_indicators", [])
            if exploitation_indicators:
                high_severity_indicators = [i for i in exploitation_indicators if i.get("severity") in ["critical", "high"]]
                intel_boost += min(0.3, len(high_severity_indicators) * 0.1)
            
            # Threat actor interest boost
            threat_actors = intel_context.get("threat_actor_interest", [])
            if threat_actors:
                intel_boost += min(0.2, len(threat_actors) * 0.05)
            
            # Calculate final score
            enhanced_score = min(1.0, base_score + intel_boost)
            
            return enhanced_score
            
        except Exception as e:
            logger.error(f"Error calculating enhanced risk score: {e}")
            return 0.5
    
    async def _calculate_exploitation_probability(self, 
                                                vulnerability: Dict[str, Any], 
                                                intel_context: Dict[str, Any]) -> float:
        """Calculate probability of exploitation based on intelligence"""
        try:
            probability = 0.0
            
            # Base probability from severity
            base_prob = {
                "critical": 0.8,
                "high": 0.6,
                "medium": 0.4,
                "low": 0.2,
                "info": 0.1
            }.get(vulnerability.get("severity", "medium"), 0.4)
            
            probability += base_prob
            
            # Active exploitation significantly increases probability
            if intel_context.get("active_exploitation", False):
                probability += 0.3
            
            # Exploitation indicators increase probability
            exploitation_indicators = intel_context.get("exploitation_indicators", [])
            if exploitation_indicators:
                probability += min(0.2, len(exploitation_indicators) * 0.05)
            
            # Threat actor interest increases probability
            if intel_context.get("threat_actor_interest"):
                probability += 0.1
            
            return min(1.0, probability)
            
        except Exception as e:
            logger.error(f"Error calculating exploitation probability: {e}")
            return 0.5
    
    async def _get_threat_actor_attribution(self, 
                                          vulnerability: Dict[str, Any], 
                                          intel_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get threat actor attribution for vulnerability"""
        attributions = []
        
        try:
            # Extract actors from exploitation indicators
            exploitation_indicators = intel_context.get("exploitation_indicators", [])
            
            actor_counts = {}
            for indicator in exploitation_indicators:
                actors = indicator.get("attributed_actors", [])
                for actor in actors:
                    actor_counts[actor] = actor_counts.get(actor, 0) + 1
            
            # Create attribution entries
            for actor, count in actor_counts.items():
                confidence = min(0.9, count / 5.0)  # More indicators = higher confidence
                
                attribution = {
                    "actor_name": actor,
                    "confidence": confidence,
                    "indicator_count": count,
                    "interest_level": "high" if count >= 3 else "medium" if count >= 2 else "low",
                    "last_activity": datetime.utcnow().isoformat()  # Would be actual last seen in production
                }
                attributions.append(attribution)
            
        except Exception as e:
            logger.error(f"Error getting threat actor attribution: {e}")
        
        return attributions
    
    def _scan_indicates_threat_category(self, scan_result: ScanResult, threat_category: str) -> bool:
        """Check if scan results indicate presence of specific threat category"""
        try:
            # Simple heuristic matching - in production would be more sophisticated
            category_indicators = {
                "malware": ["trojan", "backdoor", "virus", "malware"],
                "phishing": ["phishing", "deceptive", "fraudulent"],
                "ransomware": ["ransom", "crypto", "encrypt"],
                "network_intrusion": ["intrusion", "unauthorized", "breach"],
                "vulnerability_exploitation": ["exploit", "vulnerability", "cve"]
            }
            
            indicators = category_indicators.get(threat_category, [])
            
            # Check vulnerabilities for category indicators
            for vuln in scan_result.vulnerabilities:
                vuln_text = (vuln.get("description", "") + " " + vuln.get("name", "")).lower()
                if any(indicator in vuln_text for indicator in indicators):
                    return True
            
            # Check services for category indicators  
            for service in scan_result.services:
                service_text = (service.get("name", "") + " " + service.get("product", "")).lower()
                if any(indicator in service_text for indicator in indicators):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking threat category indication: {e}")
            return False
    
    def _count_threat_indicators_in_scan(self, scan_result: ScanResult, threat_category: str) -> int:
        """Count threat indicators of specific category in scan results"""
        try:
            count = 0
            
            # Count vulnerabilities related to category
            for vuln in scan_result.vulnerabilities:
                if self._vulnerability_matches_category(vuln, threat_category):
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error counting threat indicators: {e}")
            return 0
    
    def _scan_relates_to_emerging_threat(self, scan_result: ScanResult, emerging_threat: Dict[str, Any]) -> bool:
        """Check if scan results relate to an emerging threat"""
        try:
            threat_category = emerging_threat.get("category", "")
            return self._scan_indicates_threat_category(scan_result, threat_category)
            
        except Exception as e:
            logger.error(f"Error checking emerging threat relation: {e}")
            return False
    
    def _vulnerability_matches_category(self, vulnerability: Dict[str, Any], category: str) -> bool:
        """Check if vulnerability matches threat category"""
        try:
            vuln_text = (vulnerability.get("description", "") + " " + vulnerability.get("name", "")).lower()
            
            category_keywords = {
                "malware": ["malware", "trojan", "backdoor", "virus"],
                "phishing": ["phishing", "deceptive", "social"],
                "ransomware": ["ransom", "crypto", "encrypt"],
                "network_intrusion": ["intrusion", "unauthorized", "remote"],
                "vulnerability_exploitation": ["exploit", "vulnerability", "cve", "buffer", "injection"]
            }
            
            keywords = category_keywords.get(category, [])
            return any(keyword in vuln_text for keyword in keywords)
            
        except Exception as e:
            logger.error(f"Error matching vulnerability to category: {e}")
            return False
    
    async def _intelligence_update_loop(self):
        """Background task to update intelligence data"""
        while True:
            try:
                # Trigger intelligence fusion every 30 minutes
                if self.fusion_engine:
                    await self.fusion_engine.fuse_intelligence()
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Intelligence update loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _cache_cleanup_loop(self):
        """Background task to clean up expired cache entries"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, data in self.intelligence_cache.items():
                    if current_time - data["timestamp"] > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.intelligence_cache[key]
                
                await asyncio.sleep(600)  # Clean every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup loop error: {e}")
                await asyncio.sleep(300)
    
    async def health_check(self) -> ServiceHealth:
        """Perform health check on enhanced threat intelligence service"""
        try:
            checks = {
                "fusion_engine_available": self.fusion_engine is not None,
                "scanner_service_available": self.scanner_service is not None,
                "cache_size": len(self.intelligence_cache),
                "generated_queries": len(self.generated_queries)
            }
            
            # Check fusion engine health
            fusion_healthy = True
            if self.fusion_engine:
                try:
                    metrics = await self.fusion_engine.get_fusion_metrics()
                    checks["fusion_metrics"] = metrics
                except:
                    fusion_healthy = False
            
            status = ServiceStatus.HEALTHY if fusion_healthy else ServiceStatus.DEGRADED
            message = "Enhanced threat intelligence service operational" if fusion_healthy else "Fusion engine degraded"
            
            return ServiceHealth(
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                checks=checks
            )
            
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            )


# Global service instance
_enhanced_intelligence_service: Optional[EnhancedThreatIntelligenceService] = None

async def get_enhanced_threat_intelligence_service() -> EnhancedThreatIntelligenceService:
    """Get global enhanced threat intelligence service instance"""
    global _enhanced_intelligence_service
    
    if _enhanced_intelligence_service is None:
        _enhanced_intelligence_service = EnhancedThreatIntelligenceService()
        await _enhanced_intelligence_service.initialize()
    
    return _enhanced_intelligence_service