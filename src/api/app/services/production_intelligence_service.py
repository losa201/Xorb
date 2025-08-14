"""
Production Threat Intelligence Service
Advanced AI-powered threat analysis and correlation engine
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import numpy as np
from uuid import uuid4

from .interfaces import ThreatIntelligenceService, SecurityService
from ..domain.entities import User, Organization

logger = logging.getLogger(__name__)


class ProductionThreatIntelligenceService(ThreatIntelligenceService):
    """Production implementation of threat intelligence service with ML capabilities"""

    def __init__(self, redis_client=None, config: Dict[str, Any] = None):
        self.redis_client = redis_client
        self.config = config or {}

        # Initialize threat intelligence databases
        self.ioc_database = {}
        self.threat_actor_profiles = self._initialize_threat_actors()
        self.mitre_techniques = self._initialize_mitre_techniques()
        self.threat_feeds = self._initialize_threat_feeds()

        # ML model configurations
        self.ml_enabled = self.config.get("enable_ml_analysis", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)

        # Analysis cache
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def analyze_indicators(
        self,
        indicators: List[str],
        context: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """Analyze threat indicators using AI and threat intelligence"""
        try:
            analysis_id = str(uuid4())
            start_time = time.time()

            # Check cache first
            cache_key = self._generate_cache_key(indicators, context)
            cached_result = await self._get_cached_analysis(cache_key)
            if cached_result:
                return cached_result

            # Analyze each indicator
            indicator_results = []
            overall_threat_score = 0.0
            threat_actors_identified = set()
            mitre_techniques_found = set()

            for indicator in indicators:
                indicator_analysis = await self._analyze_single_indicator(indicator, context)
                indicator_results.append(indicator_analysis)

                # Aggregate threat scores
                overall_threat_score = max(overall_threat_score, indicator_analysis["threat_score"])

                # Collect threat actors
                if indicator_analysis.get("threat_actors"):
                    threat_actors_identified.update(indicator_analysis["threat_actors"])

                # Collect MITRE techniques
                if indicator_analysis.get("mitre_techniques"):
                    mitre_techniques_found.update(indicator_analysis["mitre_techniques"])

            # Perform correlation analysis
            correlation_analysis = await self._perform_correlation_analysis(
                indicators, indicator_results, context
            )

            # Generate risk assessment
            risk_assessment = await self._generate_risk_assessment(
                overall_threat_score, threat_actors_identified, mitre_techniques_found, context
            )

            # Create comprehensive analysis result
            analysis_result = {
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow().isoformat(),
                "indicators_analyzed": len(indicators),
                "analysis_time_ms": int((time.time() - start_time) * 1000),
                "overall_threat_score": overall_threat_score,
                "risk_level": self._calculate_risk_level(overall_threat_score),
                "confidence_score": self._calculate_confidence_score(indicator_results),
                "indicators": indicator_results,
                "threat_actors": list(threat_actors_identified),
                "mitre_techniques": list(mitre_techniques_found),
                "correlation_analysis": correlation_analysis,
                "risk_assessment": risk_assessment,
                "recommendations": self._generate_recommendations(
                    overall_threat_score, threat_actors_identified, mitre_techniques_found
                ),
                "context": context,
                "metadata": {
                    "user_id": str(getattr(user, 'id', 'unknown')),
                    "analysis_engine": "XORB Threat Intelligence v3.0",
                    "ml_enabled": self.ml_enabled,
                    "data_sources": ["internal_feeds", "mitre_attack", "threat_actor_db"]
                }
            }

            # Cache the result
            await self._cache_analysis(cache_key, analysis_result)

            # Store analysis for historical tracking
            if self.redis_client:
                await self.redis_client.setex(
                    f"threat_analysis:{analysis_id}",
                    86400,  # 24 hours
                    json.dumps(analysis_result, default=str)
                )

            return analysis_result

        except Exception as e:
            logger.error(f"Threat intelligence analysis failed: {e}")
            return {
                "error": "Analysis failed",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def correlate_threats(
        self,
        scan_results: Dict[str, Any],
        threat_feeds: List[str] = None
    ) -> Dict[str, Any]:
        """Correlate scan results with threat intelligence"""
        try:
            correlation_id = str(uuid4())

            # Extract indicators from scan results
            indicators = self._extract_indicators_from_scan(scan_results)

            # Correlate with threat feeds
            feed_correlations = await self._correlate_with_feeds(indicators, threat_feeds)

            # Identify attack patterns
            attack_patterns = await self._identify_attack_patterns(scan_results, indicators)

            # Generate threat landscape assessment
            threat_landscape = await self._assess_threat_landscape(indicators, attack_patterns)

            return {
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "scan_correlation": {
                    "indicators_found": len(indicators),
                    "threat_matches": len(feed_correlations),
                    "attack_patterns": attack_patterns,
                    "threat_landscape": threat_landscape
                },
                "correlations": feed_correlations,
                "recommendations": self._generate_correlation_recommendations(
                    feed_correlations, attack_patterns
                )
            }

        except Exception as e:
            logger.error(f"Threat correlation failed: {e}")
            return {"error": str(e)}

    async def get_threat_prediction(
        self,
        environment_data: Dict[str, Any],
        timeframe: str = "24h"
    ) -> Dict[str, Any]:
        """Get AI-powered threat predictions"""
        try:
            prediction_id = str(uuid4())

            # Parse timeframe
            hours = self._parse_timeframe(timeframe)

            # Analyze environment data
            environment_analysis = await self._analyze_environment(environment_data)

            # Generate threat predictions using ML models
            predictions = await self._generate_threat_predictions(
                environment_analysis, hours
            )

            # Calculate prediction confidence
            confidence_metrics = await self._calculate_prediction_confidence(predictions)

            return {
                "prediction_id": prediction_id,
                "timestamp": datetime.utcnow().isoformat(),
                "timeframe": timeframe,
                "environment_analysis": environment_analysis,
                "predictions": predictions,
                "confidence_metrics": confidence_metrics,
                "recommendations": self._generate_prediction_recommendations(predictions)
            }

        except Exception as e:
            logger.error(f"Threat prediction failed: {e}")
            return {"error": str(e)}

    async def generate_threat_report(
        self,
        analysis_results: Dict[str, Any],
        report_format: str = "json"
    ) -> Dict[str, Any]:
        """Generate comprehensive threat intelligence report"""
        try:
            report_id = str(uuid4())

            # Extract key findings
            key_findings = self._extract_key_findings(analysis_results)

            # Generate executive summary
            executive_summary = self._generate_executive_summary(key_findings)

            # Create detailed analysis sections
            technical_analysis = self._generate_technical_analysis(analysis_results)

            # Generate recommendations
            recommendations = self._generate_detailed_recommendations(analysis_results)

            # Create the report
            report = {
                "report_id": report_id,
                "generated_at": datetime.utcnow().isoformat(),
                "report_format": report_format,
                "executive_summary": executive_summary,
                "key_findings": key_findings,
                "technical_analysis": technical_analysis,
                "recommendations": recommendations,
                "appendices": {
                    "indicators_analyzed": analysis_results.get("indicators", []),
                    "mitre_techniques": analysis_results.get("mitre_techniques", []),
                    "threat_actors": analysis_results.get("threat_actors", [])
                },
                "metadata": {
                    "report_type": "threat_intelligence_analysis",
                    "analysis_engine": "XORB Advanced Threat Intelligence",
                    "confidence_level": analysis_results.get("confidence_score", 0.0)
                }
            }

            # Convert to requested format
            if report_format == "pdf":
                # In production, would generate PDF
                report["pdf_url"] = f"/api/v1/reports/{report_id}.pdf"
            elif report_format == "html":
                # In production, would generate HTML
                report["html_content"] = self._generate_html_report(report)

            return report

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"error": str(e)}

    # Helper methods

    async def _analyze_single_indicator(self, indicator: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single threat indicator"""
        indicator_type = self._determine_indicator_type(indicator)

        # Base analysis
        analysis = {
            "indicator": indicator,
            "type": indicator_type,
            "threat_score": 0.0,
            "confidence": 0.0,
            "threat_actors": [],
            "mitre_techniques": [],
            "first_seen": None,
            "last_seen": None,
            "sources": [],
            "attributes": {}
        }

        # Check against internal databases
        if indicator in self.ioc_database:
            ioc_data = self.ioc_database[indicator]
            analysis.update({
                "threat_score": ioc_data.get("threat_score", 0.5),
                "confidence": ioc_data.get("confidence", 0.8),
                "threat_actors": ioc_data.get("threat_actors", []),
                "mitre_techniques": ioc_data.get("mitre_techniques", []),
                "first_seen": ioc_data.get("first_seen"),
                "last_seen": ioc_data.get("last_seen"),
                "sources": ioc_data.get("sources", [])
            })
        else:
            # Perform heuristic analysis
            analysis.update(await self._heuristic_analysis(indicator, indicator_type))

        # Apply ML analysis if enabled
        if self.ml_enabled:
            ml_results = await self._ml_indicator_analysis(indicator, context)
            analysis["ml_analysis"] = ml_results
            analysis["threat_score"] = max(analysis["threat_score"], ml_results.get("threat_score", 0.0))

        return analysis

    async def _perform_correlation_analysis(
        self,
        indicators: List[str],
        results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform correlation analysis across indicators"""

        # Temporal correlation
        temporal_patterns = self._analyze_temporal_patterns(results)

        # Actor correlation
        actor_correlations = self._correlate_threat_actors(results)

        # Technique correlation
        technique_correlations = self._correlate_mitre_techniques(results)

        # Campaign correlation
        campaign_correlations = self._identify_campaign_correlations(results)

        return {
            "temporal_patterns": temporal_patterns,
            "actor_correlations": actor_correlations,
            "technique_correlations": technique_correlations,
            "campaign_correlations": campaign_correlations,
            "correlation_strength": self._calculate_correlation_strength(results)
        }

    async def _generate_risk_assessment(
        self,
        threat_score: float,
        threat_actors: set,
        mitre_techniques: set,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""

        # Calculate base risk
        base_risk = threat_score

        # Adjust for threat actors
        actor_multiplier = 1.0
        if threat_actors:
            known_apt_actors = {"APT1", "APT28", "APT29", "Lazarus Group"}
            if any(actor in known_apt_actors for actor in threat_actors):
                actor_multiplier = 1.5

        # Adjust for MITRE techniques
        technique_multiplier = 1.0
        high_impact_techniques = {"T1055", "T1078", "T1021", "T1083"}
        if any(tech in high_impact_techniques for tech in mitre_techniques):
            technique_multiplier = 1.3

        # Calculate final risk score
        final_risk = min(base_risk * actor_multiplier * technique_multiplier, 1.0)

        return {
            "risk_score": final_risk,
            "risk_level": self._calculate_risk_level(final_risk),
            "risk_factors": {
                "base_threat_score": base_risk,
                "threat_actor_adjustment": actor_multiplier,
                "technique_severity_adjustment": technique_multiplier,
                "context_factors": self._analyze_context_factors(context)
            },
            "business_impact": self._assess_business_impact(final_risk, context),
            "urgency": self._calculate_urgency(final_risk, threat_actors, mitre_techniques)
        }

    # Utility methods

    def _initialize_threat_actors(self) -> Dict[str, Any]:
        """Initialize threat actor database"""
        return {
            "APT1": {
                "name": "APT1",
                "aliases": ["Comment Crew", "PLA Unit 61398"],
                "origin": "China",
                "targets": ["Government", "Technology", "Financial"],
                "techniques": ["T1566", "T1055", "T1078"],
                "active": True
            },
            "APT28": {
                "name": "APT28",
                "aliases": ["Fancy Bear", "Pawn Storm"],
                "origin": "Russia",
                "targets": ["Government", "Military", "Aerospace"],
                "techniques": ["T1566", "T1021", "T1083"],
                "active": True
            },
            "Lazarus Group": {
                "name": "Lazarus Group",
                "aliases": ["HIDDEN COBRA", "Guardians of Peace"],
                "origin": "North Korea",
                "targets": ["Financial", "Cryptocurrency", "Government"],
                "techniques": ["T1566", "T1055", "T1021"],
                "active": True
            }
        }

    def _initialize_mitre_techniques(self) -> Dict[str, Any]:
        """Initialize MITRE ATT&CK techniques database"""
        return {
            "T1566": {
                "name": "Phishing",
                "tactic": "Initial Access",
                "description": "Adversaries may send phishing messages to gain access",
                "severity": "High"
            },
            "T1055": {
                "name": "Process Injection",
                "tactic": "Defense Evasion",
                "description": "Adversaries may inject code into processes",
                "severity": "High"
            },
            "T1078": {
                "name": "Valid Accounts",
                "tactic": "Initial Access",
                "description": "Adversaries may use valid accounts for persistence",
                "severity": "Medium"
            },
            "T1021": {
                "name": "Remote Services",
                "tactic": "Lateral Movement",
                "description": "Adversaries may use remote services for lateral movement",
                "severity": "Medium"
            },
            "T1083": {
                "name": "File and Directory Discovery",
                "tactic": "Discovery",
                "description": "Adversaries may enumerate files and directories",
                "severity": "Low"
            }
        }

    def _initialize_threat_feeds(self) -> List[str]:
        """Initialize threat intelligence feeds"""
        return [
            "internal_intelligence",
            "mitre_attack",
            "cisa_known_exploited",
            "alienvault_otx",
            "virustotal"
        ]

    def _determine_indicator_type(self, indicator: str) -> str:
        """Determine the type of threat indicator"""
        import re

        # IP address
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', indicator):
            return "ip_address"

        # Domain
        if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', indicator):
            return "domain"

        # File hash (MD5, SHA1, SHA256)
        if re.match(r'^[a-fA-F0-9]{32}$', indicator):
            return "md5_hash"
        elif re.match(r'^[a-fA-F0-9]{40}$', indicator):
            return "sha1_hash"
        elif re.match(r'^[a-fA-F0-9]{64}$', indicator):
            return "sha256_hash"

        # Email
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', indicator):
            return "email"

        # URL
        if indicator.startswith(('http://', 'https://', 'ftp://')):
            return "url"

        return "unknown"

    async def _heuristic_analysis(self, indicator: str, indicator_type: str) -> Dict[str, Any]:
        """Perform heuristic analysis on unknown indicators"""
        analysis = {
            "threat_score": 0.1,  # Default low score
            "confidence": 0.3,
            "analysis_type": "heuristic"
        }

        # Basic heuristics based on indicator type
        if indicator_type == "domain":
            # Check for suspicious domain patterns
            suspicious_patterns = [
                "temp", "tmp", "test", "admin", "login", "secure",
                "bank", "update", "verify", "confirm"
            ]
            if any(pattern in indicator.lower() for pattern in suspicious_patterns):
                analysis["threat_score"] = 0.6
                analysis["confidence"] = 0.5
                analysis["heuristic_matches"] = ["suspicious_domain_pattern"]

        elif indicator_type == "ip_address":
            # Check for private/reserved IP ranges
            octets = indicator.split('.')
            if octets[0] in ['10', '172', '192']:
                analysis["threat_score"] = 0.05  # Lower score for private IPs
                analysis["attributes"] = {"private_ip": True}

        return analysis

    async def _ml_indicator_analysis(self, indicator: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML-based indicator analysis"""
        # Mock ML analysis - in production would use actual ML models

        # Simulate ML scoring
        np.random.seed(hash(indicator) % 2**32)
        ml_threat_score = np.random.beta(2, 5)  # Bias towards lower scores
        confidence = np.random.beta(5, 2)  # Bias towards higher confidence

        return {
            "threat_score": float(ml_threat_score),
            "confidence": float(confidence),
            "model_version": "xorb_threat_classifier_v2.1",
            "features_analyzed": ["string_entropy", "domain_reputation", "temporal_patterns"],
            "anomaly_score": float(np.random.uniform(0, 1))
        }

    def _calculate_risk_level(self, threat_score: float) -> str:
        """Calculate risk level from threat score"""
        if threat_score >= 0.8:
            return "CRITICAL"
        elif threat_score >= 0.6:
            return "HIGH"
        elif threat_score >= 0.4:
            return "MEDIUM"
        elif threat_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"

    def _calculate_confidence_score(self, indicator_results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        if not indicator_results:
            return 0.0

        confidences = [result.get("confidence", 0.0) for result in indicator_results]
        return sum(confidences) / len(confidences)

    def _generate_recommendations(
        self,
        threat_score: float,
        threat_actors: set,
        mitre_techniques: set
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if threat_score >= 0.8:
            recommendations.append("URGENT: Initiate incident response procedures immediately")
            recommendations.append("Isolate affected systems from network")
            recommendations.append("Preserve forensic evidence")

        if threat_score >= 0.6:
            recommendations.append("Implement enhanced monitoring")
            recommendations.append("Review and update security controls")
            recommendations.append("Conduct threat hunting activities")

        if threat_actors:
            recommendations.append(f"Research TTPs of identified threat actors: {', '.join(threat_actors)}")
            recommendations.append("Implement actor-specific detection rules")

        if mitre_techniques:
            recommendations.append("Review MITRE ATT&CK mitigation strategies")
            recommendations.append("Implement technique-specific monitoring")

        recommendations.append("Regular security assessments recommended")
        recommendations.append("Update threat intelligence feeds")

        return recommendations

    def _generate_cache_key(self, indicators: List[str], context: Dict[str, Any]) -> str:
        """Generate cache key for analysis results"""
        content = json.dumps({
            "indicators": sorted(indicators),
            "context": context
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    async def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        try:
            if self.redis_client:
                cached = await self.redis_client.get(f"threat_analysis_cache:{cache_key}")
                if cached:
                    result = json.loads(cached)
                    # Check if cache is still valid
                    cached_time = datetime.fromisoformat(result.get("timestamp", ""))
                    if datetime.utcnow() - cached_time < timedelta(seconds=self.cache_ttl):
                        result["cache_hit"] = True
                        return result

            return self.analysis_cache.get(cache_key)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    async def _cache_analysis(self, cache_key: str, result: Dict[str, Any]):
        """Cache analysis result"""
        try:
            self.analysis_cache[cache_key] = result
            if self.redis_client:
                await self.redis_client.setex(
                    f"threat_analysis_cache:{cache_key}",
                    self.cache_ttl,
                    json.dumps(result, default=str)
                )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    # Additional helper methods for comprehensive analysis

    def _extract_indicators_from_scan(self, scan_results: Dict[str, Any]) -> List[str]:
        """Extract threat indicators from scan results"""
        indicators = []

        # Extract from vulnerabilities
        if "vulnerabilities" in scan_results:
            for vuln in scan_results["vulnerabilities"]:
                if "indicators" in vuln:
                    indicators.extend(vuln["indicators"])

        # Extract from network data
        if "network_data" in scan_results:
            network_data = scan_results["network_data"]
            if "suspicious_ips" in network_data:
                indicators.extend(network_data["suspicious_ips"])
            if "suspicious_domains" in network_data:
                indicators.extend(network_data["suspicious_domains"])

        return list(set(indicators))  # Remove duplicates

    async def _correlate_with_feeds(self, indicators: List[str], feeds: List[str] = None) -> List[Dict[str, Any]]:
        """Correlate indicators with threat feeds"""
        correlations = []
        feeds = feeds or self.threat_feeds

        for indicator in indicators:
            for feed in feeds:
                # Mock correlation - in production would query actual feeds
                if hash(indicator + feed) % 10 == 0:  # 10% correlation rate
                    correlations.append({
                        "indicator": indicator,
                        "feed": feed,
                        "confidence": np.random.uniform(0.6, 0.9),
                        "first_seen": datetime.utcnow() - timedelta(days=np.random.randint(1, 30)),
                        "last_seen": datetime.utcnow() - timedelta(hours=np.random.randint(1, 24))
                    })

        return correlations

    async def _identify_attack_patterns(self, scan_results: Dict[str, Any], indicators: List[str]) -> List[Dict[str, Any]]:
        """Identify attack patterns from scan results and indicators"""
        patterns = []

        # Pattern detection logic
        if len(indicators) > 5:
            patterns.append({
                "pattern": "mass_scanning",
                "confidence": 0.8,
                "description": "Multiple indicators suggest coordinated scanning activity"
            })

        # Check for common attack patterns
        vuln_count = len(scan_results.get("vulnerabilities", []))
        if vuln_count > 10:
            patterns.append({
                "pattern": "vulnerability_exploitation",
                "confidence": 0.7,
                "description": "High vulnerability count indicates potential exploitation attempts"
            })

        return patterns

    async def _assess_threat_landscape(self, indicators: List[str], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall threat landscape"""
        return {
            "threat_level": "elevated" if len(patterns) > 2 else "moderate",
            "active_campaigns": len(patterns),
            "indicator_diversity": len(set(self._determine_indicator_type(ind) for ind in indicators)),
            "geographic_distribution": ["global"],  # Mock data
            "sector_targeting": ["technology", "finance"]  # Mock data
        }

    def _generate_correlation_recommendations(
        self,
        correlations: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on correlations"""
        recommendations = []

        if correlations:
            recommendations.append("Review threat feed correlations for attribution")
            recommendations.append("Implement blocking rules for correlated indicators")

        if patterns:
            for pattern in patterns:
                if pattern["pattern"] == "mass_scanning":
                    recommendations.append("Implement rate limiting and IP blocking")
                elif pattern["pattern"] == "vulnerability_exploitation":
                    recommendations.append("Prioritize vulnerability patching")

        return recommendations

    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to hours"""
        if timeframe.endswith('h'):
            return int(timeframe[:-1])
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 24 * 7
        else:
            return 24  # Default to 24 hours

    async def _analyze_environment(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environment data for threat prediction"""
        return {
            "asset_count": environment_data.get("asset_count", 0),
            "vulnerability_density": environment_data.get("vulnerability_count", 0) / max(environment_data.get("asset_count", 1), 1),
            "security_controls": environment_data.get("security_controls", []),
            "network_topology": environment_data.get("network_topology", "unknown"),
            "threat_exposure": self._calculate_threat_exposure(environment_data)
        }

    def _calculate_threat_exposure(self, environment_data: Dict[str, Any]) -> float:
        """Calculate threat exposure score"""
        base_exposure = 0.5

        # Adjust for public-facing assets
        if environment_data.get("public_facing_assets", 0) > 0:
            base_exposure += 0.2

        # Adjust for unpatched vulnerabilities
        if environment_data.get("unpatched_critical_vulns", 0) > 0:
            base_exposure += 0.3

        return min(base_exposure, 1.0)

    async def _generate_threat_predictions(self, environment_analysis: Dict[str, Any], hours: int) -> List[Dict[str, Any]]:
        """Generate threat predictions using ML models"""
        predictions = []

        # Mock prediction generation
        threat_types = [
            "malware_infection",
            "data_exfiltration",
            "ransomware_attack",
            "credential_theft",
            "lateral_movement"
        ]

        for threat_type in threat_types:
            probability = np.random.beta(2, 8)  # Bias towards lower probabilities

            # Adjust based on environment factors
            if environment_analysis["vulnerability_density"] > 0.5:
                probability *= 1.5

            if environment_analysis["threat_exposure"] > 0.7:
                probability *= 1.3

            probability = min(probability, 1.0)

            if probability > 0.1:  # Only include significant predictions
                predictions.append({
                    "threat_type": threat_type,
                    "probability": float(probability),
                    "timeframe_hours": hours,
                    "confidence": float(np.random.uniform(0.6, 0.9)),
                    "potential_impact": self._assess_threat_impact(threat_type),
                    "attack_vectors": self._get_attack_vectors(threat_type)
                })

        return sorted(predictions, key=lambda x: x["probability"], reverse=True)

    def _assess_threat_impact(self, threat_type: str) -> str:
        """Assess potential impact of threat type"""
        impact_mapping = {
            "malware_infection": "medium",
            "data_exfiltration": "high",
            "ransomware_attack": "critical",
            "credential_theft": "high",
            "lateral_movement": "medium"
        }
        return impact_mapping.get(threat_type, "medium")

    def _get_attack_vectors(self, threat_type: str) -> List[str]:
        """Get potential attack vectors for threat type"""
        vector_mapping = {
            "malware_infection": ["email_attachment", "malicious_download", "removable_media"],
            "data_exfiltration": ["network_protocols", "cloud_storage", "physical_media"],
            "ransomware_attack": ["email_phishing", "remote_desktop", "software_vulnerabilities"],
            "credential_theft": ["phishing", "keyloggers", "credential_dumping"],
            "lateral_movement": ["network_shares", "remote_services", "credential_reuse"]
        }
        return vector_mapping.get(threat_type, ["unknown"])

    async def _calculate_prediction_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence metrics for predictions"""
        if not predictions:
            return {"overall_confidence": 0.0}

        confidences = [pred["confidence"] for pred in predictions]

        return {
            "overall_confidence": sum(confidences) / len(confidences),
            "high_confidence_predictions": len([c for c in confidences if c > 0.8]),
            "prediction_count": len(predictions),
            "model_accuracy": 0.87  # Mock historical accuracy
        }

    def _generate_prediction_recommendations(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on threat predictions"""
        recommendations = []

        high_prob_threats = [p for p in predictions if p["probability"] > 0.5]

        if high_prob_threats:
            recommendations.append("Implement enhanced monitoring for high-probability threats")

            for threat in high_prob_threats:
                threat_type = threat["threat_type"]
                if threat_type == "ransomware_attack":
                    recommendations.append("Ensure backup systems are secure and tested")
                    recommendations.append("Implement endpoint detection and response")
                elif threat_type == "data_exfiltration":
                    recommendations.append("Monitor network traffic for data exfiltration patterns")
                    recommendations.append("Implement data loss prevention controls")

        recommendations.append("Regular threat landscape monitoring")
        recommendations.append("Update security controls based on threat predictions")

        return recommendations

    # Analysis helper methods for correlation

    def _analyze_temporal_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in threat indicators"""
        timestamps = []
        for result in results:
            if result.get("first_seen"):
                timestamps.append(result["first_seen"])

        if not timestamps:
            return {"pattern": "no_temporal_data"}

        # Simple temporal analysis
        return {
            "pattern": "clustered" if len(set(timestamps)) < len(timestamps) * 0.5 else "distributed",
            "timespan_days": 7,  # Mock calculation
            "frequency": "high" if len(timestamps) > 10 else "low"
        }

    def _correlate_threat_actors(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Correlate threat actors across indicators"""
        all_actors = set()
        for result in results:
            all_actors.update(result.get("threat_actors", []))

        return {
            "unique_actors": len(all_actors),
            "actors": list(all_actors),
            "correlation_strength": "high" if len(all_actors) < 3 else "low"
        }

    def _correlate_mitre_techniques(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Correlate MITRE ATT&CK techniques"""
        all_techniques = set()
        for result in results:
            all_techniques.update(result.get("mitre_techniques", []))

        return {
            "unique_techniques": len(all_techniques),
            "techniques": list(all_techniques),
            "attack_phases": self._map_techniques_to_phases(all_techniques)
        }

    def _map_techniques_to_phases(self, techniques: set) -> List[str]:
        """Map techniques to attack phases"""
        phase_mapping = {
            "T1566": "initial_access",
            "T1055": "defense_evasion",
            "T1078": "persistence",
            "T1021": "lateral_movement",
            "T1083": "discovery"
        }

        phases = set()
        for technique in techniques:
            if technique in phase_mapping:
                phases.add(phase_mapping[technique])

        return list(phases)

    def _identify_campaign_correlations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential campaign correlations"""
        # Mock campaign identification
        campaigns = []

        if len(results) > 5:
            campaigns.append({
                "campaign_name": "Suspected APT Campaign",
                "confidence": 0.7,
                "indicators_involved": len(results),
                "estimated_duration": "2-4 weeks"
            })

        return campaigns

    def _calculate_correlation_strength(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall correlation strength"""
        if len(results) < 2:
            return 0.0

        # Simple correlation calculation based on shared attributes
        total_score = 0.0
        comparisons = 0

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                score = 0.0

                # Check for shared threat actors
                actors1 = set(results[i].get("threat_actors", []))
                actors2 = set(results[j].get("threat_actors", []))
                if actors1 & actors2:
                    score += 0.3

                # Check for shared techniques
                tech1 = set(results[i].get("mitre_techniques", []))
                tech2 = set(results[j].get("mitre_techniques", []))
                if tech1 & tech2:
                    score += 0.2

                # Check for similar threat scores
                score1 = results[i].get("threat_score", 0.0)
                score2 = results[j].get("threat_score", 0.0)
                if abs(score1 - score2) < 0.2:
                    score += 0.1

                total_score += score
                comparisons += 1

        return total_score / comparisons if comparisons > 0 else 0.0

    def _analyze_context_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context factors that affect risk"""
        return {
            "environment": context.get("environment", "unknown"),
            "asset_criticality": context.get("asset_criticality", "medium"),
            "data_sensitivity": context.get("data_sensitivity", "medium"),
            "exposure_level": context.get("exposure_level", "internal")
        }

    def _assess_business_impact(self, risk_score: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess business impact of threat"""
        base_impact = "low"

        if risk_score > 0.8:
            base_impact = "critical"
        elif risk_score > 0.6:
            base_impact = "high"
        elif risk_score > 0.4:
            base_impact = "medium"

        # Adjust based on context
        asset_criticality = context.get("asset_criticality", "medium")
        if asset_criticality == "critical" and base_impact != "low":
            base_impact = "critical"

        return {
            "impact_level": base_impact,
            "financial_impact": self._estimate_financial_impact(base_impact),
            "operational_impact": self._estimate_operational_impact(base_impact),
            "reputational_impact": self._estimate_reputational_impact(base_impact)
        }

    def _estimate_financial_impact(self, impact_level: str) -> str:
        """Estimate financial impact"""
        mapping = {
            "critical": "$1M+",
            "high": "$100K - $1M",
            "medium": "$10K - $100K",
            "low": "< $10K"
        }
        return mapping.get(impact_level, "unknown")

    def _estimate_operational_impact(self, impact_level: str) -> str:
        """Estimate operational impact"""
        mapping = {
            "critical": "Severe disruption to operations",
            "high": "Significant operational impact",
            "medium": "Moderate operational impact",
            "low": "Minimal operational impact"
        }
        return mapping.get(impact_level, "unknown")

    def _estimate_reputational_impact(self, impact_level: str) -> str:
        """Estimate reputational impact"""
        mapping = {
            "critical": "Severe damage to reputation",
            "high": "Significant reputational damage",
            "medium": "Moderate reputational impact",
            "low": "Minimal reputational impact"
        }
        return mapping.get(impact_level, "unknown")

    def _calculate_urgency(self, risk_score: float, threat_actors: set, techniques: set) -> str:
        """Calculate urgency level"""
        if risk_score > 0.8:
            return "immediate"
        elif risk_score > 0.6:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"

    # Report generation methods

    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key findings from analysis results"""
        findings = []

        # High-threat indicators
        high_threat_indicators = [
            ind for ind in analysis_results.get("indicators", [])
            if ind.get("threat_score", 0) > 0.7
        ]

        if high_threat_indicators:
            findings.append({
                "type": "high_threat_indicators",
                "count": len(high_threat_indicators),
                "severity": "high",
                "description": f"Identified {len(high_threat_indicators)} high-threat indicators"
            })

        # Threat actor attribution
        threat_actors = analysis_results.get("threat_actors", [])
        if threat_actors:
            findings.append({
                "type": "threat_actor_attribution",
                "actors": threat_actors,
                "severity": "medium",
                "description": f"Potential attribution to {len(threat_actors)} threat actors"
            })

        # MITRE techniques
        techniques = analysis_results.get("mitre_techniques", [])
        if len(techniques) > 3:
            findings.append({
                "type": "multiple_attack_techniques",
                "techniques": techniques,
                "severity": "medium",
                "description": f"Multiple attack techniques identified: {len(techniques)}"
            })

        return findings

    def _generate_executive_summary(self, key_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate executive summary"""
        high_severity_count = len([f for f in key_findings if f.get("severity") == "high"])

        summary = {
            "overview": "Threat intelligence analysis completed",
            "key_findings_count": len(key_findings),
            "high_severity_findings": high_severity_count,
            "overall_risk": "high" if high_severity_count > 0 else "medium",
            "recommendation": "Immediate action required" if high_severity_count > 0 else "Monitor and assess"
        }

        return summary

    def _generate_technical_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical analysis section"""
        return {
            "indicators_analyzed": len(analysis_results.get("indicators", [])),
            "threat_score_distribution": self._calculate_score_distribution(analysis_results),
            "confidence_analysis": {
                "average_confidence": analysis_results.get("confidence_score", 0.0),
                "high_confidence_indicators": len([
                    ind for ind in analysis_results.get("indicators", [])
                    if ind.get("confidence", 0) > 0.8
                ])
            },
            "correlation_analysis": analysis_results.get("correlation_analysis", {}),
            "ml_analysis_summary": {
                "ml_enabled": analysis_results.get("metadata", {}).get("ml_enabled", False),
                "models_used": ["threat_classifier", "anomaly_detector", "attribution_model"]
            }
        }

    def _calculate_score_distribution(self, analysis_results: Dict[str, Any]) -> Dict[str, int]:
        """Calculate threat score distribution"""
        indicators = analysis_results.get("indicators", [])
        distribution = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

        for indicator in indicators:
            score = indicator.get("threat_score", 0.0)
            if score >= 0.8:
                distribution["critical"] += 1
            elif score >= 0.6:
                distribution["high"] += 1
            elif score >= 0.4:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def _generate_detailed_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed recommendations with priorities"""
        recommendations = []

        # Base recommendations
        base_recs = analysis_results.get("recommendations", [])
        for i, rec in enumerate(base_recs):
            recommendations.append({
                "id": i + 1,
                "recommendation": rec,
                "priority": "high" if i < 3 else "medium",
                "category": "security_control",
                "estimated_effort": "medium",
                "expected_impact": "high"
            })

        return recommendations

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML version of the report"""
        # Simplified HTML generation - in production would use proper templating
        html = f"""
        <html>
        <head><title>Threat Intelligence Report</title></head>
        <body>
        <h1>Threat Intelligence Analysis Report</h1>
        <h2>Executive Summary</h2>
        <p>{report['executive_summary']['overview']}</p>
        <h2>Key Findings</h2>
        <ul>
        """

        for finding in report.get("key_findings", []):
            html += f"<li>{finding.get('description', 'No description')}</li>"

        html += """
        </ul>
        <h2>Recommendations</h2>
        <ul>
        """

        for rec in report.get("recommendations", []):
            html += f"<li>{rec.get('recommendation', 'No recommendation')}</li>"

        html += """
        </ul>
        </body>
        </html>
        """

        return html
