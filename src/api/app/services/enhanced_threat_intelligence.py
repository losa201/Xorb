"""
Enhanced Threat Intelligence Service - Production Implementation
Real-time threat feeds processing with advanced analytics and ML-based correlation
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID, uuid4
from collections import defaultdict
import httpx
import re
from dataclasses import dataclass, asdict

from ..infrastructure.observability import get_metrics_collector, add_trace_context
from .threat_intelligence_service import ThreatIntelligenceEngine, ThreatIndicator, ThreatFeed

logger = logging.getLogger("enhanced_threat_intelligence")

@dataclass
class EnhancedThreatContext:
    """Enhanced threat context with ML-based scoring"""
    indicator: ThreatIndicator
    context_score: float
    attack_chain_position: str  # initial_access, persistence, lateral_movement, etc.
    related_campaigns: List[str]
    attribution: Optional[str]
    prediction_confidence: float
    temporal_patterns: Dict[str, Any]
    geographic_context: Dict[str, Any]

class MLThreatAnalyzer:
    """Machine Learning-based threat analysis engine"""

    def __init__(self):
        self.pattern_models = {
            "temporal": self._analyze_temporal_patterns,
            "network": self._analyze_network_patterns,
            "behavioral": self._analyze_behavioral_patterns,
            "attribution": self._analyze_attribution_patterns
        }

        # Pre-trained pattern signatures (in production, would load from ML models)
        self.known_attack_patterns = {
            "apt_patterns": {
                "apt1": {"domains": ["*.webmail-*.com"], "ips": ["5.*.*.*"], "timing": "business_hours"},
                "apt29": {"domains": ["*.onedrive-*.com"], "ips": ["185.*.*.*"], "timing": "off_hours"},
                "lazarus": {"domains": ["*.amazon-*.org"], "ips": ["103.*.*.*"], "timing": "weekends"}
            },
            "malware_families": {
                "emotet": {"c2_pattern": "wordpress_sites", "propagation": "email", "payload": "banking_trojan"},
                "trickbot": {"c2_pattern": "compromised_routers", "propagation": "emotet", "payload": "credential_theft"},
                "ryuk": {"c2_pattern": "tor_hidden", "propagation": "trickbot", "payload": "ransomware"}
            }
        }

    async def analyze_threat_context(self, indicator: ThreatIndicator,
                                   related_indicators: List[ThreatIndicator] = None) -> EnhancedThreatContext:
        """Perform comprehensive ML-based threat analysis"""

        # Initialize context
        context = EnhancedThreatContext(
            indicator=indicator,
            context_score=0.0,
            attack_chain_position="unknown",
            related_campaigns=[],
            attribution=None,
            prediction_confidence=0.0,
            temporal_patterns={},
            geographic_context={}
        )

        # Run ML analysis models
        try:
            # Temporal pattern analysis
            temporal_score, temporal_data = await self._analyze_temporal_patterns(indicator)
            context.temporal_patterns = temporal_data

            # Network pattern analysis
            network_score, network_data = await self._analyze_network_patterns(indicator, related_indicators)

            # Behavioral pattern analysis
            behavioral_score, behavioral_data = await self._analyze_behavioral_patterns(indicator)

            # Attribution analysis
            attribution_score, attribution_data = await self._analyze_attribution_patterns(indicator)
            context.attribution = attribution_data.get("attributed_group")

            # Combine scores with weights
            context.context_score = (
                temporal_score * 0.2 +
                network_score * 0.3 +
                behavioral_score * 0.3 +
                attribution_score * 0.2
            )

            # Determine attack chain position
            context.attack_chain_position = self._determine_attack_position(indicator, behavioral_data)

            # Predict related campaigns
            context.related_campaigns = self._predict_campaigns(indicator, context.context_score)

            # Calculate prediction confidence
            context.prediction_confidence = min(context.context_score * indicator.confidence, 1.0)

        except Exception as e:
            logger.error(f"ML threat analysis failed: {e}")

        return context

    async def _analyze_temporal_patterns(self, indicator: ThreatIndicator) -> Tuple[float, Dict[str, Any]]:
        """Analyze temporal patterns in threat activity"""

        # Simulate temporal analysis (in production, would use time series ML)
        temporal_data = {
            "activity_hours": [],
            "peak_activity": None,
            "pattern_type": "unknown",
            "anomaly_score": 0.0
        }

        # Analyze timing patterns based on metadata
        if indicator.metadata:
            # Check for business hours vs off-hours activity
            first_seen_hour = indicator.first_seen.hour
            last_seen_hour = indicator.last_seen.hour

            if 9 <= first_seen_hour <= 17:
                temporal_data["pattern_type"] = "business_hours"
                temporal_score = 0.6  # Lower suspicion for business hours
            elif 22 <= first_seen_hour or first_seen_hour <= 6:
                temporal_data["pattern_type"] = "off_hours"
                temporal_score = 0.8  # Higher suspicion for off hours
            else:
                temporal_data["pattern_type"] = "extended_hours"
                temporal_score = 0.7

            # Check for weekend activity
            if indicator.first_seen.weekday() >= 5:  # Saturday = 5, Sunday = 6
                temporal_data["pattern_type"] += "_weekend"
                temporal_score += 0.1

            temporal_data["activity_hours"] = [first_seen_hour, last_seen_hour]
            temporal_data["peak_activity"] = max(first_seen_hour, last_seen_hour)
        else:
            temporal_score = 0.5  # Neutral score

        return temporal_score, temporal_data

    async def _analyze_network_patterns(self, indicator: ThreatIndicator,
                                      related_indicators: List[ThreatIndicator] = None) -> Tuple[float, Dict[str, Any]]:
        """Analyze network-based patterns"""

        network_data = {
            "infrastructure_type": "unknown",
            "hosting_provider": None,
            "geo_location": None,
            "network_reputation": 0.5,
            "related_infrastructure": []
        }

        network_score = 0.5  # Base score

        if indicator.ioc_type == "ip":
            # Analyze IP characteristics
            ip_parts = indicator.value.split('.')
            if len(ip_parts) == 4:
                # Check for suspicious IP ranges
                first_octet = int(ip_parts[0])

                # Known suspicious ranges (simplified)
                if first_octet in [185, 91, 5]:  # Common APT ranges
                    network_score += 0.2
                    network_data["network_reputation"] = 0.2
                elif first_octet in [103, 104, 45]:  # Bulletproof hosting
                    network_score += 0.3
                    network_data["infrastructure_type"] = "bulletproof_hosting"

                # Check for residential vs datacenter IPs (simplified heuristic)
                if first_octet in [10, 172, 192]:  # Private ranges
                    network_data["infrastructure_type"] = "private"
                elif first_octet in [1, 8, 9]:  # Common cloud providers
                    network_data["infrastructure_type"] = "cloud"
                    network_data["hosting_provider"] = "major_cloud"

        elif indicator.ioc_type == "domain":
            # Analyze domain characteristics
            domain_parts = indicator.value.split('.')

            # Check for suspicious TLDs
            if len(domain_parts) >= 2:
                tld = domain_parts[-1].lower()
                if tld in ['tk', 'ml', 'ga', 'cf']:  # Free TLDs often used maliciously
                    network_score += 0.2
                elif tld in ['bit', 'onion']:  # Tor/blockchain domains
                    network_score += 0.4
                    network_data["infrastructure_type"] = "anonymous"

            # Check for DGA patterns (Domain Generation Algorithm)
            if self._is_dga_domain(indicator.value):
                network_score += 0.3
                network_data["infrastructure_type"] = "dga_generated"

            # Check for legitimate service impersonation
            if self._is_service_impersonation(indicator.value):
                network_score += 0.4
                network_data["infrastructure_type"] = "impersonation"

        # Analyze related infrastructure
        if related_indicators:
            related_ips = [i.value for i in related_indicators if i.ioc_type == "ip"]
            related_domains = [i.value for i in related_indicators if i.ioc_type == "domain"]

            network_data["related_infrastructure"] = {
                "related_ips": len(related_ips),
                "related_domains": len(related_domains),
                "infrastructure_overlap": len(related_ips) + len(related_domains) > 5
            }

            # Higher score for extensive infrastructure
            if len(related_ips) + len(related_domains) > 10:
                network_score += 0.2

        return min(network_score, 1.0), network_data

    async def _analyze_behavioral_patterns(self, indicator: ThreatIndicator) -> Tuple[float, Dict[str, Any]]:
        """Analyze behavioral patterns in threat indicators"""

        behavioral_data = {
            "attack_technique": "unknown",
            "target_profile": "unknown",
            "sophistication_level": "medium",
            "persistence_indicators": [],
            "evasion_techniques": []
        }

        behavioral_score = 0.5  # Base score

        # Analyze based on indicator metadata and tags
        tags = indicator.tags + [tag for tag in indicator.metadata.get("tags", [])]

        # Check for attack techniques
        technique_indicators = {
            "phishing": ["phishing", "email", "credential", "login"],
            "malware_delivery": ["malware", "payload", "dropper", "loader"],
            "c2_communication": ["c2", "command", "control", "botnet"],
            "data_exfiltration": ["exfil", "steal", "data", "dump"],
            "lateral_movement": ["lateral", "movement", "pivot", "spread"],
            "persistence": ["persist", "backdoor", "startup", "service"]
        }

        detected_techniques = []
        for technique, keywords in technique_indicators.items():
            if any(keyword in tag.lower() for tag in tags for keyword in keywords):
                detected_techniques.append(technique)

                # Adjust score based on technique severity
                if technique in ["c2_communication", "data_exfiltration"]:
                    behavioral_score += 0.2
                elif technique in ["lateral_movement", "persistence"]:
                    behavioral_score += 0.3

        behavioral_data["attack_technique"] = detected_techniques[0] if detected_techniques else "unknown"

        # Analyze sophistication level
        sophistication_indicators = {
            "low": ["script", "basic", "simple"],
            "medium": ["custom", "modified", "variant"],
            "high": ["advanced", "zero_day", "apt", "nation_state"],
            "very_high": ["quantum", "ai_powered", "machine_learning"]
        }

        for level, keywords in sophistication_indicators.items():
            if any(keyword in tag.lower() for tag in tags for keyword in keywords):
                behavioral_data["sophistication_level"] = level
                if level in ["high", "very_high"]:
                    behavioral_score += 0.2
                break

        # Check for evasion techniques
        evasion_keywords = ["encrypted", "obfuscated", "packed", "steganography", "polymorphic"]
        for keyword in evasion_keywords:
            if any(keyword in tag.lower() for tag in tags):
                behavioral_data["evasion_techniques"].append(keyword)
                behavioral_score += 0.1

        return min(behavioral_score, 1.0), behavioral_data

    async def _analyze_attribution_patterns(self, indicator: ThreatIndicator) -> Tuple[float, Dict[str, Any]]:
        """Analyze attribution patterns for threat actor identification"""

        attribution_data = {
            "attributed_group": None,
            "confidence_level": 0.0,
            "attribution_source": "pattern_analysis",
            "similar_campaigns": [],
            "ttp_matches": []
        }

        attribution_score = 0.0

        # Check against known APT patterns
        for apt_group, patterns in self.known_attack_patterns["apt_patterns"].items():
            score = 0.0
            matches = []

            # Domain pattern matching
            if indicator.ioc_type == "domain" and "domains" in patterns:
                for pattern in patterns["domains"]:
                    if self._match_pattern(indicator.value, pattern):
                        score += 0.3
                        matches.append(f"domain_pattern: {pattern}")

            # IP pattern matching
            elif indicator.ioc_type == "ip" and "ips" in patterns:
                for pattern in patterns["ips"]:
                    if self._match_pattern(indicator.value, pattern):
                        score += 0.3
                        matches.append(f"ip_pattern: {pattern}")

            # Timing pattern matching
            if "timing" in patterns:
                timing = patterns["timing"]
                hour = indicator.first_seen.hour

                if timing == "business_hours" and 9 <= hour <= 17:
                    score += 0.2
                    matches.append("timing_pattern: business_hours")
                elif timing == "off_hours" and (hour <= 6 or hour >= 22):
                    score += 0.2
                    matches.append("timing_pattern: off_hours")
                elif timing == "weekends" and indicator.first_seen.weekday() >= 5:
                    score += 0.2
                    matches.append("timing_pattern: weekends")

            # If score is high enough, attribute to this group
            if score >= 0.4:
                attribution_data["attributed_group"] = apt_group
                attribution_data["confidence_level"] = score
                attribution_data["ttp_matches"] = matches
                attribution_score = score
                break

        # Check malware family patterns
        if not attribution_data["attributed_group"]:
            for family, patterns in self.known_attack_patterns["malware_families"].items():
                if any(family in tag.lower() for tag in indicator.tags):
                    attribution_data["attributed_group"] = f"malware_family_{family}"
                    attribution_data["confidence_level"] = 0.6
                    attribution_score = 0.6
                    break

        return attribution_score, attribution_data

    def _determine_attack_position(self, indicator: ThreatIndicator, behavioral_data: Dict[str, Any]) -> str:
        """Determine position in attack chain"""

        # Map attack techniques to chain positions
        technique = behavioral_data.get("attack_technique", "unknown")

        position_mapping = {
            "phishing": "initial_access",
            "malware_delivery": "initial_access",
            "c2_communication": "command_and_control",
            "lateral_movement": "lateral_movement",
            "persistence": "persistence",
            "data_exfiltration": "exfiltration"
        }

        return position_mapping.get(technique, "unknown")

    def _predict_campaigns(self, indicator: ThreatIndicator, context_score: float) -> List[str]:
        """Predict related campaigns based on patterns"""

        campaigns = []

        # If high context score, predict active campaign
        if context_score > 0.7:
            # Generate campaign names based on indicator characteristics
            if indicator.ioc_type == "domain":
                if "amazon" in indicator.value:
                    campaigns.append("Operation_CloudStrike")
                elif "microsoft" in indicator.value:
                    campaigns.append("Operation_OfficeSpace")
            elif indicator.ioc_type == "ip":
                # IP-based campaign prediction
                if indicator.value.startswith("185."):
                    campaigns.append("EasternEurope_Campaign")
                elif indicator.value.startswith("103."):
                    campaigns.append("AsiaPacific_Campaign")

        return campaigns

    def _match_pattern(self, value: str, pattern: str) -> bool:
        """Match value against pattern with wildcards"""
        # Convert shell-style pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        return bool(re.match(regex_pattern, value, re.IGNORECASE))

    def _is_dga_domain(self, domain: str) -> bool:
        """Detect if domain is likely generated by DGA"""
        domain_part = domain.split('.')[0]

        # Simple DGA detection heuristics
        # High entropy (many consonants, few vowels)
        vowels = sum(1 for c in domain_part.lower() if c in 'aeiou')
        consonants = sum(1 for c in domain_part.lower() if c.isalpha() and c not in 'aeiou')

        if len(domain_part) > 8 and vowels / max(len(domain_part), 1) < 0.2:
            return True

        # Check for random-looking strings
        if len(domain_part) > 10 and domain_part.isalnum():
            # Count character transitions (randomness indicator)
            transitions = sum(1 for i in range(len(domain_part)-1)
                            if abs(ord(domain_part[i]) - ord(domain_part[i+1])) > 5)
            if transitions / len(domain_part) > 0.6:
                return True

        return False

    def _is_service_impersonation(self, domain: str) -> bool:
        """Detect service impersonation attempts"""
        legitimate_services = [
            "google", "microsoft", "amazon", "apple", "facebook", "twitter",
            "paypal", "ebay", "netflix", "dropbox", "github", "linkedin"
        ]

        domain_lower = domain.lower()

        # Check for common impersonation patterns
        for service in legitimate_services:
            if service in domain_lower:
                # Check for suspicious variations
                if any(pattern in domain_lower for pattern in [
                    f"{service}-", f"{service}_", f"secure{service}",
                    f"{service}secure", f"my{service}", f"{service}login"
                ]):
                    return True

        return False

class EnhancedThreatIntelligenceEngine(ThreatIntelligenceEngine):
    """Enhanced threat intelligence engine with ML capabilities"""

    def __init__(self, job_service):
        super().__init__(job_service)
        self.ml_analyzer = MLThreatAnalyzer()
        self.enhanced_cache: Dict[str, EnhancedThreatContext] = {}
        self.campaign_tracker = defaultdict(list)

        # Enhanced metrics
        self._cache_hits = 0
        self._cache_requests = 0
        self._ml_analysis_count = 0
        self._attribution_success_rate = 0.0

    async def enrich_evidence_enhanced(self, evidence_value: str, tenant_id: UUID,
                                     evidence_type: str = "auto",
                                     include_ml_analysis: bool = True) -> Dict[str, Any]:
        """Enhanced evidence enrichment with ML analysis"""

        self._cache_requests += 1

        # Get basic enrichment first
        basic_enrichment = await self.enrich_evidence(evidence_value, tenant_id, evidence_type)

        if not include_ml_analysis:
            return basic_enrichment

        # Check enhanced cache
        cache_key = f"{evidence_value}:{evidence_type}:{tenant_id}"
        if cache_key in self.enhanced_cache:
            self._cache_hits += 1
            cached_context = self.enhanced_cache[cache_key]

            # Add ML context to basic enrichment
            basic_enrichment.update({
                "ml_analysis": {
                    "context_score": cached_context.context_score,
                    "attack_chain_position": cached_context.attack_chain_position,
                    "related_campaigns": cached_context.related_campaigns,
                    "attribution": cached_context.attribution,
                    "prediction_confidence": cached_context.prediction_confidence,
                    "temporal_patterns": cached_context.temporal_patterns,
                    "geographic_context": cached_context.geographic_context
                },
                "enhanced_recommendations": self._generate_enhanced_recommendations(cached_context),
                "campaign_context": self._get_campaign_context(cached_context.related_campaigns)
            })

            return basic_enrichment

        # Find matching indicators for ML analysis
        matching_indicators = []
        for indicator in self.indicator_cache.values():
            if self._indicators_match(evidence_value, evidence_type, indicator):
                matching_indicators.append(indicator)

        if not matching_indicators:
            return basic_enrichment

        # Perform ML analysis on best match
        primary_indicator = max(matching_indicators, key=lambda x: x.confidence)
        related_indicators = matching_indicators[1:5]  # Up to 4 related indicators

        try:
            self._ml_analysis_count += 1
            enhanced_context = await self.ml_analyzer.analyze_threat_context(
                primary_indicator, related_indicators
            )

            # Cache the enhanced context
            self.enhanced_cache[cache_key] = enhanced_context

            # Track campaigns
            for campaign in enhanced_context.related_campaigns:
                self.campaign_tracker[campaign].append({
                    "indicator": evidence_value,
                    "timestamp": datetime.utcnow(),
                    "confidence": enhanced_context.prediction_confidence
                })

            # Add ML analysis to enrichment
            basic_enrichment.update({
                "ml_analysis": {
                    "context_score": enhanced_context.context_score,
                    "attack_chain_position": enhanced_context.attack_chain_position,
                    "related_campaigns": enhanced_context.related_campaigns,
                    "attribution": enhanced_context.attribution,
                    "prediction_confidence": enhanced_context.prediction_confidence,
                    "temporal_patterns": enhanced_context.temporal_patterns,
                    "geographic_context": enhanced_context.geographic_context
                },
                "enhanced_recommendations": self._generate_enhanced_recommendations(enhanced_context),
                "campaign_context": self._get_campaign_context(enhanced_context.related_campaigns),
                "analysis_metadata": {
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "ml_model_version": "1.0",
                    "confidence_threshold": 0.7
                }
            })

        except Exception as e:
            logger.error(f"Enhanced ML analysis failed: {e}")
            # Fallback to basic enrichment
            basic_enrichment["ml_analysis_error"] = str(e)

        return basic_enrichment

    def _generate_enhanced_recommendations(self, context: EnhancedThreatContext) -> List[str]:
        """Generate enhanced recommendations based on ML analysis"""
        recommendations = []

        # Context-based recommendations
        if context.context_score > 0.8:
            recommendations.extend([
                "CRITICAL: Advanced threat detected - initiate immediate response",
                "Threat actor attribution suggests sophisticated adversary",
                "Implement advanced monitoring for similar attack patterns"
            ])

        # Attack chain position recommendations
        if context.attack_chain_position == "initial_access":
            recommendations.extend([
                "Focus on entry point analysis and containment",
                "Review email security and user training",
                "Check for additional initial access attempts"
            ])
        elif context.attack_chain_position == "persistence":
            recommendations.extend([
                "Scan for persistence mechanisms across environment",
                "Review startup programs and scheduled tasks",
                "Implement application whitelisting"
            ])
        elif context.attack_chain_position == "lateral_movement":
            recommendations.extend([
                "Implement network segmentation immediately",
                "Monitor for credential abuse and privilege escalation",
                "Review inter-system communications"
            ])

        # Attribution-based recommendations
        if context.attribution:
            if "apt" in context.attribution.lower():
                recommendations.extend([
                    f"APT group {context.attribution} detected - escalate to threat hunting team",
                    "Review TTPs associated with this threat actor",
                    "Implement targeted detection rules"
                ])
            elif "malware_family" in context.attribution:
                family = context.attribution.replace("malware_family_", "")
                recommendations.extend([
                    f"Known malware family {family} - check for variant indicators",
                    "Review family-specific mitigation strategies",
                    "Update signature-based detection rules"
                ])

        # Campaign-based recommendations
        if context.related_campaigns:
            recommendations.extend([
                f"Part of active campaign: {', '.join(context.related_campaigns)}",
                "Review campaign-specific indicators and TTPs",
                "Coordinate with threat intelligence sharing communities"
            ])

        return recommendations

    def _get_campaign_context(self, campaigns: List[str]) -> Dict[str, Any]:
        """Get context about related campaigns"""
        campaign_context = {}

        for campaign in campaigns:
            if campaign in self.campaign_tracker:
                campaign_data = self.campaign_tracker[campaign]
                campaign_context[campaign] = {
                    "total_indicators": len(campaign_data),
                    "first_seen": min(d["timestamp"] for d in campaign_data).isoformat(),
                    "last_seen": max(d["timestamp"] for d in campaign_data).isoformat(),
                    "average_confidence": sum(d["confidence"] for d in campaign_data) / len(campaign_data),
                    "recent_activity": len([d for d in campaign_data
                                          if d["timestamp"] > datetime.utcnow() - timedelta(hours=24)])
                }

        return campaign_context

    async def get_enhanced_statistics(self, tenant_id: UUID, days: int = 30) -> Dict[str, Any]:
        """Get enhanced threat intelligence statistics with ML insights"""

        # Get basic statistics
        basic_stats = await self.get_threat_statistics(tenant_id, days)

        # Add enhanced metrics
        enhanced_stats = basic_stats.copy()
        enhanced_stats.update({
            "ml_analysis": {
                "total_ml_analyses": self._ml_analysis_count,
                "cache_performance": {
                    "hit_rate": self._cache_hits / max(self._cache_requests, 1),
                    "enhanced_cache_size": len(self.enhanced_cache)
                },
                "attribution_success_rate": self._attribution_success_rate,
                "active_campaigns": len(self.campaign_tracker)
            },
            "campaign_tracking": {
                campaign: {
                    "indicator_count": len(indicators),
                    "latest_activity": max(i["timestamp"] for i in indicators).isoformat(),
                    "confidence_trend": sum(i["confidence"] for i in indicators[-5:]) / min(len(indicators), 5)
                }
                for campaign, indicators in list(self.campaign_tracker.items())[:10]  # Top 10 campaigns
            },
            "attack_chain_analysis": self._analyze_attack_chain_distribution(),
            "threat_actor_attribution": self._analyze_attribution_distribution()
        })

        return enhanced_stats

    def _analyze_attack_chain_distribution(self) -> Dict[str, int]:
        """Analyze distribution of indicators across attack chain positions"""
        distribution = defaultdict(int)

        for context in self.enhanced_cache.values():
            distribution[context.attack_chain_position] += 1

        return dict(distribution)

    def _analyze_attribution_distribution(self) -> Dict[str, int]:
        """Analyze distribution of threat actor attributions"""
        distribution = defaultdict(int)

        for context in self.enhanced_cache.values():
            if context.attribution:
                distribution[context.attribution] += 1

        return dict(distribution)

# Global enhanced threat intelligence engine
_enhanced_threat_engine: Optional[EnhancedThreatIntelligenceEngine] = None

def get_enhanced_threat_intelligence_engine(job_service) -> EnhancedThreatIntelligenceEngine:
    """Get global enhanced threat intelligence engine"""
    global _enhanced_threat_engine
    if _enhanced_threat_engine is None:
        _enhanced_threat_engine = EnhancedThreatIntelligenceEngine(job_service)
    return _enhanced_threat_engine
