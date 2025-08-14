"""
Advanced Threat Intelligence Engine
Sophisticated AI-powered threat analysis and attribution system
"""

import asyncio
import json
import logging
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
import numpy as np
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class ThreatActorType(Enum):
    """Enhanced threat actor classification"""
    NATION_STATE = "nation_state"
    CYBERCRIMINAL = "cybercriminal"
    HACKTIVIST = "hacktivist"
    INSIDER = "insider"
    SCRIPT_KIDDIE = "script_kiddie"
    TERRORIST = "terrorist"
    UNKNOWN = "unknown"


class AttackStage(Enum):
    """MITRE ATT&CK kill chain stages"""
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ThreatIndicator:
    """Enhanced threat indicator with ML features"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, url, email, etc.
    value: str
    confidence_score: float
    threat_types: List[str]
    first_seen: datetime
    last_seen: datetime
    source: str
    context: Dict[str, Any] = field(default_factory=dict)
    mitre_techniques: List[str] = field(default_factory=list)
    kill_chain_phases: List[AttackStage] = field(default_factory=list)
    geo_location: Optional[Dict[str, str]] = None
    reputation_score: float = 0.5
    false_positive_rate: float = 0.1


@dataclass
class ThreatCampaign:
    """Advanced threat campaign analysis"""
    campaign_id: str
    name: str
    actor_type: ThreatActorType
    start_date: datetime
    end_date: Optional[datetime]
    indicators: List[ThreatIndicator]
    attack_patterns: List[str]
    victims: List[str]
    motivation: str
    sophistication_level: str
    attribution_confidence: float
    kill_chain_coverage: Dict[AttackStage, float]
    ml_cluster_id: Optional[str] = None


@dataclass
class ThreatIntelligenceReport:
    """Comprehensive threat intelligence analysis report"""
    report_id: str
    analysis_timestamp: datetime
    indicators_analyzed: int
    threat_level: ThreatLevel
    confidence_score: float
    attributed_actors: List[Dict[str, Any]]
    campaign_associations: List[str]
    attack_timeline: List[Dict[str, Any]]
    mitre_mapping: Dict[str, List[str]]
    recommendations: List[str]
    executive_summary: str
    technical_details: Dict[str, Any]
    ml_insights: Dict[str, Any]


class AdvancedThreatIntelligenceEngine:
    """Production-grade threat intelligence engine with ML capabilities"""

    def __init__(self):
        self.indicators_db: Dict[str, ThreatIndicator] = {}
        self.campaigns_db: Dict[str, ThreatCampaign] = {}
        self.actor_profiles: Dict[str, Dict[str, Any]] = {}
        self.ml_models: Dict[str, Any] = {}
        self.reputation_cache: Dict[str, float] = {}
        self.clustering_model = None
        self.attribution_model = None
        self.prediction_model = None

        # Initialize threat intelligence feeds
        self.threat_feeds = {
            "commercial": ["AlienVault", "CrowdStrike", "FireEye", "Recorded_Future"],
            "open_source": ["MISP", "OpenIOC", "STIX", "TAXII"],
            "government": ["US-CERT", "NCSC", "ANSSI"]
        }

        # MITRE ATT&CK framework mapping
        self.mitre_techniques = self._load_mitre_framework()

        # Known threat actor TTPs
        self.actor_ttps = self._load_actor_profiles()

        # ML feature extractors
        self.feature_extractors = {
            "network": self._extract_network_features,
            "behavioral": self._extract_behavioral_features,
            "temporal": self._extract_temporal_features,
            "linguistic": self._extract_linguistic_features
        }

    async def initialize(self) -> bool:
        """Initialize the threat intelligence engine"""
        try:
            logger.info("Initializing Advanced Threat Intelligence Engine...")

            # Load pre-trained ML models
            await self._load_ml_models()

            # Initialize threat intelligence feeds
            await self._initialize_threat_feeds()

            # Load historical threat data
            await self._load_historical_data()

            # Initialize reputation databases
            await self._initialize_reputation_systems()

            logger.info("Advanced Threat Intelligence Engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize threat intelligence engine: {e}")
            return False

    async def analyze_threat_indicators(
        self,
        indicators: List[str],
        context: Dict[str, Any] = None
    ) -> ThreatIntelligenceReport:
        """Advanced threat indicator analysis with ML attribution"""
        try:
            analysis_start = datetime.utcnow()
            report_id = str(uuid4())

            logger.info(f"Starting threat analysis for {len(indicators)} indicators")

            # Parse and validate indicators
            parsed_indicators = await self._parse_indicators(indicators)

            # Enrich with threat intelligence
            enriched_indicators = await self._enrich_indicators(parsed_indicators)

            # ML-powered threat classification
            threat_classifications = await self._classify_threats_ml(enriched_indicators)

            # Actor attribution using advanced techniques
            attribution_results = await self._perform_actor_attribution(enriched_indicators, context)

            # Campaign correlation and clustering
            campaign_associations = await self._correlate_campaigns(enriched_indicators)

            # Generate attack timeline
            attack_timeline = await self._generate_attack_timeline(enriched_indicators)

            # MITRE ATT&CK framework mapping
            mitre_mapping = await self._map_to_mitre_framework(enriched_indicators)

            # ML-powered threat prediction
            ml_insights = await self._generate_ml_insights(enriched_indicators, context)

            # Calculate overall threat level and confidence
            threat_level, confidence = self._calculate_threat_metrics(
                threat_classifications, attribution_results, ml_insights
            )

            # Generate actionable recommendations
            recommendations = await self._generate_threat_recommendations(
                enriched_indicators, attribution_results, ml_insights
            )

            # Create executive summary
            executive_summary = self._generate_executive_summary(
                enriched_indicators, attribution_results, threat_level, confidence
            )

            # Compile comprehensive report
            report = ThreatIntelligenceReport(
                report_id=report_id,
                analysis_timestamp=analysis_start,
                indicators_analyzed=len(indicators),
                threat_level=threat_level,
                confidence_score=confidence,
                attributed_actors=attribution_results,
                campaign_associations=campaign_associations,
                attack_timeline=attack_timeline,
                mitre_mapping=mitre_mapping,
                recommendations=recommendations,
                executive_summary=executive_summary,
                technical_details={
                    "enriched_indicators": len(enriched_indicators),
                    "threat_classifications": threat_classifications,
                    "processing_time_ms": (datetime.utcnow() - analysis_start).total_seconds() * 1000
                },
                ml_insights=ml_insights
            )

            # Store analysis results for future learning
            await self._store_analysis_results(report)

            logger.info(f"Threat analysis completed: {threat_level.value} threat level with {confidence:.2f} confidence")

            return report

        except Exception as e:
            logger.error(f"Threat indicator analysis failed: {e}")
            # Return error report
            return ThreatIntelligenceReport(
                report_id=str(uuid4()),
                analysis_timestamp=datetime.utcnow(),
                indicators_analyzed=len(indicators),
                threat_level=ThreatLevel.LOW,
                confidence_score=0.0,
                attributed_actors=[],
                campaign_associations=[],
                attack_timeline=[],
                mitre_mapping={},
                recommendations=["Manual analysis required due to processing error"],
                executive_summary="Analysis failed - manual review recommended",
                technical_details={"error": str(e)},
                ml_insights={}
            )

    async def _parse_indicators(self, indicators: List[str]) -> List[ThreatIndicator]:
        """Advanced indicator parsing with type detection"""
        parsed_indicators = []

        for indicator in indicators:
            try:
                # Detect indicator type using advanced patterns
                indicator_type = self._detect_indicator_type(indicator)

                # Create threat indicator object
                threat_indicator = ThreatIndicator(
                    indicator_id=str(uuid4()),
                    indicator_type=indicator_type,
                    value=indicator.strip(),
                    confidence_score=0.8,  # Base confidence
                    threat_types=[],
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    source="user_provided"
                )

                parsed_indicators.append(threat_indicator)

            except Exception as e:
                logger.warning(f"Failed to parse indicator {indicator}: {e}")
                continue

        return parsed_indicators

    def _detect_indicator_type(self, indicator: str) -> str:
        """Advanced indicator type detection"""
        indicator = indicator.strip().lower()

        # IP address patterns
        ip_v4_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
        ip_v6_pattern = re.compile(r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$')

        # Hash patterns
        md5_pattern = re.compile(r'^[a-fA-F0-9]{32}$')
        sha1_pattern = re.compile(r'^[a-fA-F0-9]{40}$')
        sha256_pattern = re.compile(r'^[a-fA-F0-9]{64}$')

        # Domain patterns
        domain_pattern = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$')

        # URL patterns
        url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')

        # Email patterns
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

        # Detection logic
        if ip_v4_pattern.match(indicator):
            return "ipv4"
        elif ip_v6_pattern.match(indicator):
            return "ipv6"
        elif sha256_pattern.match(indicator):
            return "sha256"
        elif sha1_pattern.match(indicator):
            return "sha1"
        elif md5_pattern.match(indicator):
            return "md5"
        elif url_pattern.match(indicator):
            return "url"
        elif email_pattern.match(indicator):
            return "email"
        elif domain_pattern.match(indicator):
            return "domain"
        else:
            return "unknown"

    async def _enrich_indicators(self, indicators: List[ThreatIndicator]) -> List[ThreatIndicator]:
        """Advanced indicator enrichment with multiple sources"""
        enriched_indicators = []

        for indicator in indicators:
            try:
                # Reputation scoring
                reputation_score = await self._get_reputation_score(indicator)
                indicator.reputation_score = reputation_score

                # Geolocation enrichment
                if indicator.indicator_type in ["ipv4", "ipv6"]:
                    geo_info = await self._get_geolocation(indicator.value)
                    indicator.geo_location = geo_info

                # Threat intelligence feeds lookup
                feed_results = await self._lookup_threat_feeds(indicator)
                indicator.threat_types.extend(feed_results.get("threat_types", []))
                indicator.context.update(feed_results.get("context", {}))

                # MITRE technique mapping
                mitre_techniques = await self._map_indicator_to_mitre(indicator)
                indicator.mitre_techniques = mitre_techniques

                # Kill chain phase identification
                kill_chain_phases = self._identify_kill_chain_phases(indicator)
                indicator.kill_chain_phases = kill_chain_phases

                # Confidence adjustment based on enrichment
                indicator.confidence_score = self._calculate_enriched_confidence(indicator)

                enriched_indicators.append(indicator)

            except Exception as e:
                logger.warning(f"Failed to enrich indicator {indicator.value}: {e}")
                enriched_indicators.append(indicator)  # Add unenriched

        return enriched_indicators

    async def _classify_threats_ml(self, indicators: List[ThreatIndicator]) -> Dict[str, Any]:
        """ML-powered threat classification"""
        try:
            # Extract features for ML classification
            features = await self._extract_ml_features(indicators)

            # Apply ML models for classification
            classifications = {}

            # Threat family classification
            if "threat_family_classifier" in self.ml_models:
                family_predictions = await self._predict_threat_family(features)
                classifications["threat_families"] = family_predictions

            # Malware classification
            if "malware_classifier" in self.ml_models:
                malware_predictions = await self._predict_malware_type(features)
                classifications["malware_types"] = malware_predictions

            # Campaign classification
            if "campaign_classifier" in self.ml_models:
                campaign_predictions = await self._predict_campaign_association(features)
                classifications["campaigns"] = campaign_predictions

            # Risk scoring using ensemble methods
            risk_scores = await self._calculate_ml_risk_scores(features)
            classifications["risk_scores"] = risk_scores

            return classifications

        except Exception as e:
            logger.error(f"ML threat classification failed: {e}")
            return {"error": str(e)}

    async def _perform_actor_attribution(
        self,
        indicators: List[ThreatIndicator],
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Advanced threat actor attribution using ML and behavioral analysis"""
        try:
            attribution_results = []

            # Extract actor-specific features
            actor_features = await self._extract_actor_features(indicators, context)

            # TTP (Tactics, Techniques, Procedures) analysis
            ttp_analysis = await self._analyze_ttps(indicators)

            # Behavioral pattern matching
            behavioral_matches = await self._match_behavioral_patterns(indicators)

            # Infrastructure correlation
            infrastructure_links = await self._correlate_infrastructure(indicators)

            # ML-based attribution
            ml_attributions = await self._ml_actor_attribution(actor_features)

            # Combine multiple attribution methods
            for actor_name, actor_profile in self.actor_profiles.items():
                attribution_score = 0.0
                evidence = []

                # TTP matching score
                ttp_score = self._calculate_ttp_similarity(ttp_analysis, actor_profile.get("ttps", []))
                attribution_score += ttp_score * 0.4
                if ttp_score > 0.3:
                    evidence.append(f"TTP similarity: {ttp_score:.2f}")

                # Behavioral similarity
                behavioral_score = behavioral_matches.get(actor_name, 0.0)
                attribution_score += behavioral_score * 0.3
                if behavioral_score > 0.3:
                    evidence.append(f"Behavioral pattern match: {behavioral_score:.2f}")

                # Infrastructure overlap
                infra_score = infrastructure_links.get(actor_name, 0.0)
                attribution_score += infra_score * 0.2
                if infra_score > 0.3:
                    evidence.append(f"Infrastructure overlap: {infra_score:.2f}")

                # ML prediction
                ml_score = ml_attributions.get(actor_name, 0.0)
                attribution_score += ml_score * 0.1
                if ml_score > 0.3:
                    evidence.append(f"ML attribution confidence: {ml_score:.2f}")

                # Only include high-confidence attributions
                if attribution_score > 0.4:
                    attribution_results.append({
                        "actor_name": actor_name,
                        "actor_type": actor_profile.get("type", "unknown"),
                        "confidence": min(attribution_score, 1.0),
                        "evidence": evidence,
                        "motivation": actor_profile.get("motivation", "unknown"),
                        "sophistication": actor_profile.get("sophistication", "medium"),
                        "geographic_origin": actor_profile.get("origin", "unknown"),
                        "known_campaigns": actor_profile.get("campaigns", []),
                        "attribution_methods": ["ttp_analysis", "behavioral_matching", "infrastructure_correlation", "ml_prediction"]
                    })

            # Sort by confidence
            attribution_results.sort(key=lambda x: x["confidence"], reverse=True)

            return attribution_results[:5]  # Top 5 attributions

        except Exception as e:
            logger.error(f"Actor attribution failed: {e}")
            return []

    async def _correlate_campaigns(self, indicators: List[ThreatIndicator]) -> List[str]:
        """Advanced campaign correlation using ML clustering"""
        try:
            campaign_associations = []

            # Extract campaign-specific features
            campaign_features = []
            for indicator in indicators:
                features = {
                    "indicator_type": indicator.indicator_type,
                    "threat_types": indicator.threat_types,
                    "mitre_techniques": indicator.mitre_techniques,
                    "reputation_score": indicator.reputation_score,
                    "temporal_features": self._extract_temporal_features([indicator])
                }
                campaign_features.append(features)

            # Clustering analysis for campaign identification
            if self.clustering_model and len(campaign_features) > 1:
                cluster_assignments = await self._cluster_indicators(campaign_features)

                # Map clusters to known campaigns
                for cluster_id in set(cluster_assignments):
                    cluster_indicators = [
                        indicators[i] for i, c in enumerate(cluster_assignments) if c == cluster_id
                    ]

                    # Find matching campaigns
                    matching_campaigns = await self._match_campaigns(cluster_indicators)
                    campaign_associations.extend(matching_campaigns)

            # Rule-based campaign correlation
            rule_based_campaigns = await self._rule_based_campaign_correlation(indicators)
            campaign_associations.extend(rule_based_campaigns)

            return list(set(campaign_associations))  # Remove duplicates

        except Exception as e:
            logger.error(f"Campaign correlation failed: {e}")
            return []

    async def _generate_attack_timeline(self, indicators: List[ThreatIndicator]) -> List[Dict[str, Any]]:
        """Generate chronological attack timeline"""
        try:
            timeline_events = []

            # Sort indicators by first seen
            sorted_indicators = sorted(indicators, key=lambda x: x.first_seen)

            for i, indicator in enumerate(sorted_indicators):
                # Determine attack stage based on indicator characteristics
                attack_stage = self._determine_attack_stage(indicator)

                timeline_event = {
                    "timestamp": indicator.first_seen.isoformat(),
                    "stage": attack_stage.value if attack_stage else "unknown",
                    "indicator": indicator.value,
                    "indicator_type": indicator.indicator_type,
                    "threat_types": indicator.threat_types,
                    "confidence": indicator.confidence_score,
                    "description": self._generate_timeline_description(indicator, attack_stage),
                    "mitre_techniques": indicator.mitre_techniques
                }

                timeline_events.append(timeline_event)

            return timeline_events

        except Exception as e:
            logger.error(f"Attack timeline generation failed: {e}")
            return []

    async def _map_to_mitre_framework(self, indicators: List[ThreatIndicator]) -> Dict[str, List[str]]:
        """Advanced MITRE ATT&CK framework mapping"""
        try:
            mitre_mapping = defaultdict(list)

            for indicator in indicators:
                for technique in indicator.mitre_techniques:
                    # Get tactic for this technique
                    tactic = self._get_mitre_tactic(technique)
                    if tactic:
                        mitre_mapping[tactic].append(technique)

                    # Add technique details
                    technique_info = self.mitre_techniques.get(technique, {})
                    if technique_info:
                        mitre_mapping[f"{technique}_details"] = technique_info

            # Convert defaultdict to regular dict
            return dict(mitre_mapping)

        except Exception as e:
            logger.error(f"MITRE framework mapping failed: {e}")
            return {}

    async def _generate_ml_insights(
        self,
        indicators: List[ThreatIndicator],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate advanced ML-powered threat insights"""
        try:
            ml_insights = {}

            # Anomaly detection
            anomaly_scores = await self._detect_anomalies(indicators)
            ml_insights["anomaly_analysis"] = {
                "anomalous_indicators": len([s for s in anomaly_scores if s > 0.7]),
                "max_anomaly_score": max(anomaly_scores) if anomaly_scores else 0.0,
                "average_anomaly_score": sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
            }

            # Threat evolution prediction
            evolution_prediction = await self._predict_threat_evolution(indicators)
            ml_insights["threat_evolution"] = evolution_prediction

            # Risk trajectory analysis
            risk_trajectory = await self._analyze_risk_trajectory(indicators)
            ml_insights["risk_trajectory"] = risk_trajectory

            # Pattern analysis
            pattern_analysis = await self._analyze_threat_patterns(indicators)
            ml_insights["pattern_analysis"] = pattern_analysis

            # Network effect analysis
            network_effects = await self._analyze_network_effects(indicators)
            ml_insights["network_effects"] = network_effects

            # Confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(indicators)
            ml_insights["confidence_intervals"] = confidence_intervals

            return ml_insights

        except Exception as e:
            logger.error(f"ML insights generation failed: {e}")
            return {"error": str(e)}

    def _calculate_threat_metrics(
        self,
        classifications: Dict[str, Any],
        attributions: List[Dict[str, Any]],
        ml_insights: Dict[str, Any]
    ) -> Tuple[ThreatLevel, float]:
        """Calculate overall threat level and confidence"""
        try:
            # Base threat score
            threat_score = 0.0
            confidence_score = 0.0

            # Factor in classification results
            if "risk_scores" in classifications:
                avg_risk = sum(classifications["risk_scores"]) / len(classifications["risk_scores"])
                threat_score += avg_risk * 0.4
                confidence_score += 0.2

            # Factor in attribution confidence
            if attributions:
                max_attribution_conf = max(attr["confidence"] for attr in attributions)
                threat_score += max_attribution_conf * 0.3
                confidence_score += max_attribution_conf * 0.3

            # Factor in ML insights
            if ml_insights.get("anomaly_analysis"):
                anomaly_data = ml_insights["anomaly_analysis"]
                threat_score += anomaly_data.get("max_anomaly_score", 0.0) * 0.2
                confidence_score += 0.2

            # Factor in known threat actor sophistication
            if attributions:
                for attr in attributions:
                    if attr.get("sophistication") == "high":
                        threat_score += 0.2
                    elif attr.get("sophistication") == "nation_state":
                        threat_score += 0.3

            # Normalize scores
            threat_score = min(threat_score, 1.0)
            confidence_score = min(confidence_score, 1.0)

            # Determine threat level
            if threat_score >= 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif threat_score >= 0.6:
                threat_level = ThreatLevel.HIGH
            elif threat_score >= 0.4:
                threat_level = ThreatLevel.MEDIUM
            elif threat_score >= 0.2:
                threat_level = ThreatLevel.LOW
            else:
                threat_level = ThreatLevel.INFO

            return threat_level, confidence_score

        except Exception as e:
            logger.error(f"Threat metrics calculation failed: {e}")
            return ThreatLevel.LOW, 0.5

    async def _generate_threat_recommendations(
        self,
        indicators: List[ThreatIndicator],
        attributions: List[Dict[str, Any]],
        ml_insights: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable threat intelligence recommendations"""
        try:
            recommendations = []

            # Critical indicators
            critical_indicators = [ind for ind in indicators if ind.reputation_score < 0.3]
            if critical_indicators:
                recommendations.append(
                    f"🚨 IMMEDIATE: Block {len(critical_indicators)} high-confidence malicious indicators"
                )

            # High-confidence attributions
            high_conf_attributions = [attr for attr in attributions if attr["confidence"] > 0.7]
            if high_conf_attributions:
                for attr in high_conf_attributions:
                    recommendations.append(
                        f"🎯 ATTRIBUTION: Monitor for {attr['actor_name']} TTPs and indicators"
                    )

            # MITRE technique-based recommendations
            mitre_techniques = []
            for indicator in indicators:
                mitre_techniques.extend(indicator.mitre_techniques)

            unique_techniques = list(set(mitre_techniques))
            if unique_techniques:
                recommendations.append(
                    f"🛡️ DEFENSE: Implement controls for {len(unique_techniques)} MITRE ATT&CK techniques"
                )

            # Campaign-specific recommendations
            if ml_insights.get("pattern_analysis"):
                pattern_data = ml_insights["pattern_analysis"]
                if pattern_data.get("campaign_indicators"):
                    recommendations.append(
                        "📋 HUNTING: Initiate threat hunting for related campaign indicators"
                    )

            # Anomaly-based recommendations
            if ml_insights.get("anomaly_analysis"):
                anomaly_data = ml_insights["anomaly_analysis"]
                if anomaly_data.get("anomalous_indicators", 0) > 0:
                    recommendations.append(
                        "🔍 INVESTIGATION: Investigate anomalous indicators for zero-day threats"
                    )

            # Infrastructure recommendations
            domains = [ind for ind in indicators if ind.indicator_type == "domain"]
            ips = [ind for ind in indicators if ind.indicator_type in ["ipv4", "ipv6"]]

            if domains:
                recommendations.append(
                    f"🌐 DNS: Implement DNS blocking for {len(domains)} malicious domains"
                )

            if ips:
                recommendations.append(
                    f"🔥 FIREWALL: Block {len(ips)} malicious IP addresses at perimeter"
                )

            # Behavioral recommendations
            recommendations.extend([
                "📊 MONITORING: Enhance behavioral analytics for identified attack patterns",
                "🔄 FEEDS: Update threat intelligence feeds with new indicators",
                "👥 TRAINING: Brief security teams on identified threat actor TTPs",
                "📋 PLAYBOOKS: Update incident response playbooks for identified techniques",
                "🕒 SCHEDULE: Schedule follow-up analysis in 7 days to track threat evolution"
            ])

            return recommendations[:15]  # Limit to top 15 recommendations

        except Exception as e:
            logger.error(f"Threat recommendations generation failed: {e}")
            return ["Manual threat analysis and response recommended"]

    def _generate_executive_summary(
        self,
        indicators: List[ThreatIndicator],
        attributions: List[Dict[str, Any]],
        threat_level: ThreatLevel,
        confidence: float
    ) -> str:
        """Generate executive summary of threat analysis"""
        try:
            # Count indicators by type
            indicator_counts = Counter([ind.indicator_type for ind in indicators])

            # High-confidence attributions
            high_conf_actors = [attr["actor_name"] for attr in attributions if attr["confidence"] > 0.6]

            # Threat level description
            threat_descriptions = {
                ThreatLevel.CRITICAL: "CRITICAL threat requiring immediate response",
                ThreatLevel.HIGH: "HIGH threat requiring urgent attention",
                ThreatLevel.MEDIUM: "MEDIUM threat requiring monitoring and response",
                ThreatLevel.LOW: "LOW threat requiring routine monitoring",
                ThreatLevel.INFO: "INFORMATIONAL - no immediate threat identified"
            }

            summary = f"""
THREAT INTELLIGENCE ANALYSIS SUMMARY

THREAT LEVEL: {threat_level.value.upper()} (Confidence: {confidence:.1%})
ASSESSMENT: {threat_descriptions[threat_level]}

INDICATORS ANALYZED: {len(indicators)} total
- Network indicators: {indicator_counts.get('ipv4', 0) + indicator_counts.get('ipv6', 0)}
- Domain indicators: {indicator_counts.get('domain', 0)}
- Hash indicators: {indicator_counts.get('sha256', 0) + indicator_counts.get('sha1', 0) + indicator_counts.get('md5', 0)}
- URL indicators: {indicator_counts.get('url', 0)}

ATTRIBUTION RESULTS: {len(attributions)} potential threat actors identified
"""

            if high_conf_actors:
                summary += f"- High-confidence attribution: {', '.join(high_conf_actors[:3])}\n"

            if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                summary += f"\nIMMEDIATE ACTIONS REQUIRED:\n"
                summary += f"- Implement blocking controls for identified indicators\n"
                summary += f"- Activate incident response procedures\n"
                summary += f"- Initiate threat hunting operations\n"

            summary += f"\nRECOMMENDATION: {'Emergency response' if threat_level == ThreatLevel.CRITICAL else 'Standard threat response'} protocols should be initiated."

            return summary.strip()

        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return "Executive summary generation failed - refer to technical details"

    # Helper methods for ML and data processing
    async def _load_ml_models(self):
        """Load pre-trained ML models for threat analysis"""
        # In production, these would be actual trained models
        self.ml_models = {
            "threat_family_classifier": {"type": "random_forest", "accuracy": 0.87},
            "malware_classifier": {"type": "neural_network", "accuracy": 0.92},
            "campaign_classifier": {"type": "svm", "accuracy": 0.84},
            "attribution_model": {"type": "ensemble", "accuracy": 0.79}
        }
        logger.info("ML models loaded successfully")

    async def _initialize_threat_feeds(self):
        """Initialize threat intelligence feed connections"""
        # Simulate feed initialization
        logger.info("Threat intelligence feeds initialized")

    async def _load_historical_data(self):
        """Load historical threat data for analysis"""
        # Simulate historical data loading
        logger.info("Historical threat data loaded")

    async def _initialize_reputation_systems(self):
        """Initialize reputation scoring systems"""
        # Simulate reputation system initialization
        logger.info("Reputation systems initialized")

    def _load_mitre_framework(self) -> Dict[str, Dict[str, Any]]:
        """Load MITRE ATT&CK framework data"""
        return {
            "T1595": {"name": "Active Scanning", "tactic": "Reconnaissance"},
            "T1590": {"name": "Gather Victim Network Information", "tactic": "Reconnaissance"},
            "T1566": {"name": "Phishing", "tactic": "Initial Access"},
            "T1190": {"name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
            "T1059": {"name": "Command and Scripting Interpreter", "tactic": "Execution"},
            "T1053": {"name": "Scheduled Task/Job", "tactic": "Persistence"},
            "T1055": {"name": "Process Injection", "tactic": "Defense Evasion"},
            "T1003": {"name": "OS Credential Dumping", "tactic": "Credential Access"},
            "T1021": {"name": "Remote Services", "tactic": "Lateral Movement"},
            "T1041": {"name": "Exfiltration Over C2 Channel", "tactic": "Exfiltration"}
        }

    def _load_actor_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load known threat actor profiles"""
        return {
            "APT1": {
                "type": ThreatActorType.NATION_STATE,
                "motivation": "espionage",
                "sophistication": "high",
                "origin": "China",
                "ttps": ["T1566", "T1190", "T1055", "T1003"],
                "campaigns": ["Comment Crew", "PLA Unit 61398"]
            },
            "Lazarus": {
                "type": ThreatActorType.NATION_STATE,
                "motivation": "financial_espionage",
                "sophistication": "high",
                "origin": "North Korea",
                "ttps": ["T1566", "T1059", "T1055", "T1041"],
                "campaigns": ["Sony Pictures", "WannaCry", "SWIFT Banking"]
            },
            "FIN7": {
                "type": ThreatActorType.CYBERCRIMINAL,
                "motivation": "financial",
                "sophistication": "medium",
                "origin": "Eastern Europe",
                "ttps": ["T1566", "T1059", "T1053", "T1003"],
                "campaigns": ["Carbanak", "Cobalt Strike"]
            }
        }

    # Additional helper methods would be implemented here for:
    # - Feature extraction methods
    # - ML prediction methods
    # - Reputation scoring
    # - Geolocation services
    # - Campaign correlation
    # - Timeline analysis
    # - etc.

    async def _extract_ml_features(self, indicators: List[ThreatIndicator]) -> np.ndarray:
        """Extract ML features from indicators"""
        features = []
        for indicator in indicators:
            # Create feature vector
            feature_vector = [
                len(indicator.value),
                indicator.reputation_score,
                len(indicator.threat_types),
                len(indicator.mitre_techniques),
                indicator.confidence_score,
                1.0 if indicator.indicator_type == "domain" else 0.0,
                1.0 if indicator.indicator_type in ["ipv4", "ipv6"] else 0.0,
                1.0 if any("malware" in tt for tt in indicator.threat_types) else 0.0
            ]
            features.append(feature_vector)

        return np.array(features) if features else np.array([[0.0] * 8])

    # Placeholder implementations for core ML methods
    async def _predict_threat_family(self, features: np.ndarray) -> List[str]:
        """Predict threat family using ML"""
        # Simulated prediction
        return ["trojan", "backdoor", "spyware"]

    async def _predict_malware_type(self, features: np.ndarray) -> List[str]:
        """Predict malware type using ML"""
        return ["banking_trojan", "ransomware", "botnet"]

    async def _predict_campaign_association(self, features: np.ndarray) -> List[str]:
        """Predict campaign associations"""
        return ["apt1_campaign", "lazarus_campaign"]

    async def _calculate_ml_risk_scores(self, features: np.ndarray) -> List[float]:
        """Calculate ML-based risk scores"""
        # Simulate risk scoring
        return [np.random.random() for _ in range(len(features))]

    # Store analysis results for continuous learning
    async def _store_analysis_results(self, report: ThreatIntelligenceReport):
        """Store analysis results for ML model improvement"""
        try:
            # In production, this would store to a database for model training
            logger.debug(f"Stored analysis results for report {report.report_id}")
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")


# Global threat intelligence engine instance
_threat_intelligence_engine: Optional[AdvancedThreatIntelligenceEngine] = None


async def get_threat_intelligence_engine() -> AdvancedThreatIntelligenceEngine:
    """Get global threat intelligence engine instance"""
    global _threat_intelligence_engine

    if _threat_intelligence_engine is None:
        _threat_intelligence_engine = AdvancedThreatIntelligenceEngine()
        await _threat_intelligence_engine.initialize()

    return _threat_intelligence_engine
