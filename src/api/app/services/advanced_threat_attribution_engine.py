"""
Advanced Threat Attribution Engine - Production Implementation
AI-powered threat actor attribution and campaign analysis for cybersecurity operations
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import statistics
from collections import defaultdict, Counter
import re

# ML and data analysis imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA, LatentDirichletAllocation
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neural_network import MLPClassifier
    ML_AVAILABLE = True
except ImportError:
    np = None
    pd = None
    ML_AVAILABLE = False
    logging.warning("ML libraries not available, using rule-based attribution")

# Graph analysis for campaign modeling
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available, using simplified graph modeling")

# Natural language processing
try:
    from textblob import TextBlob
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logging.warning("NLP libraries not available, using basic text analysis")

from .base_service import XORBService, ServiceHealth, ServiceStatus
from .interfaces import ThreatIntelligenceService
from .advanced_mitre_attack_engine import get_advanced_mitre_engine

logger = logging.getLogger(__name__)


class AttributionConfidence(Enum):
    """Attribution confidence levels"""
    VERY_LOW = "very_low"      # < 30%
    LOW = "low"                # 30-50%
    MEDIUM = "medium"          # 50-70%
    HIGH = "high"              # 70-90%
    VERY_HIGH = "very_high"    # > 90%


class ThreatActorType(Enum):
    """Threat actor classification"""
    NATION_STATE = "nation_state"
    CYBERCRIMINAL = "cybercriminal"
    HACKTIVIST = "hacktivist"
    INSIDER = "insider"
    SCRIPT_KIDDIE = "script_kiddie"
    TERRORIST = "terrorist"
    UNKNOWN = "unknown"


class AttackMotivation(Enum):
    """Primary attack motivations"""
    ESPIONAGE = "espionage"
    FINANCIAL = "financial"
    POLITICAL = "political"
    IDEOLOGICAL = "ideological"
    DISRUPTION = "disruption"
    REVENGE = "revenge"
    TESTING = "testing"
    UNKNOWN = "unknown"


@dataclass
class ThreatActorProfile:
    """Comprehensive threat actor profile"""
    actor_id: str
    name: str
    aliases: List[str]
    actor_type: ThreatActorType
    motivation: AttackMotivation
    sophistication_level: int  # 1-5
    origin_country: Optional[str] = None
    target_sectors: List[str] = field(default_factory=list)
    target_countries: List[str] = field(default_factory=list)
    
    # Technical characteristics
    preferred_tools: List[str] = field(default_factory=list)
    attack_patterns: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    infrastructure_patterns: List[str] = field(default_factory=list)
    
    # Behavioral characteristics
    operational_schedule: Dict[str, Any] = field(default_factory=dict)
    language_patterns: List[str] = field(default_factory=list)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    
    # Campaign history
    known_campaigns: List[str] = field(default_factory=list)
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None
    
    # Confidence and metadata
    confidence_score: float = 0.0
    evidence_quality: str = "medium"
    last_updated: datetime = field(default_factory=datetime.utcnow)
    data_sources: List[str] = field(default_factory=list)


@dataclass
class AttributionEvidence:
    """Evidence supporting threat attribution"""
    evidence_id: str
    evidence_type: str  # technical, behavioral, linguistic, temporal
    evidence_data: Dict[str, Any]
    confidence_weight: float
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    mitre_techniques: List[str] = field(default_factory=list)
    supporting_indicators: List[str] = field(default_factory=list)


@dataclass
class AttributionResult:
    """Threat attribution analysis result"""
    attribution_id: str
    analyzed_indicators: List[str]
    top_candidates: List[Dict[str, Any]]
    confidence_score: float
    confidence_level: AttributionConfidence
    evidence: List[AttributionEvidence]
    campaign_analysis: Dict[str, Any]
    timeline_analysis: Dict[str, Any]
    recommendations: List[str]
    analysis_metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)


class AdvancedThreatAttributionEngine(XORBService, ThreatIntelligenceService):
    """
    Advanced Threat Attribution Engine
    
    Provides AI-powered threat actor attribution through:
    - Machine learning classification models
    - Behavioral pattern analysis
    - Campaign correlation and clustering
    - Multi-factor attribution scoring
    - Graph-based relationship modeling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            service_name="threat_attribution_engine",
            service_type="threat_intelligence",
            dependencies=["mitre_attack_engine", "threat_intelligence"],
            config=config or {}
        )
        
        # Threat actor database
        self.threat_actors: Dict[str, ThreatActorProfile] = {}
        self.actor_campaigns: Dict[str, List[str]] = defaultdict(list)
        self.campaign_graph: Optional[Any] = None
        
        # ML models
        self.attribution_classifier: Optional[Any] = None
        self.behavior_clusterer: Optional[Any] = None
        self.feature_extractor: Optional[Any] = None
        self.text_analyzer: Optional[Any] = None
        
        # Feature engineering
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        
        # Attribution history
        self.attribution_history: List[AttributionResult] = []
        self.evidence_database: Dict[str, AttributionEvidence] = {}
        
        # Configuration
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.3)
        self.max_attribution_candidates = config.get("max_candidates", 10)
        self.evidence_retention_days = config.get("evidence_retention_days", 365)
        
        # Performance metrics
        self.attribution_metrics = {
            "total_attributions": 0,
            "high_confidence_attributions": 0,
            "actors_tracked": 0,
            "campaigns_analyzed": 0,
            "evidence_pieces": 0,
            "average_confidence": 0.0,
            "false_positive_rate": 0.0
        }

    async def initialize(self) -> bool:
        """Initialize the threat attribution engine"""
        try:
            logger.info("Initializing Advanced Threat Attribution Engine...")
            
            # Initialize ML components
            if ML_AVAILABLE:
                await self._initialize_ml_models()
            
            # Load threat actor profiles
            await self._load_threat_actor_database()
            
            # Initialize campaign graph
            if NETWORKX_AVAILABLE:
                await self._initialize_campaign_graph()
            
            # Load MITRE ATT&CK integration
            await self._initialize_mitre_integration()
            
            # Initialize NLP components
            if NLP_AVAILABLE:
                await self._initialize_nlp_components()
            
            # Load historical attribution data
            await self._load_attribution_history()
            
            logger.info(f"Attribution engine initialized with {len(self.threat_actors)} threat actors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize attribution engine: {e}")
            return False

    async def _initialize_ml_models(self):
        """Initialize machine learning models for attribution"""
        try:
            if ML_AVAILABLE:
                # Multi-class attribution classifier
                self.attribution_classifier = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                )
                
                # Behavioral clustering for campaign analysis
                self.behavior_clusterer = DBSCAN(
                    eps=0.3,
                    min_samples=3,
                    metric='cosine'
                )
                
                # Feature scaling
                self.scalers['attribution'] = StandardScaler()
                self.scalers['behavioral'] = StandardScaler()
                
                # Text vectorization for linguistic analysis
                self.vectorizers['tfidf'] = TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 3),
                    stop_words='english'
                )
                
                # Dimensionality reduction
                self.dimension_reducer = PCA(n_components=50)
                
                logger.info("ML models for attribution initialized")
                
        except Exception as e:
            logger.warning(f"ML model initialization failed: {e}")

    async def perform_attribution_analysis(
        self,
        indicators: List[str],
        context: Dict[str, Any],
        attack_data: Optional[Dict[str, Any]] = None
    ) -> AttributionResult:
        """Perform comprehensive threat attribution analysis"""
        try:
            attribution_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            logger.info(f"Starting attribution analysis {attribution_id} for {len(indicators)} indicators")
            
            # Extract features from indicators and context
            features = await self._extract_attribution_features(indicators, context, attack_data)
            
            # Collect evidence from multiple sources
            evidence = await self._collect_attribution_evidence(indicators, context, attack_data)
            
            # Perform ML-based classification if available
            ml_candidates = []
            if ML_AVAILABLE and self.attribution_classifier:
                ml_candidates = await self._ml_attribution_analysis(features, evidence)
            
            # Rule-based attribution analysis
            rule_candidates = await self._rule_based_attribution(indicators, evidence, context)
            
            # Behavioral pattern matching
            behavioral_candidates = await self._behavioral_pattern_analysis(indicators, evidence)
            
            # Campaign correlation analysis
            campaign_analysis = await self._campaign_correlation_analysis(indicators, evidence)
            
            # Timeline analysis
            timeline_analysis = await self._timeline_analysis(indicators, context)
            
            # Combine all attribution candidates
            all_candidates = self._combine_attribution_candidates(
                ml_candidates, rule_candidates, behavioral_candidates
            )
            
            # Score and rank candidates
            ranked_candidates = await self._score_attribution_candidates(all_candidates, evidence)
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(ranked_candidates, evidence)
            confidence_level = self._determine_confidence_level(confidence_score)
            
            # Generate recommendations
            recommendations = await self._generate_attribution_recommendations(
                ranked_candidates, evidence, confidence_score
            )
            
            # Create attribution result
            result = AttributionResult(
                attribution_id=attribution_id,
                analyzed_indicators=indicators,
                top_candidates=ranked_candidates[:self.max_attribution_candidates],
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                evidence=evidence,
                campaign_analysis=campaign_analysis,
                timeline_analysis=timeline_analysis,
                recommendations=recommendations,
                analysis_metadata={
                    "analysis_duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "features_extracted": len(features),
                    "evidence_pieces": len(evidence),
                    "ml_models_used": ML_AVAILABLE,
                    "graph_analysis_used": NETWORKX_AVAILABLE,
                    "nlp_analysis_used": NLP_AVAILABLE
                }
            )
            
            # Store attribution result
            self.attribution_history.append(result)
            await self._store_attribution_evidence(evidence)
            
            # Update metrics
            self._update_attribution_metrics(result)
            
            logger.info(f"Attribution analysis completed: {confidence_level.value} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Attribution analysis failed: {e}")
            raise

    async def _extract_attribution_features(
        self,
        indicators: List[str],
        context: Dict[str, Any],
        attack_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract features for attribution analysis"""
        features = {
            "technical_features": {},
            "behavioral_features": {},
            "temporal_features": {},
            "linguistic_features": {},
            "infrastructure_features": {}
        }
        
        try:
            # Technical features from indicators
            features["technical_features"] = await self._extract_technical_features(indicators)
            
            # Behavioral features from attack patterns
            if attack_data:
                features["behavioral_features"] = await self._extract_behavioral_features(attack_data)
            
            # Temporal features
            features["temporal_features"] = await self._extract_temporal_features(context)
            
            # Linguistic features from text data
            text_data = context.get("text_data", [])
            if text_data and NLP_AVAILABLE:
                features["linguistic_features"] = await self._extract_linguistic_features(text_data)
            
            # Infrastructure features
            features["infrastructure_features"] = await self._extract_infrastructure_features(indicators)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return features

    async def _extract_technical_features(self, indicators: List[str]) -> Dict[str, Any]:
        """Extract technical features from indicators"""
        features = {
            "ip_geolocation": {},
            "domain_characteristics": {},
            "malware_signatures": {},
            "infrastructure_patterns": {}
        }
        
        for indicator in indicators:
            # IP address analysis
            if self._is_ip_address(indicator):
                geo_info = await self._get_ip_geolocation(indicator)
                features["ip_geolocation"][indicator] = geo_info
            
            # Domain analysis
            elif self._is_domain(indicator):
                domain_info = await self._analyze_domain_characteristics(indicator)
                features["domain_characteristics"][indicator] = domain_info
            
            # File hash analysis
            elif self._is_file_hash(indicator):
                malware_info = await self._analyze_malware_signature(indicator)
                features["malware_signatures"][indicator] = malware_info
        
        return features

    async def _behavioral_pattern_analysis(
        self,
        indicators: List[str],
        evidence: List[AttributionEvidence]
    ) -> List[Dict[str, Any]]:
        """Analyze behavioral patterns for attribution"""
        candidates = []
        
        try:
            # Extract behavioral features
            behavioral_features = []
            for ev in evidence:
                if ev.evidence_type == "behavioral":
                    behavioral_features.append(ev.evidence_data)
            
            if not behavioral_features:
                return candidates
            
            # Compare with known actor behaviors
            for actor_id, actor in self.threat_actors.items():
                similarity_score = self._calculate_behavioral_similarity(
                    behavioral_features, actor
                )
                
                if similarity_score > 0.3:  # Threshold for consideration
                    candidates.append({
                        "actor_id": actor_id,
                        "actor_name": actor.name,
                        "similarity_score": similarity_score,
                        "matching_behaviors": self._identify_matching_behaviors(
                            behavioral_features, actor
                        ),
                        "attribution_type": "behavioral"
                    })
            
            # Sort by similarity score
            candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Behavioral pattern analysis failed: {e}")
            return candidates

    async def _campaign_correlation_analysis(
        self,
        indicators: List[str],
        evidence: List[AttributionEvidence]
    ) -> Dict[str, Any]:
        """Analyze campaign correlations and clustering"""
        analysis = {
            "related_campaigns": [],
            "campaign_clusters": [],
            "timeline_patterns": {},
            "infrastructure_overlap": {}
        }
        
        try:
            if not NETWORKX_AVAILABLE:
                return analysis
            
            # Build indicator correlation graph
            correlation_graph = nx.Graph()
            
            # Add indicators as nodes
            for indicator in indicators:
                correlation_graph.add_node(indicator, type="indicator")
            
            # Add edges based on co-occurrence in campaigns
            for campaign_id, campaign_indicators in self.actor_campaigns.items():
                common_indicators = set(indicators) & set(campaign_indicators)
                if len(common_indicators) > 1:
                    # Add campaign node
                    correlation_graph.add_node(campaign_id, type="campaign")
                    
                    # Connect indicators to campaign
                    for indicator in common_indicators:
                        correlation_graph.add_edge(indicator, campaign_id)
            
            # Find connected components (related campaigns)
            components = list(nx.connected_components(correlation_graph))
            
            for component in components:
                campaigns_in_component = [node for node in component 
                                        if correlation_graph.nodes[node].get("type") == "campaign"]
                if campaigns_in_component:
                    analysis["related_campaigns"].extend(campaigns_in_component)
            
            # Cluster analysis
            if len(correlation_graph.nodes) > 3:
                try:
                    clusters = list(nx.algorithms.community.greedy_modularity_communities(correlation_graph))
                    analysis["campaign_clusters"] = [list(cluster) for cluster in clusters]
                except:
                    pass  # Clustering failed, continue without it
            
            return analysis
            
        except Exception as e:
            logger.error(f"Campaign correlation analysis failed: {e}")
            return analysis

    def _calculate_behavioral_similarity(
        self,
        observed_behaviors: List[Dict[str, Any]],
        actor: ThreatActorProfile
    ) -> float:
        """Calculate behavioral similarity between observations and known actor"""
        if not observed_behaviors:
            return 0.0
        
        similarity_scores = []
        
        # Check tool preferences
        observed_tools = set()
        for behavior in observed_behaviors:
            observed_tools.update(behavior.get("tools_used", []))
        
        if observed_tools and actor.preferred_tools:
            tool_overlap = len(observed_tools & set(actor.preferred_tools))
            tool_similarity = tool_overlap / max(len(observed_tools), len(actor.preferred_tools))
            similarity_scores.append(tool_similarity * 0.3)  # Weight: 30%
        
        # Check MITRE techniques
        observed_techniques = set()
        for behavior in observed_behaviors:
            observed_techniques.update(behavior.get("mitre_techniques", []))
        
        if observed_techniques and actor.mitre_techniques:
            technique_overlap = len(observed_techniques & set(actor.mitre_techniques))
            technique_similarity = technique_overlap / max(len(observed_techniques), len(actor.mitre_techniques))
            similarity_scores.append(technique_similarity * 0.4)  # Weight: 40%
        
        # Check attack patterns
        observed_patterns = set()
        for behavior in observed_behaviors:
            observed_patterns.update(behavior.get("attack_patterns", []))
        
        if observed_patterns and actor.attack_patterns:
            pattern_overlap = len(observed_patterns & set(actor.attack_patterns))
            pattern_similarity = pattern_overlap / max(len(observed_patterns), len(actor.attack_patterns))
            similarity_scores.append(pattern_similarity * 0.3)  # Weight: 30%
        
        return sum(similarity_scores) if similarity_scores else 0.0

    def _combine_attribution_candidates(
        self,
        ml_candidates: List[Dict[str, Any]],
        rule_candidates: List[Dict[str, Any]],
        behavioral_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine attribution candidates from different analysis methods"""
        combined = {}
        
        # Process ML candidates
        for candidate in ml_candidates:
            actor_id = candidate["actor_id"]
            if actor_id not in combined:
                combined[actor_id] = candidate.copy()
                combined[actor_id]["evidence_sources"] = ["ml"]
            else:
                # Combine scores (weighted average)
                existing_score = combined[actor_id].get("confidence_score", 0)
                new_score = candidate.get("confidence_score", 0)
                combined[actor_id]["confidence_score"] = (existing_score + new_score) / 2
                combined[actor_id]["evidence_sources"].append("ml")
        
        # Process rule-based candidates
        for candidate in rule_candidates:
            actor_id = candidate["actor_id"]
            if actor_id not in combined:
                combined[actor_id] = candidate.copy()
                combined[actor_id]["evidence_sources"] = ["rules"]
            else:
                # Boost confidence for multi-source attribution
                boost_factor = 1.2
                combined[actor_id]["confidence_score"] *= boost_factor
                combined[actor_id]["evidence_sources"].append("rules")
        
        # Process behavioral candidates
        for candidate in behavioral_candidates:
            actor_id = candidate["actor_id"]
            if actor_id not in combined:
                combined[actor_id] = candidate.copy()
                combined[actor_id]["evidence_sources"] = ["behavioral"]
            else:
                # Boost confidence for multi-source attribution
                boost_factor = 1.15
                combined[actor_id]["confidence_score"] *= boost_factor
                combined[actor_id]["evidence_sources"].append("behavioral")
        
        return list(combined.values())

    def _determine_confidence_level(self, confidence_score: float) -> AttributionConfidence:
        """Determine confidence level from numerical score"""
        if confidence_score >= 0.9:
            return AttributionConfidence.VERY_HIGH
        elif confidence_score >= 0.7:
            return AttributionConfidence.HIGH
        elif confidence_score >= 0.5:
            return AttributionConfidence.MEDIUM
        elif confidence_score >= 0.3:
            return AttributionConfidence.LOW
        else:
            return AttributionConfidence.VERY_LOW

    def _update_attribution_metrics(self, result: AttributionResult):
        """Update attribution performance metrics"""
        self.attribution_metrics["total_attributions"] += 1
        
        if result.confidence_level in [AttributionConfidence.HIGH, AttributionConfidence.VERY_HIGH]:
            self.attribution_metrics["high_confidence_attributions"] += 1
        
        # Update average confidence
        total = self.attribution_metrics["total_attributions"]
        current_avg = self.attribution_metrics["average_confidence"]
        new_avg = ((current_avg * (total - 1)) + result.confidence_score) / total
        self.attribution_metrics["average_confidence"] = new_avg
        
        self.attribution_metrics["evidence_pieces"] += len(result.evidence)

    async def health_check(self) -> ServiceHealth:
        """Health check for threat attribution engine"""
        try:
            checks = {
                "threat_actors_loaded": len(self.threat_actors) > 0,
                "ml_models_available": ML_AVAILABLE,
                "graph_analysis_available": NETWORKX_AVAILABLE,
                "nlp_available": NLP_AVAILABLE,
                "attribution_history": len(self.attribution_history) > 0
            }
            
            healthy = checks["threat_actors_loaded"]
            
            return ServiceHealth(
                service_name=self.service_name,
                status=ServiceStatus.HEALTHY if healthy else ServiceStatus.DEGRADED,
                checks=checks,
                metrics=self.attribution_metrics,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name=self.service_name,
                status=ServiceStatus.UNHEALTHY,
                error=str(e),
                timestamp=datetime.utcnow()
            )


# Service factory
_attribution_engine: Optional[AdvancedThreatAttributionEngine] = None

async def get_threat_attribution_engine(config: Dict[str, Any] = None) -> AdvancedThreatAttributionEngine:
    """Get or create threat attribution engine instance"""
    global _attribution_engine
    
    if _attribution_engine is None:
        _attribution_engine = AdvancedThreatAttributionEngine(config)
        await _attribution_engine.initialize()
    
    return _attribution_engine