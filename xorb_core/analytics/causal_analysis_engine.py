#!/usr/bin/env python3
"""
Advanced Causal Analysis Engine for XORB
Implements threat clustering, adaptive scoring, and causal inference
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Union, Tuple
import logging
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import networkx as nx

logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Types of analysis supported"""
    THREAT_CLUSTERING = "threat_clustering"
    CAUSAL_INFERENCE = "causal_inference"
    ANOMALY_DETECTION = "anomaly_detection"
    CORRELATION_ANALYSIS = "correlation_analysis"
    IMPACT_ASSESSMENT = "impact_assessment"
    PREDICTIVE_MODELING = "predictive_modeling"


class ThreatCategory(str, Enum):
    """Threat categories for clustering"""
    MALWARE = "malware"
    PHISHING = "phishing"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"
    INSIDER_THREAT = "insider_threat"
    ADVANCED_PERSISTENT_THREAT = "advanced_persistent_threat"
    DENIAL_OF_SERVICE = "denial_of_service"
    DATA_EXFILTRATION = "data_exfiltration"
    SOCIAL_ENGINEERING = "social_engineering"


class CausalRelationType(str, Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    CORRELATION = "correlation"
    CONFOUNDING = "confounding"
    MEDIATING = "mediating"
    MODERATING = "moderating"


@dataclass
class ThreatEvent:
    """Individual threat event for analysis"""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: float
    source_ip: Optional[str] = None
    target_ip: Optional[str] = None
    protocol: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: Optional[ThreatCategory] = None
    risk_score: float = 0.0
    

@dataclass
class ThreatCluster:
    """Cluster of related threat events"""
    cluster_id: str
    events: List[ThreatEvent]
    centroid: Dict[str, float]
    cluster_score: float
    threat_category: ThreatCategory
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    size: int = 0
    cohesion: float = 0.0
    

@dataclass
class CausalRelation:
    """Causal relationship between events or features"""
    relation_id: str
    cause_event: str
    effect_event: str
    relation_type: CausalRelationType
    strength: float
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    

@dataclass
class AdaptiveScore:
    """Adaptive scoring model result"""
    score_id: str
    entity_id: str
    score_type: str
    base_score: float
    adjusted_score: float
    adjustment_factors: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ThreatClusteringEngine:
    """Advanced threat clustering with multiple algorithms"""
    
    def __init__(self, min_cluster_size: int = 5, max_clusters: int = 50):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.feature_extractor = ThreatFeatureExtractor()
        self.clusters: Dict[str, ThreatCluster] = {}
        self.cluster_evolution_history: List[Dict[str, Any]] = []
        
    async def cluster_threats(self, events: List[ThreatEvent],
                            algorithm: str = "dbscan") -> List[ThreatCluster]:
        """Cluster threat events using specified algorithm"""
        
        if len(events) < self.min_cluster_size:
            logger.warning(f"Not enough events for clustering: {len(events)}")
            return []
            
        # Extract features for clustering
        features_matrix, feature_names = await self._extract_clustering_features(events)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_matrix)
        
        # Apply clustering algorithm
        if algorithm == "dbscan":
            clusters = await self._dbscan_clustering(events, features_normalized)
        elif algorithm == "kmeans":
            clusters = await self._kmeans_clustering(events, features_normalized)
        elif algorithm == "hierarchical":
            clusters = await self._hierarchical_clustering(events, features_normalized)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
            
        # Post-process clusters
        refined_clusters = await self._refine_clusters(clusters)
        
        # Update cluster store
        for cluster in refined_clusters:
            self.clusters[cluster.cluster_id] = cluster
            
        logger.info(f"Created {len(refined_clusters)} threat clusters using {algorithm}")
        return refined_clusters
        
    async def _extract_clustering_features(self, events: List[ThreatEvent]) -> Tuple[np.ndarray, List[str]]:
        """Extract numerical features for clustering"""
        
        feature_vectors = []
        feature_names = [
            "severity", "hour_of_day", "day_of_week", "source_entropy",
            "target_entropy", "protocol_numeric", "feature_count",
            "time_since_last_event", "event_frequency"
        ]
        
        # Calculate time-based features
        event_times = [e.timestamp for e in events]
        event_times.sort()
        
        for i, event in enumerate(events):
            features = []
            
            # Basic features
            features.append(event.severity)
            features.append(event.timestamp.hour)
            features.append(event.timestamp.weekday())
            
            # IP address entropy (simplified)
            source_entropy = self._calculate_ip_entropy(event.source_ip) if event.source_ip else 0.0
            target_entropy = self._calculate_ip_entropy(event.target_ip) if event.target_ip else 0.0
            features.extend([source_entropy, target_entropy])
            
            # Protocol encoding
            protocol_map = {"tcp": 1, "udp": 2, "icmp": 3, "http": 4, "https": 5}
            protocol_numeric = protocol_map.get(event.protocol, 0) if event.protocol else 0
            features.append(protocol_numeric)
            
            # Event features
            features.append(len(event.features))
            
            # Temporal features
            if i > 0:
                time_diff = (event.timestamp - events[i-1].timestamp).total_seconds()
                features.append(min(3600, time_diff))  # Cap at 1 hour
            else:
                features.append(0)
                
            # Event frequency (events in last hour)
            hour_ago = event.timestamp - timedelta(hours=1)
            freq = sum(1 for e in events if hour_ago <= e.timestamp <= event.timestamp)
            features.append(freq)
            
            feature_vectors.append(features)
            
        return np.array(feature_vectors), feature_names
        
    def _calculate_ip_entropy(self, ip: str) -> float:
        """Calculate entropy of IP address octets"""
        try:
            octets = [int(x) for x in ip.split('.')]
            # Simple entropy calculation
            return np.std(octets) / 255.0
        except:
            return 0.0
            
    async def _dbscan_clustering(self, events: List[ThreatEvent],
                               features: np.ndarray) -> List[ThreatCluster]:
        """DBSCAN clustering for density-based threat grouping"""
        
        # Optimize DBSCAN parameters
        eps = await self._optimize_dbscan_eps(features)
        min_samples = max(3, self.min_cluster_size // 2)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features)
        
        # Group events by cluster
        clusters_dict = defaultdict(list)
        for event, label in zip(events, cluster_labels):
            if label != -1:  # Ignore noise points
                clusters_dict[label].append(event)
                
        # Create cluster objects
        clusters = []
        for cluster_id, cluster_events in clusters_dict.items():
            if len(cluster_events) >= self.min_cluster_size:
                cluster = await self._create_threat_cluster(
                    f"dbscan_{cluster_id}_{int(time.time())}", 
                    cluster_events, 
                    features
                )
                clusters.append(cluster)
                
        return clusters
        
    async def _kmeans_clustering(self, events: List[ThreatEvent],
                               features: np.ndarray) -> List[ThreatCluster]:
        """K-means clustering for threat grouping"""
        
        # Determine optimal number of clusters
        n_clusters = min(self.max_clusters, max(2, len(events) // self.min_cluster_size))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Group events by cluster
        clusters_dict = defaultdict(list)
        for event, label in zip(events, cluster_labels):
            clusters_dict[label].append(event)
            
        # Create cluster objects
        clusters = []
        for cluster_id, cluster_events in clusters_dict.items():
            if len(cluster_events) >= self.min_cluster_size:
                cluster = await self._create_threat_cluster(
                    f"kmeans_{cluster_id}_{int(time.time())}", 
                    cluster_events, 
                    features,
                    centroid=kmeans.cluster_centers_[cluster_id]
                )
                clusters.append(cluster)
                
        return clusters
        
    async def _hierarchical_clustering(self, events: List[ThreatEvent],
                                     features: np.ndarray) -> List[ThreatCluster]:
        """Hierarchical clustering for threat grouping"""
        
        from sklearn.cluster import AgglomerativeClustering
        
        # Determine number of clusters
        n_clusters = min(self.max_clusters, max(2, len(events) // self.min_cluster_size))
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = hierarchical.fit_predict(features)
        
        # Group events by cluster
        clusters_dict = defaultdict(list)
        for event, label in zip(events, cluster_labels):
            clusters_dict[label].append(event)
            
        # Create cluster objects
        clusters = []
        for cluster_id, cluster_events in clusters_dict.items():
            if len(cluster_events) >= self.min_cluster_size:
                cluster = await self._create_threat_cluster(
                    f"hierarchical_{cluster_id}_{int(time.time())}", 
                    cluster_events, 
                    features
                )
                clusters.append(cluster)
                
        return clusters
        
    async def _optimize_dbscan_eps(self, features: np.ndarray) -> float:
        """Optimize epsilon parameter for DBSCAN"""
        
        from sklearn.neighbors import NearestNeighbors
        
        # Use k-distance graph to find optimal eps
        k = min(10, features.shape[0] // 2)
        nbrs = NearestNeighbors(n_neighbors=k).fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        # Sort distances to k-th nearest neighbor
        k_distances = np.sort(distances[:, k-1])
        
        # Find elbow point (simplified)
        diff = np.diff(k_distances)
        elbow_idx = np.argmax(diff) if len(diff) > 0 else len(k_distances) // 2
        
        return k_distances[elbow_idx]
        
    async def _create_threat_cluster(self, cluster_id: str, events: List[ThreatEvent],
                                   features: np.ndarray, centroid: np.ndarray = None) -> ThreatCluster:
        """Create threat cluster object with analysis"""
        
        if centroid is None:
            # Calculate centroid
            event_indices = [i for i, e in enumerate(events)]
            if event_indices:
                centroid = np.mean(features[event_indices], axis=0)
            else:
                centroid = np.zeros(features.shape[1])
                
        # Determine threat category
        threat_category = await self._classify_cluster_threat_type(events)
        
        # Calculate cluster score and cohesion
        cluster_score = await self._calculate_cluster_score(events)
        cohesion = await self._calculate_cluster_cohesion(events, features)
        
        cluster = ThreatCluster(
            cluster_id=cluster_id,
            events=events,
            centroid={f"feature_{i}": float(v) for i, v in enumerate(centroid)},
            cluster_score=cluster_score,
            threat_category=threat_category,
            size=len(events),
            cohesion=cohesion
        )
        
        return cluster
        
    async def _classify_cluster_threat_type(self, events: List[ThreatEvent]) -> ThreatCategory:
        """Classify cluster threat type based on events"""
        
        # Count event types and features
        event_types = defaultdict(int)
        feature_patterns = defaultdict(int)
        
        for event in events:
            event_types[event.event_type] += 1
            for feature_key in event.features.keys():
                feature_patterns[feature_key] += 1
                
        # Simple classification logic (could be enhanced with ML)
        if "malware" in " ".join(event_types.keys()).lower():
            return ThreatCategory.MALWARE
        elif "phishing" in " ".join(event_types.keys()).lower():
            return ThreatCategory.PHISHING
        elif "dos" in " ".join(event_types.keys()).lower():
            return ThreatCategory.DENIAL_OF_SERVICE
        elif len(set(e.source_ip for e in events if e.source_ip)) == 1:
            # Single source IP might indicate APT
            return ThreatCategory.ADVANCED_PERSISTENT_THREAT
        else:
            return ThreatCategory.VULNERABILITY_EXPLOIT  # Default
            
    async def _calculate_cluster_score(self, events: List[ThreatEvent]) -> float:
        """Calculate overall cluster threat score"""
        
        if not events:
            return 0.0
            
        # Weight factors
        severity_weight = 0.4
        frequency_weight = 0.3
        diversity_weight = 0.2
        recency_weight = 0.1
        
        # Severity score
        avg_severity = np.mean([e.severity for e in events])
        
        # Frequency score (events per hour)
        time_span = (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds()
        frequency_score = len(events) / max(1, time_span / 3600)  # events per hour
        frequency_score = min(1.0, frequency_score / 10)  # normalize to 0-1
        
        # Diversity score (unique sources/targets)
        unique_sources = len(set(e.source_ip for e in events if e.source_ip))
        unique_targets = len(set(e.target_ip for e in events if e.target_ip))
        diversity_score = (unique_sources + unique_targets) / (len(events) * 2)
        
        # Recency score (how recent are the events)
        now = datetime.utcnow()
        avg_age_hours = np.mean([(now - e.timestamp).total_seconds() / 3600 for e in events])
        recency_score = max(0, 1.0 - avg_age_hours / 24)  # decay over 24 hours
        
        # Combined score
        cluster_score = (
            avg_severity * severity_weight +
            frequency_score * frequency_weight +
            diversity_score * diversity_weight +
            recency_score * recency_weight
        )
        
        return min(1.0, cluster_score)
        
    async def _calculate_cluster_cohesion(self, events: List[ThreatEvent],
                                        features: np.ndarray) -> float:
        """Calculate cluster cohesion (internal similarity)"""
        
        if len(events) < 2:
            return 1.0
            
        # Get features for these events
        event_indices = list(range(len(events)))
        cluster_features = features[event_indices]
        
        # Calculate pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        
        distances = euclidean_distances(cluster_features)
        
        # Cohesion is inverse of average distance
        avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
        cohesion = 1.0 / (1.0 + avg_distance)
        
        return cohesion
        
    async def _refine_clusters(self, clusters: List[ThreatCluster]) -> List[ThreatCluster]:
        """Refine clusters by merging similar ones and filtering weak ones"""
        
        refined_clusters = []
        
        # Filter out weak clusters
        for cluster in clusters:
            if cluster.cluster_score > 0.3 and cluster.size >= self.min_cluster_size:
                refined_clusters.append(cluster)
                
        # TODO: Implement cluster merging logic for very similar clusters
        
        return refined_clusters


class CausalInferenceEngine:
    """Causal inference engine for threat analysis"""
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.causal_relations: Dict[str, CausalRelation] = {}
        self.feature_importance_cache: Dict[str, Dict[str, float]] = {}
        
    async def discover_causal_relations(self, events: List[ThreatEvent],
                                      features: List[str]) -> List[CausalRelation]:
        """Discover causal relationships between threat features"""
        
        # Create feature matrix
        feature_matrix = await self._create_feature_matrix(events, features)
        
        # Apply causal discovery algorithms
        relations = []
        
        # Granger causality for temporal relationships
        temporal_relations = await self._granger_causality_analysis(feature_matrix, features)
        relations.extend(temporal_relations)
        
        # Pearl's causal inference for structural relationships
        structural_relations = await self._structural_causal_analysis(feature_matrix, features)
        relations.extend(structural_relations)
        
        # Information-theoretic causality
        information_relations = await self._information_causality_analysis(feature_matrix, features)
        relations.extend(information_relations)
        
        # Update causal graph
        await self._update_causal_graph(relations)
        
        return relations
        
    async def _create_feature_matrix(self, events: List[ThreatEvent],
                                   features: List[str]) -> pd.DataFrame:
        """Create feature matrix for causal analysis"""
        
        data = []
        
        for event in events:
            row = {
                'timestamp': event.timestamp,
                'severity': event.severity,
                'event_type': event.event_type,
                'source_ip': event.source_ip,
                'target_ip': event.target_ip,
                'protocol': event.protocol
            }
            
            # Add custom features
            for feature in features:
                row[feature] = event.features.get(feature, 0)
                
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Convert categorical variables to numerical
        for col in ['event_type', 'source_ip', 'target_ip', 'protocol']:
            if col in df.columns:
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
                
        return df
        
    async def _granger_causality_analysis(self, data: pd.DataFrame,
                                        features: List[str]) -> List[CausalRelation]:
        """Granger causality test for temporal relationships"""
        
        relations = []
        
        # Sort by timestamp for time series analysis
        data_sorted = data.sort_values('timestamp')
        
        # Test all pairs of numerical features
        numerical_features = data_sorted.select_dtypes(include=[np.number]).columns
        
        for cause_feature in numerical_features:
            for effect_feature in numerical_features:
                if cause_feature != effect_feature:
                    
                    # Simplified Granger causality test
                    strength, confidence = await self._test_granger_causality(
                        data_sorted[cause_feature], 
                        data_sorted[effect_feature]
                    )
                    
                    if confidence > 0.7:  # Significance threshold
                        relation = CausalRelation(
                            relation_id=f"granger_{cause_feature}_{effect_feature}_{int(time.time())}",
                            cause_event=cause_feature,
                            effect_event=effect_feature,
                            relation_type=CausalRelationType.DIRECT_CAUSE if strength > 0.8 else CausalRelationType.INDIRECT_CAUSE,
                            strength=strength,
                            confidence=confidence,
                            evidence={"method": "granger_causality", "lag": 1}
                        )
                        relations.append(relation)
                        
        return relations
        
    async def _test_granger_causality(self, cause_series: pd.Series,
                                    effect_series: pd.Series, 
                                    max_lag: int = 3) -> Tuple[float, float]:
        """Simplified Granger causality test"""
        
        try:
            # Calculate lagged correlation
            correlations = []
            
            for lag in range(1, max_lag + 1):
                if len(cause_series) > lag:
                    cause_lagged = cause_series.shift(lag)
                    correlation = cause_lagged.corr(effect_series)
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
                        
            if correlations:
                strength = max(correlations)
                confidence = np.mean(correlations)
                return strength, confidence
            else:
                return 0.0, 0.0
                
        except Exception as e:
            logger.error(f"Granger causality test error: {e}")
            return 0.0, 0.0
            
    async def _structural_causal_analysis(self, data: pd.DataFrame,
                                        features: List[str]) -> List[CausalRelation]:
        """Structural causal model analysis"""
        
        relations = []
        
        # Use conditional independence tests
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for i, cause in enumerate(numerical_features):
            for j, effect in enumerate(numerical_features):
                if i != j:
                    for k, mediator in enumerate(numerical_features):
                        if k != i and k != j:
                            
                            # Test conditional independence
                            independence_score = await self._test_conditional_independence(
                                data, cause, effect, mediator
                            )
                            
                            if independence_score < 0.3:  # Strong dependence
                                relation_type = CausalRelationType.MEDIATING
                                strength = 1.0 - independence_score
                                confidence = 0.8
                                
                                relation = CausalRelation(
                                    relation_id=f"structural_{cause}_{effect}_{mediator}_{int(time.time())}",
                                    cause_event=cause,
                                    effect_event=effect,
                                    relation_type=relation_type,
                                    strength=strength,
                                    confidence=confidence,
                                    evidence={"method": "conditional_independence", "mediator": mediator}
                                )
                                relations.append(relation)
                                
        return relations
        
    async def _test_conditional_independence(self, data: pd.DataFrame,
                                           cause: str, effect: str, 
                                           mediator: str) -> float:
        """Test conditional independence between cause and effect given mediator"""
        
        try:
            # Partial correlation as proxy for conditional independence
            from scipy.stats import pearsonr
            
            # Calculate correlations
            cause_effect_corr = data[cause].corr(data[effect])
            cause_mediator_corr = data[cause].corr(data[mediator])
            effect_mediator_corr = data[effect].corr(data[mediator])
            
            # Partial correlation formula
            numerator = cause_effect_corr - (cause_mediator_corr * effect_mediator_corr)
            denominator = np.sqrt((1 - cause_mediator_corr**2) * (1 - effect_mediator_corr**2))
            
            if denominator != 0:
                partial_corr = numerator / denominator
                return abs(partial_corr)
            else:
                return 1.0  # Independence
                
        except Exception as e:
            logger.error(f"Conditional independence test error: {e}")
            return 1.0
            
    async def _information_causality_analysis(self, data: pd.DataFrame,
                                            features: List[str]) -> List[CausalRelation]:
        """Information-theoretic causality analysis"""
        
        relations = []
        
        # Transfer entropy as causality measure
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for cause in numerical_features:
            for effect in numerical_features:
                if cause != effect:
                    
                    transfer_entropy = await self._calculate_transfer_entropy(
                        data[cause], data[effect]
                    )
                    
                    if transfer_entropy > 0.1:  # Threshold for significance
                        relation = CausalRelation(
                            relation_id=f"transfer_entropy_{cause}_{effect}_{int(time.time())}",
                            cause_event=cause,
                            effect_event=effect,
                            relation_type=CausalRelationType.DIRECT_CAUSE,
                            strength=transfer_entropy,
                            confidence=min(1.0, transfer_entropy * 2),
                            evidence={"method": "transfer_entropy"}
                        )
                        relations.append(relation)
                        
        return relations
        
    async def _calculate_transfer_entropy(self, cause_series: pd.Series,
                                        effect_series: pd.Series) -> float:
        """Calculate transfer entropy between two time series"""
        
        try:
            # Simplified transfer entropy calculation
            # In practice, would use more sophisticated methods
            
            # Discretize series
            cause_discrete = pd.cut(cause_series, bins=5, labels=False)
            effect_discrete = pd.cut(effect_series, bins=5, labels=False)
            
            # Calculate mutual information
            from sklearn.metrics import mutual_info_score
            
            # Shift effect series to test causal direction
            effect_shifted = effect_discrete.shift(-1).dropna()
            cause_aligned = cause_discrete[:len(effect_shifted)]
            
            if len(cause_aligned) > 0:
                mi = mutual_info_score(cause_aligned, effect_shifted)
                return mi
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Transfer entropy calculation error: {e}")
            return 0.0
            
    async def _update_causal_graph(self, relations: List[CausalRelation]):
        """Update causal graph with discovered relations"""
        
        for relation in relations:
            # Add nodes
            self.causal_graph.add_node(relation.cause_event)
            self.causal_graph.add_node(relation.effect_event)
            
            # Add edge with attributes
            self.causal_graph.add_edge(
                relation.cause_event,
                relation.effect_event,
                weight=relation.strength,
                relation_type=relation.relation_type.value,
                confidence=relation.confidence
            )
            
            # Store relation
            self.causal_relations[relation.relation_id] = relation
            
    async def get_causal_path(self, cause: str, effect: str) -> List[str]:
        """Find causal path between cause and effect"""
        
        try:
            if self.causal_graph.has_node(cause) and self.causal_graph.has_node(effect):
                path = nx.shortest_path(self.causal_graph, cause, effect)
                return path
            else:
                return []
        except nx.NetworkXNoPath:
            return []


class AdaptiveScoringEngine:
    """Adaptive scoring system that learns from feedback"""
    
    def __init__(self):
        self.scoring_models: Dict[str, Any] = {}
        self.feedback_history: List[Dict[str, Any]] = []
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
    async def calculate_adaptive_score(self, entity_id: str, entity_data: Dict[str, Any],
                                     score_type: str = "risk") -> AdaptiveScore:
        """Calculate adaptive score with continuous learning"""
        
        # Get or create scoring model
        if score_type not in self.scoring_models:
            await self._initialize_scoring_model(score_type)
            
        # Calculate base score
        base_score = await self._calculate_base_score(entity_data, score_type)
        
        # Apply adaptive adjustments
        adjustment_factors = await self._calculate_adjustments(entity_id, entity_data, score_type)
        
        # Combine base score with adjustments
        adjusted_score = await self._apply_adjustments(base_score, adjustment_factors)
        
        # Calculate confidence
        confidence = await self._calculate_confidence(entity_id, score_type)
        
        adaptive_score = AdaptiveScore(
            score_id=f"{score_type}_{entity_id}_{int(time.time())}",
            entity_id=entity_id,
            score_type=score_type,
            base_score=base_score,
            adjusted_score=adjusted_score,
            adjustment_factors=adjustment_factors,
            confidence=confidence
        )
        
        return adaptive_score
        
    async def _initialize_scoring_model(self, score_type: str):
        """Initialize scoring model for given type"""
        
        # Default models for different score types
        if score_type == "risk":
            self.scoring_models[score_type] = {
                "weights": {
                    "severity": 0.3,
                    "frequency": 0.25,
                    "impact": 0.2,
                    "likelihood": 0.15,
                    "detectability": 0.1
                },
                "adjustments": {
                    "temporal_decay": 0.95,
                    "frequency_boost": 1.2,
                    "false_positive_penalty": 0.8
                }
            }
        elif score_type == "priority":
            self.scoring_models[score_type] = {
                "weights": {
                    "business_impact": 0.4,
                    "urgency": 0.3,
                    "resource_availability": 0.2,
                    "complexity": 0.1
                }
            }
            
        self.model_performance[score_type] = {
            "accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.8,
            "f1_score": 0.68
        }
        
    async def _calculate_base_score(self, entity_data: Dict[str, Any], score_type: str) -> float:
        """Calculate base score using model weights"""
        
        model = self.scoring_models[score_type]
        weights = model["weights"]
        
        score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in entity_data:
                value = float(entity_data[feature])
                score += value * weight
                total_weight += weight
                
        # Normalize score
        if total_weight > 0:
            score = score / total_weight
            
        return min(1.0, max(0.0, score))
        
    async def _calculate_adjustments(self, entity_id: str, entity_data: Dict[str, Any],
                                   score_type: str) -> Dict[str, float]:
        """Calculate adaptive adjustments based on historical data"""
        
        adjustments = {}
        
        # Temporal adjustment (recent events get higher weight)
        age_hours = entity_data.get("age_hours", 0)
        temporal_factor = np.exp(-age_hours / 24)  # Decay over 24 hours
        adjustments["temporal"] = temporal_factor
        
        # Frequency adjustment (repeated patterns)
        frequency = entity_data.get("frequency", 1)
        frequency_factor = 1.0 + np.log(frequency) / 10
        adjustments["frequency"] = min(2.0, frequency_factor)
        
        # Confidence adjustment (based on model performance)
        model_accuracy = self.model_performance.get(score_type, {}).get("accuracy", 0.7)
        confidence_factor = 0.5 + model_accuracy * 0.5
        adjustments["confidence"] = confidence_factor
        
        # Feedback adjustment (learn from historical feedback)
        feedback_factor = await self._get_feedback_adjustment(entity_id, score_type)
        adjustments["feedback"] = feedback_factor
        
        return adjustments
        
    async def _get_feedback_adjustment(self, entity_id: str, score_type: str) -> float:
        """Get adjustment factor based on historical feedback"""
        
        # Look for relevant feedback
        relevant_feedback = [
            f for f in self.feedback_history
            if f.get("entity_id") == entity_id and f.get("score_type") == score_type
        ]
        
        if not relevant_feedback:
            return 1.0
            
        # Calculate average feedback adjustment
        adjustments = [f.get("adjustment", 1.0) for f in relevant_feedback[-10:]]  # Last 10
        return np.mean(adjustments)
        
    async def _apply_adjustments(self, base_score: float,
                               adjustment_factors: Dict[str, float]) -> float:
        """Apply adjustment factors to base score"""
        
        adjusted_score = base_score
        
        # Apply multiplicative adjustments
        for factor_name, factor_value in adjustment_factors.items():
            if factor_name in ["temporal", "frequency", "confidence"]:
                adjusted_score *= factor_value
            elif factor_name == "feedback":
                # Additive adjustment for feedback
                adjusted_score += (factor_value - 1.0) * 0.1
                
        return min(1.0, max(0.0, adjusted_score))
        
    async def _calculate_confidence(self, entity_id: str, score_type: str) -> float:
        """Calculate confidence in the adaptive score"""
        
        # Base confidence from model performance
        base_confidence = self.model_performance.get(score_type, {}).get("accuracy", 0.7)
        
        # Adjust based on data completeness
        # This would need access to the actual entity data completeness
        data_completeness = 0.8  # Placeholder
        
        # Adjust based on feedback history
        feedback_count = len([
            f for f in self.feedback_history
            if f.get("entity_id") == entity_id and f.get("score_type") == score_type
        ])
        feedback_factor = min(1.0, feedback_count / 10)  # More feedback = higher confidence
        
        confidence = base_confidence * data_completeness * (0.5 + feedback_factor * 0.5)
        return min(1.0, max(0.1, confidence))
        
    async def provide_feedback(self, score_id: str, actual_outcome: float,
                             feedback_type: str = "accuracy"):
        """Provide feedback to improve adaptive scoring"""
        
        feedback = {
            "score_id": score_id,
            "actual_outcome": actual_outcome,
            "feedback_type": feedback_type,
            "timestamp": datetime.utcnow().isoformat(),
            "adjustment": actual_outcome  # Simplified
        }
        
        self.feedback_history.append(feedback)
        
        # Update model performance (simplified)
        await self._update_model_performance(feedback)
        
    async def _update_model_performance(self, feedback: Dict[str, Any]):
        """Update model performance metrics based on feedback"""
        
        # This is a simplified implementation
        # In practice, would use more sophisticated learning algorithms
        
        score_type = feedback.get("score_type", "risk")
        
        if score_type in self.model_performance:
            current_accuracy = self.model_performance[score_type]["accuracy"]
            
            # Simple exponential moving average
            alpha = 0.1
            feedback_accuracy = min(1.0, feedback.get("actual_outcome", 0.5))
            
            new_accuracy = alpha * feedback_accuracy + (1 - alpha) * current_accuracy
            self.model_performance[score_type]["accuracy"] = new_accuracy


class ThreatFeatureExtractor:
    """Extract features from threat events for analysis"""
    
    def __init__(self):
        self.feature_extractors = {
            "temporal": self._extract_temporal_features,
            "network": self._extract_network_features,
            "content": self._extract_content_features,
            "behavioral": self._extract_behavioral_features
        }
        
    async def extract_features(self, event: ThreatEvent) -> Dict[str, Any]:
        """Extract all features from threat event"""
        
        features = {}
        
        for extractor_name, extractor_func in self.feature_extractors.items():
            try:
                extracted = await extractor_func(event)
                features.update(extracted)
            except Exception as e:
                logger.error(f"Feature extraction error ({extractor_name}): {e}")
                
        return features
        
    async def _extract_temporal_features(self, event: ThreatEvent) -> Dict[str, Any]:
        """Extract temporal features"""
        
        return {
            "hour_of_day": event.timestamp.hour,
            "day_of_week": event.timestamp.weekday(),
            "month": event.timestamp.month,
            "is_weekend": event.timestamp.weekday() >= 5,
            "is_business_hours": 9 <= event.timestamp.hour <= 17
        }
        
    async def _extract_network_features(self, event: ThreatEvent) -> Dict[str, Any]:
        """Extract network-based features"""
        
        features = {}
        
        if event.source_ip:
            features.update({
                "source_ip_class": self._get_ip_class(event.source_ip),
                "source_is_private": self._is_private_ip(event.source_ip),
                "source_country": "unknown"  # Would use GeoIP lookup
            })
            
        if event.target_ip:
            features.update({
                "target_ip_class": self._get_ip_class(event.target_ip),
                "target_is_private": self._is_private_ip(event.target_ip),
                "target_country": "unknown"  # Would use GeoIP lookup
            })
            
        if event.protocol:
            features["protocol_type"] = event.protocol.lower()
            
        return features
        
    async def _extract_content_features(self, event: ThreatEvent) -> Dict[str, Any]:
        """Extract content-based features"""
        
        features = {
            "event_type_length": len(event.event_type),
            "feature_count": len(event.features),
            "metadata_count": len(event.metadata)
        }
        
        # Analyze event type
        event_type_lower = event.event_type.lower()
        features.update({
            "contains_attack": "attack" in event_type_lower,
            "contains_scan": "scan" in event_type_lower,
            "contains_exploit": "exploit" in event_type_lower,
            "contains_malware": "malware" in event_type_lower
        })
        
        return features
        
    async def _extract_behavioral_features(self, event: ThreatEvent) -> Dict[str, Any]:
        """Extract behavioral features"""
        
        # This would analyze patterns across multiple events
        # For now, return basic features
        
        return {
            "severity_level": "high" if event.severity > 0.7 else "medium" if event.severity > 0.4 else "low",
            "risk_category": event.risk_score
        }
        
    def _get_ip_class(self, ip: str) -> str:
        """Get IP address class (A, B, C)"""
        try:
            first_octet = int(ip.split('.')[0])
            if 1 <= first_octet <= 126:
                return "A"
            elif 128 <= first_octet <= 191:
                return "B"
            elif 192 <= first_octet <= 223:
                return "C"
            else:
                return "Other"
        except:
            return "Unknown"
            
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private range"""
        try:
            import ipaddress
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except:
            return False


class CausalAnalyticsEngine:
    """Main causal analytics engine combining all components"""
    
    def __init__(self):
        self.threat_clustering = ThreatClusteringEngine()
        self.causal_inference = CausalInferenceEngine()
        self.adaptive_scoring = AdaptiveScoringEngine()
        self.feature_extractor = ThreatFeatureExtractor()
        
        # Analytics state
        self.analysis_history: List[Dict[str, Any]] = []
        self.active_models: Dict[str, Any] = {}
        
    async def comprehensive_analysis(self, events: List[ThreatEvent],
                                   analysis_types: List[AnalysisType] = None) -> Dict[str, Any]:
        """Perform comprehensive causal analysis"""
        
        if not analysis_types:
            analysis_types = [
                AnalysisType.THREAT_CLUSTERING,
                AnalysisType.CAUSAL_INFERENCE,
                AnalysisType.ANOMALY_DETECTION
            ]
            
        results = {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "event_count": len(events),
            "analysis_types": [at.value for at in analysis_types]
        }
        
        # Extract features for all events
        logger.info("Extracting features from events...")
        enriched_events = []
        for event in events:
            features = await self.feature_extractor.extract_features(event)
            event.features.update(features)
            enriched_events.append(event)
            
        # Threat clustering
        if AnalysisType.THREAT_CLUSTERING in analysis_types:
            logger.info("Performing threat clustering...")
            clusters = await self.threat_clustering.cluster_threats(enriched_events)
            results["threat_clusters"] = [
                {
                    "cluster_id": c.cluster_id,
                    "size": c.size,
                    "threat_category": c.threat_category.value,
                    "cluster_score": c.cluster_score,
                    "cohesion": c.cohesion
                }
                for c in clusters
            ]
            
        # Causal inference
        if AnalysisType.CAUSAL_INFERENCE in analysis_types:
            logger.info("Discovering causal relationships...")
            feature_names = list(enriched_events[0].features.keys()) if enriched_events else []
            causal_relations = await self.causal_inference.discover_causal_relations(
                enriched_events, feature_names
            )
            results["causal_relations"] = [
                {
                    "cause": r.cause_event,
                    "effect": r.effect_event,
                    "type": r.relation_type.value,
                    "strength": r.strength,
                    "confidence": r.confidence
                }
                for r in causal_relations
            ]
            
        # Anomaly detection
        if AnalysisType.ANOMALY_DETECTION in analysis_types:
            logger.info("Detecting anomalies...")
            anomalies = await self._detect_anomalies(enriched_events)
            results["anomalies"] = anomalies
            
        # Adaptive scoring
        for event in enriched_events[:10]:  # Score first 10 events
            adaptive_score = await self.adaptive_scoring.calculate_adaptive_score(
                event.event_id, event.features, "risk"
            )
            if "adaptive_scores" not in results:
                results["adaptive_scores"] = []
            results["adaptive_scores"].append({
                "event_id": event.event_id,
                "base_score": adaptive_score.base_score,
                "adjusted_score": adaptive_score.adjusted_score,
                "confidence": adaptive_score.confidence
            })
            
        # Store analysis
        self.analysis_history.append(results)
        
        logger.info(f"Completed comprehensive analysis: {results['analysis_id']}")
        return results
        
    async def _detect_anomalies(self, events: List[ThreatEvent]) -> List[Dict[str, Any]]:
        """Detect anomalous events using isolation forest"""
        
        if len(events) < 10:
            return []
            
        # Create feature matrix
        feature_matrix = []
        for event in events:
            features = [
                event.severity,
                event.timestamp.hour,
                event.timestamp.weekday(),
                len(event.features),
                event.risk_score
            ]
            feature_matrix.append(features)
            
        # Apply isolation forest
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = isolation_forest.fit_predict(feature_matrix)
        
        # Identify anomalies
        anomalies = []
        for i, (event, score) in enumerate(zip(events, anomaly_scores)):
            if score == -1:  # Anomaly
                anomalies.append({
                    "event_id": event.event_id,
                    "severity": event.severity,
                    "timestamp": event.timestamp.isoformat(),
                    "anomaly_score": float(isolation_forest.score_samples([feature_matrix[i]])[0])
                })
                
        return anomalies
        
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        
        return {
            "total_analyses": len(self.analysis_history),
            "threat_clusters_discovered": len(self.threat_clustering.clusters),
            "causal_relations_discovered": len(self.causal_inference.causal_relations),
            "adaptive_models_active": len(self.adaptive_scoring.scoring_models),
            "recent_analyses": self.analysis_history[-5:] if self.analysis_history else [],
            "system_status": {
                "clustering_engine": "active",
                "causal_inference_engine": "active",
                "adaptive_scoring_engine": "active",
                "feature_extractor": "active"
            }
        }