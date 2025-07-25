#!/usr/bin/env python3
"""
Root Cause Attribution System for Xorb 2.0

This module implements sophisticated root cause analysis capabilities using causal inference,
graph propagation, and statistical methods to identify the underlying causes of mispredictions,
failures, and performance issues in the AI orchestration system.
"""

import asyncio
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from enum import Enum
import json
import statistics
from pathlib import Path


class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"           # A directly causes B
    INDIRECT_CAUSE = "indirect_cause"       # A causes B through intermediates
    CONTRIBUTING_FACTOR = "contributing_factor"  # A contributes to B
    NECESSARY_CONDITION = "necessary_condition"   # A is necessary for B
    SUFFICIENT_CONDITION = "sufficient_condition" # A is sufficient for B
    INHIBITING_FACTOR = "inhibiting_factor"      # A prevents or reduces B
    CONFOUNDING_FACTOR = "confounding_factor"    # A affects both cause and effect


class FailureType(Enum):
    """Types of failures to analyze"""
    PREDICTION_ERROR = "prediction_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT_FAILURE = "timeout_failure"
    SECURITY_BREACH = "security_breach"
    INTEGRATION_FAILURE = "integration_failure"
    DATA_QUALITY_ISSUE = "data_quality_issue"


@dataclass
class CausalFactor:
    """Represents a potential causal factor"""
    factor_id: str
    factor_name: str
    factor_type: str  # 'environmental', 'system', 'data', 'configuration', 'temporal'
    
    # Factor characteristics
    importance_score: float = 0.0
    confidence_score: float = 0.0
    temporal_priority: float = 0.0  # How early this factor appeared
    
    # Statistical properties
    correlation_strength: float = 0.0
    causal_strength: float = 0.0
    frequency_of_occurrence: int = 0
    
    # Context information
    context_data: Dict[str, Any] = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)
    
    # Temporal information
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    observation_count: int = 1


@dataclass
class CausalRelation:
    """Represents a causal relationship between factors"""
    relation_id: str
    cause_factor_id: str
    effect_factor_id: str
    relation_type: CausalRelationType
    
    # Relationship strength
    causal_strength: float = 0.0
    confidence: float = 0.0
    statistical_significance: float = 0.0
    
    # Evidence supporting the relationship
    evidence_count: int = 0
    correlation_coefficient: float = 0.0
    temporal_precedence_score: float = 0.0
    intervention_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Contextual information
    contexts_observed: List[str] = field(default_factory=list)
    mediating_factors: List[str] = field(default_factory=list)
    
    # Temporal information
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RootCauseAnalysisResult:
    """Results of root cause analysis"""
    analysis_id: str
    failure_event_id: str
    failure_type: FailureType
    timestamp: datetime
    
    # Primary root causes
    primary_causes: List[str]  # Factor IDs
    contributing_factors: List[str]  # Factor IDs
    inhibiting_factors: List[str]  # Factor IDs
    
    # Causal pathway
    causal_chain: List[Tuple[str, str, float]]  # (cause, effect, strength)
    causal_graph_summary: Dict[str, Any]
    
    # Confidence and metrics
    overall_confidence: float = 0.0
    explanation_completeness: float = 0.0
    prediction_accuracy: float = 0.0
    
    # Recommendations
    preventive_actions: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)
    system_improvements: List[str] = field(default_factory=list)
    
    # Analysis metadata
    analysis_duration_seconds: float = 0.0
    factors_considered: int = 0
    relationships_analyzed: int = 0


class CausalInferenceEngine:
    """Engine for causal inference using various statistical methods"""
    
    def __init__(self, significance_threshold: float = 0.05):
        self.significance_threshold = significance_threshold
        self.logger = logging.getLogger(__name__)
    
    async def infer_causal_relationships(self,
                                       factors: List[CausalFactor],
                                       events_data: List[Dict[str, Any]]) -> List[CausalRelation]:
        """Infer causal relationships between factors using multiple methods"""
        
        relations = []
        
        # Method 1: Temporal precedence analysis
        temporal_relations = await self._temporal_precedence_analysis(factors, events_data)
        relations.extend(temporal_relations)
        
        # Method 2: Correlation and conditional independence
        correlation_relations = await self._correlation_analysis(factors, events_data)
        relations.extend(correlation_relations)
        
        # Method 3: Intervention analysis (if intervention data is available)
        intervention_relations = await self._intervention_analysis(factors, events_data)
        relations.extend(intervention_relations)
        
        # Method 4: Counterfactual reasoning
        counterfactual_relations = await self._counterfactual_analysis(factors, events_data)
        relations.extend(counterfactual_relations)
        
        # Consolidate and rank relationships
        consolidated_relations = await self._consolidate_relations(relations)
        
        return consolidated_relations
    
    async def _temporal_precedence_analysis(self,
                                          factors: List[CausalFactor],
                                          events_data: List[Dict[str, Any]]) -> List[CausalRelation]:
        """Analyze temporal precedence to infer causality"""
        
        relations = []
        
        # Create timeline of factor observations
        factor_timeline = defaultdict(list)
        
        for event in events_data:
            timestamp = event.get('timestamp', datetime.utcnow())
            for factor_id, factor_data in event.get('factors', {}).items():
                factor_timeline[factor_id].append({
                    'timestamp': timestamp,
                    'value': factor_data.get('value', 0),
                    'context': factor_data.get('context', {})
                })
        
        # Analyze precedence patterns
        for cause_factor in factors:
            for effect_factor in factors:
                if cause_factor.factor_id == effect_factor.factor_id:
                    continue
                
                precedence_score = await self._calculate_temporal_precedence(
                    cause_factor.factor_id, effect_factor.factor_id, factor_timeline
                )
                
                if precedence_score > 0.6:  # Significant temporal precedence
                    relation = CausalRelation(
                        relation_id=f"temporal_{cause_factor.factor_id}_{effect_factor.factor_id}",
                        cause_factor_id=cause_factor.factor_id,
                        effect_factor_id=effect_factor.factor_id,
                        relation_type=CausalRelationType.DIRECT_CAUSE,
                        causal_strength=precedence_score * 0.7,  # Temporal evidence alone
                        confidence=precedence_score,
                        temporal_precedence_score=precedence_score,
                        evidence_count=len(factor_timeline[cause_factor.factor_id])
                    )
                    relations.append(relation)
        
        return relations
    
    async def _correlation_analysis(self,
                                  factors: List[CausalFactor],
                                  events_data: List[Dict[str, Any]]) -> List[CausalRelation]:
        """Analyze correlations and conditional dependencies"""
        
        relations = []
        
        # Extract factor values for correlation analysis
        factor_values = defaultdict(list)
        
        for event in events_data:
            for factor_id, factor_data in event.get('factors', {}).items():
                factor_values[factor_id].append(factor_data.get('value', 0))
        
        # Calculate pairwise correlations
        for cause_factor in factors:
            for effect_factor in factors:
                if cause_factor.factor_id == effect_factor.factor_id:
                    continue
                
                cause_values = factor_values.get(cause_factor.factor_id, [])
                effect_values = factor_values.get(effect_factor.factor_id, [])
                
                if len(cause_values) >= 10 and len(effect_values) >= 10:
                    # Ensure same length
                    min_length = min(len(cause_values), len(effect_values))
                    cause_values = cause_values[:min_length]
                    effect_values = effect_values[:min_length]
                    
                    # Calculate correlation
                    correlation = np.corrcoef(cause_values, effect_values)[0, 1]
                    
                    if not np.isnan(correlation) and abs(correlation) > 0.5:
                        # Test for conditional independence (simplified)
                        significance = await self._test_statistical_significance(
                            cause_values, effect_values
                        )
                        
                        if significance < self.significance_threshold:
                            relation_type = (CausalRelationType.DIRECT_CAUSE if abs(correlation) > 0.7
                                           else CausalRelationType.CONTRIBUTING_FACTOR)
                            
                            relation = CausalRelation(
                                relation_id=f"corr_{cause_factor.factor_id}_{effect_factor.factor_id}",
                                cause_factor_id=cause_factor.factor_id,
                                effect_factor_id=effect_factor.factor_id,
                                relation_type=relation_type,
                                causal_strength=abs(correlation) * 0.6,  # Correlation evidence
                                confidence=1.0 - significance,
                                correlation_coefficient=correlation,
                                statistical_significance=significance,
                                evidence_count=min_length
                            )
                            relations.append(relation)
        
        return relations
    
    async def _intervention_analysis(self,
                                   factors: List[CausalFactor],
                                   events_data: List[Dict[str, Any]]) -> List[CausalRelation]:
        """Analyze intervention data to infer causality"""
        
        relations = []
        
        # Look for intervention events in the data
        intervention_events = [
            event for event in events_data
            if event.get('event_type') == 'intervention'
        ]
        
        if not intervention_events:
            return relations
        
        # Analyze each intervention
        for intervention in intervention_events:
            intervened_factor = intervention.get('intervened_factor')
            if not intervened_factor:
                continue
            
            # Find before/after periods
            intervention_time = intervention.get('timestamp', datetime.utcnow())
            before_period = intervention_time - timedelta(hours=1)
            after_period = intervention_time + timedelta(hours=1)
            
            # Analyze effects of intervention
            for factor in factors:
                if factor.factor_id == intervened_factor:
                    continue
                
                effect_strength = await self._measure_intervention_effect(
                    intervened_factor, factor.factor_id, intervention_time, events_data
                )
                
                if effect_strength > 0.3:
                    relation = CausalRelation(
                        relation_id=f"interv_{intervened_factor}_{factor.factor_id}",
                        cause_factor_id=intervened_factor,
                        effect_factor_id=factor.factor_id,
                        relation_type=CausalRelationType.DIRECT_CAUSE,
                        causal_strength=effect_strength,
                        confidence=0.9,  # High confidence from intervention
                        evidence_count=1,
                        intervention_evidence=[{
                            'intervention_time': intervention_time,
                            'effect_magnitude': effect_strength,
                            'intervention_details': intervention
                        }]
                    )
                    relations.append(relation)
        
        return relations
    
    async def _counterfactual_analysis(self,
                                     factors: List[CausalFactor],
                                     events_data: List[Dict[str, Any]]) -> List[CausalRelation]:
        """Perform counterfactual reasoning to infer causality"""
        
        relations = []
        
        # Simple counterfactual analysis based on presence/absence patterns
        factor_presence = defaultdict(list)
        
        for event in events_data:
            present_factors = set(event.get('factors', {}).keys())
            event_outcome = event.get('outcome', 0)
            
            for factor_id in [f.factor_id for f in factors]:
                factor_presence[factor_id].append({
                    'present': factor_id in present_factors,
                    'outcome': event_outcome,
                    'timestamp': event.get('timestamp', datetime.utcnow())
                })
        
        # Analyze counterfactual patterns
        for cause_factor in factors:
            for effect_factor in factors:
                if cause_factor.factor_id == effect_factor.factor_id:
                    continue
                
                counterfactual_strength = await self._calculate_counterfactual_strength(
                    cause_factor.factor_id, effect_factor.factor_id, factor_presence, events_data
                )
                
                if counterfactual_strength > 0.4:
                    relation = CausalRelation(
                        relation_id=f"counter_{cause_factor.factor_id}_{effect_factor.factor_id}",
                        cause_factor_id=cause_factor.factor_id,
                        effect_factor_id=effect_factor.factor_id,
                        relation_type=CausalRelationType.NECESSARY_CONDITION,
                        causal_strength=counterfactual_strength,
                        confidence=counterfactual_strength * 0.8,
                        evidence_count=len(factor_presence[cause_factor.factor_id])
                    )
                    relations.append(relation)
        
        return relations
    
    async def _calculate_temporal_precedence(self,
                                           cause_id: str,
                                           effect_id: str,
                                           factor_timeline: Dict[str, List[Dict]]) -> float:
        """Calculate temporal precedence score between two factors"""
        
        cause_events = factor_timeline.get(cause_id, [])
        effect_events = factor_timeline.get(effect_id, [])
        
        if not cause_events or not effect_events:
            return 0.0
        
        precedence_count = 0
        total_pairs = 0
        
        # Count how often cause precedes effect
        for cause_event in cause_events:
            for effect_event in effect_events:
                total_pairs += 1
                
                # Check if cause precedes effect within reasonable time window
                time_diff = (effect_event['timestamp'] - cause_event['timestamp']).total_seconds()
                
                if 0 < time_diff <= 3600:  # Cause precedes effect within 1 hour
                    precedence_count += 1
        
        if total_pairs == 0:
            return 0.0
        
        precedence_score = precedence_count / total_pairs
        return precedence_score
    
    async def _test_statistical_significance(self,
                                           cause_values: List[float],
                                           effect_values: List[float]) -> float:
        """Test statistical significance of correlation"""
        
        # Simplified t-test for correlation significance
        n = len(cause_values)
        if n < 3:
            return 1.0  # Not significant
        
        correlation = np.corrcoef(cause_values, effect_values)[0, 1]
        if np.isnan(correlation):
            return 1.0
        
        # t-statistic for correlation
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        
        # Simplified p-value calculation (would use actual t-distribution in production)
        p_value = max(0.001, min(0.999, 2 * (1 - abs(t_stat) / 10.0)))
        
        return p_value
    
    async def _measure_intervention_effect(self,
                                         intervened_factor: str,
                                         target_factor: str,
                                         intervention_time: datetime,
                                         events_data: List[Dict[str, Any]]) -> float:
        """Measure the effect of intervention on target factor"""
        
        before_window = intervention_time - timedelta(hours=1)
        after_window = intervention_time + timedelta(hours=1)
        
        before_values = []
        after_values = []
        
        for event in events_data:
            event_time = event.get('timestamp', datetime.utcnow())
            factors = event.get('factors', {})
            
            if target_factor in factors:
                target_value = factors[target_factor].get('value', 0)
                
                if before_window <= event_time < intervention_time:
                    before_values.append(target_value)
                elif intervention_time < event_time <= after_window:
                    after_values.append(target_value)
        
        if not before_values or not after_values:
            return 0.0
        
        # Calculate effect size (normalized difference)
        before_mean = np.mean(before_values)
        after_mean = np.mean(after_values)
        
        if before_mean == 0:
            return 0.0
        
        effect_size = abs(after_mean - before_mean) / abs(before_mean)
        return min(1.0, effect_size)
    
    async def _calculate_counterfactual_strength(self,
                                               cause_id: str,
                                               effect_id: str,
                                               factor_presence: Dict[str, List[Dict]],
                                               events_data: List[Dict[str, Any]]) -> float:
        """Calculate counterfactual causal strength"""
        
        # Analyze outcomes when cause is present vs absent
        cause_present_outcomes = []
        cause_absent_outcomes = []
        
        for event in events_data:
            factors = event.get('factors', {})
            outcome = event.get('outcome', 0)
            
            # Check if effect factor is present/active in this event
            effect_present = effect_id in factors and factors[effect_id].get('value', 0) > 0
            
            if not effect_present:
                continue
            
            if cause_id in factors and factors[cause_id].get('value', 0) > 0:
                cause_present_outcomes.append(outcome)
            else:
                cause_absent_outcomes.append(outcome)
        
        if not cause_present_outcomes or not cause_absent_outcomes:
            return 0.0
        
        # Calculate difference in outcomes
        present_mean = np.mean(cause_present_outcomes)
        absent_mean = np.mean(cause_absent_outcomes)
        
        # Normalize by the baseline (absent) mean
        if absent_mean == 0:
            return 0.0
        
        counterfactual_strength = abs(present_mean - absent_mean) / abs(absent_mean)
        return min(1.0, counterfactual_strength)
    
    async def _consolidate_relations(self, relations: List[CausalRelation]) -> List[CausalRelation]:
        """Consolidate and rank causal relations from different methods"""
        
        # Group relations by cause-effect pair
        relation_groups = defaultdict(list)
        
        for relation in relations:
            key = (relation.cause_factor_id, relation.effect_factor_id)
            relation_groups[key].append(relation)
        
        consolidated_relations = []
        
        for (cause_id, effect_id), group_relations in relation_groups.items():
            if len(group_relations) == 1:
                consolidated_relations.append(group_relations[0])
            else:
                # Combine evidence from multiple methods
                consolidated_relation = await self._combine_relation_evidence(group_relations)
                consolidated_relations.append(consolidated_relation)
        
        # Rank by causal strength
        consolidated_relations.sort(key=lambda r: r.causal_strength, reverse=True)
        
        return consolidated_relations
    
    async def _combine_relation_evidence(self, relations: List[CausalRelation]) -> CausalRelation:
        """Combine evidence from multiple causal inference methods"""
        
        # Use the relation with highest individual strength as base
        base_relation = max(relations, key=lambda r: r.causal_strength)
        
        # Combine strengths using weighted average
        total_weight = 0
        weighted_strength = 0
        
        method_weights = {
            'interv': 1.0,    # Intervention evidence is strongest
            'counter': 0.8,   # Counterfactual evidence
            'temporal': 0.6,  # Temporal precedence
            'corr': 0.4       # Correlation evidence
        }
        
        for relation in relations:
            method = relation.relation_id.split('_')[0]
            weight = method_weights.get(method, 0.5)
            
            weighted_strength += weight * relation.causal_strength
            total_weight += weight
        
        if total_weight > 0:
            combined_strength = weighted_strength / total_weight
        else:
            combined_strength = base_relation.causal_strength
        
        # Create consolidated relation
        consolidated = CausalRelation(
            relation_id=f"consolidated_{base_relation.cause_factor_id}_{base_relation.effect_factor_id}",
            cause_factor_id=base_relation.cause_factor_id,
            effect_factor_id=base_relation.effect_factor_id,
            relation_type=base_relation.relation_type,
            causal_strength=combined_strength,
            confidence=min(0.95, sum(r.confidence for r in relations) / len(relations)),
            evidence_count=sum(r.evidence_count for r in relations),
            correlation_coefficient=np.mean([r.correlation_coefficient for r in relations 
                                           if r.correlation_coefficient != 0.0]),
            temporal_precedence_score=max(r.temporal_precedence_score for r in relations),
            intervention_evidence=[evidence for r in relations for evidence in r.intervention_evidence]
        )
        
        return consolidated


class GraphPropagationAnalyzer:
    """Analyzer for propagating causal effects through factor graphs"""
    
    def __init__(self, propagation_threshold: float = 0.3):
        self.propagation_threshold = propagation_threshold
        self.logger = logging.getLogger(__name__)
    
    async def build_causal_graph(self,
                               factors: List[CausalFactor],
                               relations: List[CausalRelation]) -> nx.DiGraph:
        """Build causal graph from factors and relations"""
        
        graph = nx.DiGraph()
        
        # Add factor nodes
        for factor in factors:
            graph.add_node(factor.factor_id, 
                          name=factor.factor_name,
                          type=factor.factor_type,
                          importance=factor.importance_score,
                          confidence=factor.confidence_score)
        
        # Add causal edges
        for relation in relations:
            if relation.causal_strength >= self.propagation_threshold:
                graph.add_edge(relation.cause_factor_id,
                             relation.effect_factor_id,
                             weight=relation.causal_strength,
                             relation_type=relation.relation_type.value,
                             confidence=relation.confidence)
        
        return graph
    
    async def propagate_root_causes(self,
                                  causal_graph: nx.DiGraph,
                                  failure_factors: List[str],
                                  max_depth: int = 5) -> Dict[str, float]:
        """Propagate backwards from failure factors to find root causes"""
        
        root_cause_scores = defaultdict(float)
        
        # For each failure factor, trace back through the causal graph
        for failure_factor in failure_factors:
            if failure_factor not in causal_graph:
                continue
            
            # Breadth-first search backwards through causal graph
            visited = set()
            queue = [(failure_factor, 1.0, 0)]  # (node, strength, depth)
            
            while queue:
                current_node, current_strength, depth = queue.pop(0)
                
                if depth >= max_depth or current_node in visited:
                    continue
                
                visited.add(current_node)
                
                # Add to root cause scores
                root_cause_scores[current_node] += current_strength
                
                # Propagate to predecessor nodes
                for predecessor in causal_graph.predecessors(current_node):
                    edge_data = causal_graph[predecessor][current_node]
                    edge_strength = edge_data.get('weight', 0.0)
                    
                    # Decay strength with distance and edge strength
                    propagated_strength = current_strength * edge_strength * (0.8 ** depth)
                    
                    if propagated_strength >= self.propagation_threshold:
                        queue.append((predecessor, propagated_strength, depth + 1))
        
        return dict(root_cause_scores)
    
    async def identify_causal_pathways(self,
                                     causal_graph: nx.DiGraph,
                                     root_causes: List[str],
                                     failure_factors: List[str]) -> List[List[str]]:
        """Identify causal pathways from root causes to failures"""
        
        causal_pathways = []
        
        for root_cause in root_causes:
            for failure_factor in failure_factors:
                if root_cause == failure_factor:
                    continue
                
                try:
                    # Find all simple paths (no cycles)
                    paths = list(nx.all_simple_paths(
                        causal_graph, root_cause, failure_factor, cutoff=6
                    ))
                    
                    # Score paths by cumulative causal strength
                    scored_paths = []
                    for path in paths:
                        path_strength = await self._calculate_path_strength(causal_graph, path)
                        if path_strength >= self.propagation_threshold:
                            scored_paths.append((path, path_strength))
                    
                    # Sort by strength and add top paths
                    scored_paths.sort(key=lambda x: x[1], reverse=True)
                    causal_pathways.extend([path for path, _ in scored_paths[:3]])  # Top 3 paths
                    
                except nx.NetworkXNoPath:
                    continue
        
        return causal_pathways
    
    async def _calculate_path_strength(self, graph: nx.DiGraph, path: List[str]) -> float:
        """Calculate cumulative strength of a causal path"""
        
        if len(path) < 2:
            return 0.0
        
        path_strength = 1.0
        
        for i in range(len(path) - 1):
            edge_data = graph[path[i]][path[i + 1]]
            edge_strength = edge_data.get('weight', 0.0)
            path_strength *= edge_strength
        
        return path_strength
    
    async def analyze_graph_properties(self, causal_graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze structural properties of the causal graph"""
        
        analysis = {}
        
        # Basic graph properties
        analysis['num_nodes'] = causal_graph.number_of_nodes()
        analysis['num_edges'] = causal_graph.number_of_edges()
        analysis['density'] = nx.density(causal_graph)
        
        # Centrality measures
        if causal_graph.number_of_nodes() > 0:
            analysis['in_degree_centrality'] = nx.in_degree_centrality(causal_graph)
            analysis['out_degree_centrality'] = nx.out_degree_centrality(causal_graph)
            analysis['betweenness_centrality'] = nx.betweenness_centrality(causal_graph)
            analysis['pagerank'] = nx.pagerank(causal_graph)
        
        # Identify strongly connected components
        analysis['strongly_connected_components'] = [
            list(component) for component in nx.strongly_connected_components(causal_graph)
        ]
        
        # Find cycles (potential feedback loops)
        try:
            analysis['cycles'] = list(nx.simple_cycles(causal_graph))
        except:
            analysis['cycles'] = []
        
        # Topological properties
        analysis['is_dag'] = nx.is_directed_acyclic_graph(causal_graph)
        
        if analysis['is_dag']:
            analysis['topological_sort'] = list(nx.topological_sort(causal_graph))
        
        return analysis


class RootCauseAttributionSystem:
    """Main root cause attribution system"""
    
    def __init__(self,
                 max_factors: int = 1000,
                 max_relations: int = 5000):
        
        self.max_factors = max_factors
        self.max_relations = max_relations
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.causal_inference_engine = CausalInferenceEngine()
        self.graph_propagation_analyzer = GraphPropagationAnalyzer()
        
        # Knowledge storage
        self.factors: Dict[str, CausalFactor] = {}
        self.relations: Dict[str, CausalRelation] = {}
        
        # Analysis history
        self.analysis_history: List[RootCauseAnalysisResult] = []
        self.max_history_size = 100
        
        # Performance metrics
        self.attribution_metrics = {
            'total_analyses_performed': 0,
            'average_analysis_time_seconds': 0.0,
            'accuracy_rate': 0.0,
            'factors_identified': 0,
            'relations_discovered': 0,
            'most_common_root_causes': Counter(),
            'success_rate_by_failure_type': defaultdict(list)
        }
        
        self.logger.info("Root cause attribution system initialized")
    
    async def analyze_failure_event(self,
                                  failure_event: Dict[str, Any],
                                  historical_data: List[Dict[str, Any]] = None) -> RootCauseAnalysisResult:
        """Perform comprehensive root cause analysis for a failure event"""
        
        analysis_start_time = datetime.utcnow()
        
        # Extract basic information
        failure_event_id = failure_event.get('event_id', f"failure_{analysis_start_time.strftime('%Y%m%d_%H%M%S')}")
        failure_type = FailureType(failure_event.get('failure_type', 'prediction_error'))
        
        # Extract and analyze factors
        event_factors = await self._extract_factors_from_event(failure_event)
        historical_factors = await self._extract_factors_from_history(historical_data or [])
        
        all_factors = list({f.factor_id: f for f in event_factors + historical_factors}.values())
        
        # Update factor knowledge base
        await self._update_factor_knowledge(all_factors)
        
        # Infer causal relationships
        all_events_data = [failure_event] + (historical_data or [])
        causal_relations = await self.causal_inference_engine.infer_causal_relationships(
            all_factors, all_events_data
        )
        
        # Update relations knowledge base
        await self._update_relations_knowledge(causal_relations)
        
        # Build causal graph
        causal_graph = await self.graph_propagation_analyzer.build_causal_graph(
            all_factors, causal_relations
        )
        
        # Identify failure factors (factors present in the failure event)
        failure_factor_ids = [f.factor_id for f in event_factors]
        
        # Propagate to find root causes
        root_cause_scores = await self.graph_propagation_analyzer.propagate_root_causes(
            causal_graph, failure_factor_ids
        )
        
        # Rank and categorize causes
        ranked_causes = sorted(root_cause_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_causes = [cause_id for cause_id, score in ranked_causes[:3] if score > 0.7]
        contributing_factors = [cause_id for cause_id, score in ranked_causes if 0.3 <= score <= 0.7]
        
        # Identify inhibiting factors (factors that would have prevented the failure)
        inhibiting_factors = await self._identify_inhibiting_factors(
            causal_graph, failure_factor_ids, all_factors
        )
        
        # Find causal pathways
        causal_pathways = await self.graph_propagation_analyzer.identify_causal_pathways(
            causal_graph, primary_causes, failure_factor_ids
        )
        
        # Convert pathways to causal chain format
        causal_chain = []
        for pathway in causal_pathways[:5]:  # Top 5 pathways
            for i in range(len(pathway) - 1):
                cause_id, effect_id = pathway[i], pathway[i + 1]
                
                # Find relation strength
                relation = next((r for r in causal_relations 
                               if r.cause_factor_id == cause_id and r.effect_factor_id == effect_id), None)
                strength = relation.causal_strength if relation else 0.5
                
                causal_chain.append((cause_id, effect_id, strength))
        
        # Analyze graph properties
        graph_properties = await self.graph_propagation_analyzer.analyze_graph_properties(causal_graph)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            primary_causes, contributing_factors, inhibiting_factors, causal_graph
        )
        
        # Calculate confidence metrics
        overall_confidence = await self._calculate_analysis_confidence(
            primary_causes, causal_relations, len(all_events_data)
        )
        
        # Create analysis result
        analysis_end_time = datetime.utcnow()
        analysis_duration = (analysis_end_time - analysis_start_time).total_seconds()
        
        result = RootCauseAnalysisResult(
            analysis_id=f"rca_{analysis_start_time.strftime('%Y%m%d_%H%M%S_%f')}",
            failure_event_id=failure_event_id,
            failure_type=failure_type,
            timestamp=analysis_start_time,
            primary_causes=primary_causes,
            contributing_factors=contributing_factors,
            inhibiting_factors=inhibiting_factors,
            causal_chain=causal_chain,
            causal_graph_summary=graph_properties,
            overall_confidence=overall_confidence,
            explanation_completeness=len(primary_causes + contributing_factors) / max(len(all_factors), 1),
            preventive_actions=recommendations['preventive_actions'],
            monitoring_recommendations=recommendations['monitoring_recommendations'],
            system_improvements=recommendations['system_improvements'],
            analysis_duration_seconds=analysis_duration,
            factors_considered=len(all_factors),
            relationships_analyzed=len(causal_relations)
        )
        
        # Store analysis result
        self.analysis_history.append(result)
        if len(self.analysis_history) > self.max_history_size:
            self.analysis_history = self.analysis_history[-self.max_history_size:]
        
        # Update metrics
        await self._update_analysis_metrics(result)
        
        self.logger.info(f"Root cause analysis completed for {failure_event_id}: "
                        f"{len(primary_causes)} primary causes identified")
        
        return result
    
    async def _extract_factors_from_event(self, event: Dict[str, Any]) -> List[CausalFactor]:
        """Extract causal factors from an event"""
        
        factors = []
        event_timestamp = event.get('timestamp', datetime.utcnow())
        
        # System factors
        system_state = event.get('system_state', {})
        for key, value in system_state.items():
            factor = CausalFactor(
                factor_id=f"system_{key}",
                factor_name=f"System {key}",
                factor_type="system",
                importance_score=0.5,
                confidence_score=0.8,
                context_data={'value': value, 'source': 'system_state'},
                first_observed=event_timestamp,
                last_observed=event_timestamp
            )
            factors.append(factor)
        
        # Environmental factors
        environment = event.get('environment', {})
        for key, value in environment.items():
            factor = CausalFactor(
                factor_id=f"env_{key}",
                factor_name=f"Environment {key}",
                factor_type="environmental",
                importance_score=0.6,
                confidence_score=0.7,
                context_data={'value': value, 'source': 'environment'},
                first_observed=event_timestamp,
                last_observed=event_timestamp
            )
            factors.append(factor)
        
        # Configuration factors
        config = event.get('configuration', {})
        for key, value in config.items():
            factor = CausalFactor(
                factor_id=f"config_{key}",
                factor_name=f"Configuration {key}",
                factor_type="configuration",
                importance_score=0.7,
                confidence_score=0.9,
                context_data={'value': value, 'source': 'configuration'},
                first_observed=event_timestamp,
                last_observed=event_timestamp
            )
            factors.append(factor)
        
        # Data quality factors
        data_quality = event.get('data_quality', {})
        for key, value in data_quality.items():
            factor = CausalFactor(
                factor_id=f"data_{key}",
                factor_name=f"Data Quality {key}",
                factor_type="data",
                importance_score=0.8,
                confidence_score=0.8,
                context_data={'value': value, 'source': 'data_quality'},
                first_observed=event_timestamp,
                last_observed=event_timestamp
            )
            factors.append(factor)
        
        # Temporal factors
        temporal_context = event.get('temporal_context', {})
        for key, value in temporal_context.items():
            factor = CausalFactor(
                factor_id=f"temporal_{key}",
                factor_name=f"Temporal {key}",
                factor_type="temporal",
                importance_score=0.4,
                confidence_score=0.6,
                context_data={'value': value, 'source': 'temporal_context'},
                first_observed=event_timestamp,
                last_observed=event_timestamp
            )
            factors.append(factor)
        
        return factors
    
    async def _extract_factors_from_history(self, historical_data: List[Dict[str, Any]]) -> List[CausalFactor]:
        """Extract and aggregate factors from historical data"""
        
        factor_aggregates = defaultdict(list)
        
        # Collect all factor observations
        for event in historical_data:
            event_factors = await self._extract_factors_from_event(event)
            
            for factor in event_factors:
                factor_aggregates[factor.factor_id].append(factor)
        
        # Aggregate factors
        aggregated_factors = []
        
        for factor_id, factor_list in factor_aggregates.items():
            if not factor_list:
                continue
            
            # Use the first factor as template
            base_factor = factor_list[0]
            
            # Aggregate statistics
            importance_scores = [f.importance_score for f in factor_list]
            confidence_scores = [f.confidence_score for f in factor_list]
            
            aggregated_factor = CausalFactor(
                factor_id=factor_id,
                factor_name=base_factor.factor_name,
                factor_type=base_factor.factor_type,
                importance_score=np.mean(importance_scores),
                confidence_score=np.mean(confidence_scores),
                temporal_priority=1.0 / len(factor_list),  # Earlier = higher priority
                frequency_of_occurrence=len(factor_list),
                context_data=base_factor.context_data,
                first_observed=min(f.first_observed for f in factor_list),
                last_observed=max(f.last_observed for f in factor_list),
                observation_count=len(factor_list)
            )
            
            aggregated_factors.append(aggregated_factor)
        
        return aggregated_factors
    
    async def _update_factor_knowledge(self, factors: List[CausalFactor]):
        """Update the factor knowledge base"""
        
        for factor in factors:
            if factor.factor_id in self.factors:
                # Update existing factor
                existing = self.factors[factor.factor_id]
                existing.importance_score = (existing.importance_score + factor.importance_score) / 2
                existing.confidence_score = (existing.confidence_score + factor.confidence_score) / 2
                existing.frequency_of_occurrence += 1
                existing.last_observed = max(existing.last_observed, factor.last_observed)
                existing.observation_count += factor.observation_count
            else:
                # Add new factor
                self.factors[factor.factor_id] = factor
        
        # Maintain size limits
        if len(self.factors) > self.max_factors:
            # Remove least important factors
            sorted_factors = sorted(self.factors.items(), 
                                  key=lambda x: x[1].importance_score * x[1].confidence_score)
            factors_to_remove = len(self.factors) - self.max_factors
            
            for factor_id, _ in sorted_factors[:factors_to_remove]:
                del self.factors[factor_id]
    
    async def _update_relations_knowledge(self, relations: List[CausalRelation]):
        """Update the relations knowledge base"""
        
        for relation in relations:
            if relation.relation_id in self.relations:
                # Update existing relation
                existing = self.relations[relation.relation_id]
                existing.causal_strength = (existing.causal_strength + relation.causal_strength) / 2
                existing.confidence = (existing.confidence + relation.confidence) / 2
                existing.evidence_count += relation.evidence_count
                existing.last_observed = max(existing.last_observed, relation.last_observed)
            else:
                # Add new relation
                self.relations[relation.relation_id] = relation
        
        # Maintain size limits
        if len(self.relations) > self.max_relations:
            # Remove weakest relations
            sorted_relations = sorted(self.relations.items(),
                                    key=lambda x: x[1].causal_strength * x[1].confidence)
            relations_to_remove = len(self.relations) - self.max_relations
            
            for relation_id, _ in sorted_relations[:relations_to_remove]:
                del self.relations[relation_id]
    
    async def _identify_inhibiting_factors(self,
                                         causal_graph: nx.DiGraph,
                                         failure_factors: List[str],
                                         all_factors: List[CausalFactor]) -> List[str]:
        """Identify factors that could have prevented the failure"""
        
        inhibiting_factors = []
        
        # Look for factors that have negative causal relationships with failure factors
        for failure_factor in failure_factors:
            if failure_factor not in causal_graph:
                continue
            
            # Find factors that could inhibit this failure factor
            for factor in all_factors:
                if factor.factor_id == failure_factor:
                    continue
                
                # Check if there's a path that could inhibit the failure
                if factor.factor_id in causal_graph:
                    # Look for inhibiting relations
                    relation_key = f"consolidated_{factor.factor_id}_{failure_factor}"
                    if relation_key in self.relations:
                        relation = self.relations[relation_key]
                        if (relation.relation_type == CausalRelationType.INHIBITING_FACTOR and
                            relation.causal_strength > 0.5):
                            inhibiting_factors.append(factor.factor_id)
        
        # Also identify factors that were absent but typically present in successful cases
        # This would require historical success/failure data analysis
        
        return inhibiting_factors
    
    async def _generate_recommendations(self,
                                      primary_causes: List[str],
                                      contributing_factors: List[str],
                                      inhibiting_factors: List[str],
                                      causal_graph: nx.DiGraph) -> Dict[str, List[str]]:
        """Generate actionable recommendations based on root cause analysis"""
        
        recommendations = {
            'preventive_actions': [],
            'monitoring_recommendations': [],
            'system_improvements': []
        }
        
        # Preventive actions for primary causes
        for cause_id in primary_causes:
            if cause_id in self.factors:
                factor = self.factors[cause_id]
                
                if factor.factor_type == "system":
                    recommendations['preventive_actions'].append(
                        f"Monitor and maintain system component: {factor.factor_name}"
                    )
                elif factor.factor_type == "configuration":
                    recommendations['preventive_actions'].append(
                        f"Review and validate configuration: {factor.factor_name}"
                    )
                elif factor.factor_type == "data":
                    recommendations['preventive_actions'].append(
                        f"Implement data quality checks for: {factor.factor_name}"
                    )
        
        # Monitoring recommendations for contributing factors
        for factor_id in contributing_factors:
            if factor_id in self.factors:
                factor = self.factors[factor_id]
                recommendations['monitoring_recommendations'].append(
                    f"Add monitoring for: {factor.factor_name} (confidence: {factor.confidence_score:.2f})"
                )
        
        # System improvements based on inhibiting factors
        for factor_id in inhibiting_factors:
            if factor_id in self.factors:
                factor = self.factors[factor_id]
                recommendations['system_improvements'].append(
                    f"Strengthen protective mechanism: {factor.factor_name}"
                )
        
        # Graph-based recommendations
        graph_analysis = await self.graph_propagation_analyzer.analyze_graph_properties(causal_graph)
        
        # Identify high-centrality nodes for monitoring
        if 'betweenness_centrality' in graph_analysis:
            high_centrality_nodes = sorted(
                graph_analysis['betweenness_centrality'].items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            
            for node_id, centrality in high_centrality_nodes:
                if node_id in self.factors:
                    factor = self.factors[node_id]
                    recommendations['monitoring_recommendations'].append(
                        f"Critical monitoring point (high centrality): {factor.factor_name}"
                    )
        
        # Identify feedback loops for attention
        if graph_analysis.get('cycles'):
            recommendations['system_improvements'].append(
                f"Review {len(graph_analysis['cycles'])} feedback loops in the system"
            )
        
        return recommendations
    
    async def _calculate_analysis_confidence(self,
                                           primary_causes: List[str],
                                           causal_relations: List[CausalRelation],
                                           data_points: int) -> float:
        """Calculate overall confidence in the analysis"""
        
        confidence_factors = []
        
        # Confidence based on data volume
        data_confidence = min(1.0, data_points / 50.0)  # Full confidence with 50+ data points
        confidence_factors.append(data_confidence)
        
        # Confidence based on causal relation strength
        if causal_relations:
            avg_relation_confidence = np.mean([r.confidence for r in causal_relations])
            confidence_factors.append(avg_relation_confidence)
        
        # Confidence based on primary cause strength
        if primary_causes:
            primary_cause_factors = [self.factors[cause_id] for cause_id in primary_causes 
                                   if cause_id in self.factors]
            if primary_cause_factors:
                avg_cause_confidence = np.mean([f.confidence_score for f in primary_cause_factors])
                confidence_factors.append(avg_cause_confidence)
        
        # Overall confidence is the harmonic mean of individual confidences
        if confidence_factors:
            harmonic_mean = len(confidence_factors) / sum(1/c for c in confidence_factors if c > 0)
            return min(0.95, harmonic_mean)
        else:
            return 0.5
    
    async def _update_analysis_metrics(self, result: RootCauseAnalysisResult):
        """Update system metrics based on analysis result"""
        
        self.attribution_metrics['total_analyses_performed'] += 1
        
        # Update average analysis time
        total_analyses = self.attribution_metrics['total_analyses_performed']
        current_avg = self.attribution_metrics['average_analysis_time_seconds']
        new_avg = ((current_avg * (total_analyses - 1)) + result.analysis_duration_seconds) / total_analyses
        self.attribution_metrics['average_analysis_time_seconds'] = new_avg
        
        # Track common root causes
        for cause_id in result.primary_causes:
            if cause_id in self.factors:
                cause_name = self.factors[cause_id].factor_name
                self.attribution_metrics['most_common_root_causes'][cause_name] += 1
        
        # Update factors and relations discovered
        self.attribution_metrics['factors_identified'] = len(self.factors)
        self.attribution_metrics['relations_discovered'] = len(self.relations)
        
        # Track success rate by failure type (would need feedback mechanism)
        self.attribution_metrics['success_rate_by_failure_type'][result.failure_type.value].append(
            result.overall_confidence
        )
    
    async def get_attribution_status(self) -> Dict[str, Any]:
        """Get current status of the attribution system"""
        
        return {
            'system_active': True,
            'knowledge_base': {
                'factors_count': len(self.factors),
                'relations_count': len(self.relations),
                'analysis_history_size': len(self.analysis_history)
            },
            'metrics': self.attribution_metrics.copy(),
            'recent_analyses': len([
                analysis for analysis in self.analysis_history
                if (datetime.utcnow() - analysis.timestamp).total_seconds() < 3600
            ]),
            'top_root_causes': dict(self.attribution_metrics['most_common_root_causes'].most_common(10)),
            'average_confidence': np.mean([
                analysis.overall_confidence for analysis in self.analysis_history[-20:]
            ]) if self.analysis_history else 0.0
        }
    
    async def query_historical_patterns(self, 
                                      failure_type: FailureType = None,
                                      time_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """Query historical root cause patterns"""
        
        # Filter analysis history
        filtered_analyses = self.analysis_history
        
        if failure_type:
            filtered_analyses = [a for a in filtered_analyses if a.failure_type == failure_type]
        
        if time_range:
            start_time, end_time = time_range
            filtered_analyses = [a for a in filtered_analyses 
                               if start_time <= a.timestamp <= end_time]
        
        if not filtered_analyses:
            return {'status': 'no_data', 'filtered_count': 0}
        
        # Analyze patterns
        patterns = {
            'total_analyses': len(filtered_analyses),
            'average_confidence': np.mean([a.overall_confidence for a in filtered_analyses]),
            'most_common_primary_causes': Counter(),
            'most_common_contributing_factors': Counter(),
            'average_analysis_time': np.mean([a.analysis_duration_seconds for a in filtered_analyses]),
            'success_trends': []
        }
        
        # Count cause frequencies
        for analysis in filtered_analyses:
            for cause_id in analysis.primary_causes:
                if cause_id in self.factors:
                    cause_name = self.factors[cause_id].factor_name
                    patterns['most_common_primary_causes'][cause_name] += 1
            
            for factor_id in analysis.contributing_factors:
                if factor_id in self.factors:
                    factor_name = self.factors[factor_id].factor_name
                    patterns['most_common_contributing_factors'][factor_name] += 1
        
        # Convert counters to regular dicts
        patterns['most_common_primary_causes'] = dict(patterns['most_common_primary_causes'].most_common(10))
        patterns['most_common_contributing_factors'] = dict(patterns['most_common_contributing_factors'].most_common(10))
        
        return patterns


if __name__ == "__main__":
    async def main():
        # Example usage
        attribution_system = RootCauseAttributionSystem()
        
        # Create example failure event
        failure_event = {
            'event_id': 'failure_001',
            'failure_type': 'prediction_error',
            'timestamp': datetime.utcnow(),
            'system_state': {
                'cpu_usage': 0.95,
                'memory_usage': 0.87,
                'disk_io': 0.76
            },
            'environment': {
                'target_complexity': 0.8,
                'network_latency': 0.3
            },
            'configuration': {
                'model_threshold': 0.7,
                'timeout_seconds': 300
            },
            'data_quality': {
                'completeness': 0.85,
                'accuracy': 0.78
            },
            'outcome': 0.2  # Low success
        }
        
        # Create example historical data
        historical_data = []
        for i in range(20):
            event = {
                'event_id': f'event_{i}',
                'timestamp': datetime.utcnow() - timedelta(hours=i),
                'system_state': {
                    'cpu_usage': np.random.random(),
                    'memory_usage': np.random.random(),
                    'disk_io': np.random.random()
                },
                'environment': {
                    'target_complexity': np.random.random(),
                    'network_latency': np.random.random()
                },
                'outcome': np.random.random(),
                'factors': {
                    f'system_cpu_usage': {'value': np.random.random()},
                    f'env_target_complexity': {'value': np.random.random()}
                }
            }
            historical_data.append(event)
        
        # Perform root cause analysis
        result = await attribution_system.analyze_failure_event(failure_event, historical_data)
        
        print(f"Root Cause Analysis Result:")
        print(f"  Analysis ID: {result.analysis_id}")
        print(f"  Primary Causes: {result.primary_causes}")
        print(f"  Contributing Factors: {result.contributing_factors}")
        print(f"  Overall Confidence: {result.overall_confidence:.2f}")
        print(f"  Analysis Duration: {result.analysis_duration_seconds:.2f}s")
        print(f"  Preventive Actions: {len(result.preventive_actions)}")
        
        # Get system status
        status = await attribution_system.get_attribution_status()
        print(f"\nAttribution System Status:")
        print(f"  Factors: {status['knowledge_base']['factors_count']}")
        print(f"  Relations: {status['knowledge_base']['relations_count']}")
        print(f"  Average Confidence: {status['average_confidence']:.2f}")
        
        # Query historical patterns
        patterns = await attribution_system.query_historical_patterns()
        print(f"\nHistorical Patterns: {json.dumps(patterns, indent=2, default=str)}")
    
    asyncio.run(main())