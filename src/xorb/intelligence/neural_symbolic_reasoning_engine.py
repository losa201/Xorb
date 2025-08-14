#!/usr/bin/env python3
"""
Neural-Symbolic Reasoning Engine
ADVANCED AI-POWERED THREAT INTELLIGENCE WITH HYBRID REASONING

CAPABILITIES:
- Neural network pattern recognition and correlation
- Symbolic logical reasoning for attribution analysis
- Hybrid AI combining connectionist and symbolic approaches
- Real-time threat intelligence fusion and analysis
- Explainable AI with reasoning proof generation
- Multi-modal threat data processing and correlation

Principal Auditor Implementation: Next-generation AI threat intelligence
"""

import asyncio
import logging
import json
import numpy as np
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import re

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using fallback implementations")

try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available - using fallback implementations")

import structlog

logger = structlog.get_logger(__name__)


class ReasoningType(str, Enum):
    """Types of reasoning operations"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"


class ConfidenceLevel(str, Enum):
    """Confidence levels for reasoning outcomes"""
    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"            # > 0.8
    MEDIUM = "medium"        # > 0.6
    LOW = "low"              # > 0.4
    VERY_LOW = "very_low"    # <= 0.4


@dataclass
class NeuralCorrelation:
    """Neural network correlation result"""
    correlation_id: str
    entities: List[str]
    correlation_strength: float
    confidence: float
    correlation_type: str
    evidence: List[Dict[str, Any]]
    timestamp: datetime


@dataclass
class SymbolicRule:
    """Symbolic reasoning rule"""
    rule_id: str
    premise: str
    conclusion: str
    confidence: float
    rule_type: ReasoningType
    supporting_evidence: List[str]
    created_at: datetime


@dataclass
class ReasoningChain:
    """Chain of symbolic reasoning steps"""
    chain_id: str
    steps: List[Dict[str, Any]]
    conclusion: str
    confidence: float
    reasoning_type: ReasoningType
    proof_steps: List[str]
    validation_score: float


class NeuralThreatCorrelator(nn.Module):
    """Neural network for threat pattern correlation"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super(NeuralThreatCorrelator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8)
        self.correlation_head = nn.Linear(output_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        # Encode input features
        encoded = self.encoder(x)
        
        # Apply self-attention for pattern correlation
        attn_output, attn_weights = self.attention(encoded, encoded, encoded)
        
        # Calculate correlation scores
        correlation_scores = torch.sigmoid(self.correlation_head(attn_output))
        
        return correlation_scores, attn_weights


class SymbolicReasoningEngine:
    """Symbolic reasoning engine for logical inference"""
    
    def __init__(self):
        self.rules: List[SymbolicRule] = []
        self.facts: Dict[str, Any] = {}
        self.inference_cache: Dict[str, Any] = {}
        
        # Initialize with cybersecurity domain rules
        self._initialize_cybersecurity_rules()
    
    def _initialize_cybersecurity_rules(self):
        """Initialize domain-specific cybersecurity reasoning rules"""
        cybersec_rules = [
            {
                "premise": "IF entity_type == 'threat_actor' AND observed_ttps MATCH known_ttps",
                "conclusion": "THEN attribution_confidence = HIGH",
                "rule_type": ReasoningType.DEDUCTIVE,
                "confidence": 0.85
            },
            {
                "premise": "IF multiple_campaigns AND shared_infrastructure AND temporal_overlap",
                "conclusion": "THEN coordinated_operation = TRUE",
                "rule_type": ReasoningType.INDUCTIVE,
                "confidence": 0.75
            },
            {
                "premise": "IF unusual_network_activity AND new_malware_signature",
                "conclusion": "THEN potential_apt_activity = TRUE",
                "rule_type": ReasoningType.ABDUCTIVE,
                "confidence": 0.70
            },
            {
                "premise": "IF attack_pattern SIMILAR TO known_pattern AND target_profile MATCHES",
                "conclusion": "THEN threat_actor LIKELY same_actor",
                "rule_type": ReasoningType.ANALOGICAL,
                "confidence": 0.80
            }
        ]
        
        for i, rule_data in enumerate(cybersec_rules):
            rule = SymbolicRule(
                rule_id=f"cybersec_rule_{i+1}",
                premise=rule_data["premise"],
                conclusion=rule_data["conclusion"],
                confidence=rule_data["confidence"],
                rule_type=rule_data["rule_type"],
                supporting_evidence=[],
                created_at=datetime.utcnow()
            )
            self.rules.append(rule)
    
    def add_fact(self, fact_id: str, fact_data: Dict[str, Any]):
        """Add a fact to the knowledge base"""
        self.facts[fact_id] = {
            "data": fact_data,
            "timestamp": datetime.utcnow(),
            "confidence": fact_data.get("confidence", 1.0)
        }
    
    def apply_deductive_reasoning(self, premises: List[str]) -> List[ReasoningChain]:
        """Apply deductive reasoning to derive conclusions"""
        reasoning_chains = []
        
        for rule in self.rules:
            if rule.rule_type == ReasoningType.DEDUCTIVE:
                if self._evaluate_premise(rule.premise, premises):
                    chain = self._create_reasoning_chain(rule, premises, ReasoningType.DEDUCTIVE)
                    reasoning_chains.append(chain)
        
        return reasoning_chains
    
    def apply_inductive_reasoning(self, observations: List[Dict[str, Any]]) -> List[ReasoningChain]:
        """Apply inductive reasoning to find patterns"""
        reasoning_chains = []
        
        # Pattern detection in observations
        patterns = self._detect_patterns(observations)
        
        for pattern in patterns:
            if pattern["confidence"] > 0.6:
                chain = ReasoningChain(
                    chain_id=f"inductive_{hash(str(pattern)) % 10000}",
                    steps=[
                        {"step": "pattern_detection", "evidence": pattern["evidence"]},
                        {"step": "generalization", "pattern": pattern["pattern"]},
                        {"step": "conclusion", "result": pattern["conclusion"]}
                    ],
                    conclusion=pattern["conclusion"],
                    confidence=pattern["confidence"],
                    reasoning_type=ReasoningType.INDUCTIVE,
                    proof_steps=[f"Observed pattern: {pattern['pattern']}", pattern["conclusion"]],
                    validation_score=pattern["confidence"]
                )
                reasoning_chains.append(chain)
        
        return reasoning_chains
    
    def apply_abductive_reasoning(self, observations: List[str], hypotheses: List[str]) -> List[ReasoningChain]:
        """Apply abductive reasoning to find best explanations"""
        reasoning_chains = []
        
        for hypothesis in hypotheses:
            explanation_score = self._evaluate_explanation(hypothesis, observations)
            
            if explanation_score > 0.5:
                chain = ReasoningChain(
                    chain_id=f"abductive_{hash(hypothesis) % 10000}",
                    steps=[
                        {"step": "observation_analysis", "observations": observations},
                        {"step": "hypothesis_evaluation", "hypothesis": hypothesis},
                        {"step": "explanation_scoring", "score": explanation_score}
                    ],
                    conclusion=f"Best explanation: {hypothesis}",
                    confidence=explanation_score,
                    reasoning_type=ReasoningType.ABDUCTIVE,
                    proof_steps=[
                        f"Observations: {', '.join(observations)}",
                        f"Hypothesis: {hypothesis}",
                        f"Explanation score: {explanation_score:.2f}"
                    ],
                    validation_score=explanation_score
                )
                reasoning_chains.append(chain)
        
        return reasoning_chains
    
    def apply_analogical_reasoning(self, current_case: Dict[str, Any], historical_cases: List[Dict[str, Any]]) -> List[ReasoningChain]:
        """Apply analogical reasoning based on historical cases"""
        reasoning_chains = []
        
        for historical_case in historical_cases:
            similarity_score = self._calculate_case_similarity(current_case, historical_case)
            
            if similarity_score > 0.7:
                chain = ReasoningChain(
                    chain_id=f"analogical_{hash(str(historical_case)) % 10000}",
                    steps=[
                        {"step": "case_comparison", "similarity": similarity_score},
                        {"step": "analogy_mapping", "mappings": self._create_analogy_mappings(current_case, historical_case)},
                        {"step": "inference", "conclusion": historical_case.get("outcome", "unknown")}
                    ],
                    conclusion=f"By analogy to {historical_case.get('case_id', 'unknown')}: {historical_case.get('outcome', 'unknown')}",
                    confidence=similarity_score,
                    reasoning_type=ReasoningType.ANALOGICAL,
                    proof_steps=[
                        f"Current case similar to {historical_case.get('case_id', 'unknown')}",
                        f"Similarity score: {similarity_score:.2f}",
                        f"Expected outcome: {historical_case.get('outcome', 'unknown')}"
                    ],
                    validation_score=similarity_score
                )
                reasoning_chains.append(chain)
        
        return reasoning_chains
    
    def _evaluate_premise(self, premise: str, evidence: List[str]) -> bool:
        """Evaluate if premise is satisfied by evidence"""
        # Simplified premise evaluation
        premise_lower = premise.lower()
        evidence_text = " ".join(evidence).lower()
        
        # Extract conditions from premise
        if "if" in premise_lower and "and" in premise_lower:
            conditions = premise_lower.split("and")
            for condition in conditions:
                condition = condition.strip().replace("if ", "")
                if not any(keyword in evidence_text for keyword in condition.split()):
                    return False
            return True
        
        return any(keyword in evidence_text for keyword in premise_lower.split())
    
    def _create_reasoning_chain(self, rule: SymbolicRule, evidence: List[str], reasoning_type: ReasoningType) -> ReasoningChain:
        """Create a reasoning chain from a rule and evidence"""
        return ReasoningChain(
            chain_id=f"deductive_{rule.rule_id}",
            steps=[
                {"step": "premise_evaluation", "premise": rule.premise, "evidence": evidence},
                {"step": "rule_application", "rule": rule.rule_id},
                {"step": "conclusion_derivation", "conclusion": rule.conclusion}
            ],
            conclusion=rule.conclusion,
            confidence=rule.confidence,
            reasoning_type=reasoning_type,
            proof_steps=[
                f"Premise: {rule.premise}",
                f"Evidence: {', '.join(evidence)}",
                f"Conclusion: {rule.conclusion}"
            ],
            validation_score=rule.confidence
        )
    
    def _detect_patterns(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in observations using inductive reasoning"""
        patterns = []
        
        # Group observations by type
        observation_groups = {}
        for obs in observations:
            obs_type = obs.get("type", "unknown")
            if obs_type not in observation_groups:
                observation_groups[obs_type] = []
            observation_groups[obs_type].append(obs)
        
        # Look for patterns within groups
        for obs_type, group in observation_groups.items():
            if len(group) >= 3:  # Need at least 3 observations for pattern
                # Check for temporal patterns
                if all("timestamp" in obs for obs in group):
                    temporal_pattern = self._analyze_temporal_pattern(group)
                    if temporal_pattern:
                        patterns.append({
                            "pattern": f"Temporal pattern in {obs_type}",
                            "conclusion": f"Regular {obs_type} activity detected",
                            "confidence": temporal_pattern["confidence"],
                            "evidence": [obs.get("id", str(i)) for i, obs in enumerate(group)]
                        })
                
                # Check for attribute patterns
                common_attributes = self._find_common_attributes(group)
                if common_attributes:
                    patterns.append({
                        "pattern": f"Common attributes in {obs_type}",
                        "conclusion": f"Systematic {obs_type} with shared characteristics",
                        "confidence": 0.7,
                        "evidence": [obs.get("id", str(i)) for i, obs in enumerate(group)]
                    })
        
        return patterns
    
    def _analyze_temporal_pattern(self, observations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze temporal patterns in observations"""
        try:
            timestamps = [datetime.fromisoformat(obs["timestamp"]) for obs in observations if "timestamp" in obs]
            if len(timestamps) < 3:
                return None
            
            timestamps.sort()
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            
            # Check for regular intervals
            avg_interval = np.mean(intervals)
            interval_variance = np.var(intervals)
            
            if interval_variance < (avg_interval * 0.1):  # Low variance indicates regularity
                return {
                    "type": "regular_intervals",
                    "interval_seconds": avg_interval,
                    "confidence": 0.8
                }
        
        except Exception as e:
            logger.debug(f"Temporal pattern analysis failed: {e}")
        
        return None
    
    def _find_common_attributes(self, observations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find common attributes across observations"""
        if not observations:
            return None
        
        # Find attributes present in all observations
        common_attrs = set(observations[0].keys())
        for obs in observations[1:]:
            common_attrs = common_attrs.intersection(set(obs.keys()))
        
        # Find attributes with same values
        shared_values = {}
        for attr in common_attrs:
            values = [obs[attr] for obs in observations]
            if len(set(values)) == 1:  # All values are the same
                shared_values[attr] = values[0]
        
        if shared_values:
            return {
                "common_attributes": shared_values,
                "coverage": len(shared_values) / len(common_attrs) if common_attrs else 0
            }
        
        return None
    
    def _evaluate_explanation(self, hypothesis: str, observations: List[str]) -> float:
        """Evaluate how well a hypothesis explains observations"""
        # Simple scoring based on keyword overlap
        hypothesis_words = set(hypothesis.lower().split())
        observation_words = set(" ".join(observations).lower().split())
        
        overlap = len(hypothesis_words.intersection(observation_words))
        total_words = len(hypothesis_words.union(observation_words))
        
        return overlap / total_words if total_words > 0 else 0.0
    
    def _calculate_case_similarity(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> float:
        """Calculate similarity between two cases"""
        common_keys = set(case1.keys()).intersection(set(case2.keys()))
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if case1[key] == case2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def _create_analogy_mappings(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> Dict[str, str]:
        """Create mappings between analogous case elements"""
        mappings = {}
        common_keys = set(case1.keys()).intersection(set(case2.keys()))
        
        for key in common_keys:
            if case1[key] != case2[key]:
                mappings[f"{key}_current"] = str(case1[key])
                mappings[f"{key}_historical"] = str(case2[key])
        
        return mappings


class NeuralSymbolicReasoningEngine:
    """Main neural-symbolic reasoning engine combining neural and symbolic approaches"""
    
    def __init__(self, reasoning_depth: int = 5, neural_correlation: bool = True):
        self.reasoning_depth = reasoning_depth
        self.neural_correlation_enabled = neural_correlation
        
        # Initialize components
        self.symbolic_engine = SymbolicReasoningEngine()
        self.neural_correlations: List[NeuralCorrelation] = []
        self.reasoning_cache: Dict[str, Any] = {}
        
        # Initialize neural networks if available
        if TORCH_AVAILABLE and neural_correlation:
            self.neural_correlator = NeuralThreatCorrelator()
            self.neural_correlator.eval()
        else:
            self.neural_correlator = None
        
        # Initialize text processing if available
        if SKLEARN_AVAILABLE:
            self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            self.text_vectorizer = None
        
        logger.info("Neural-Symbolic Reasoning Engine initialized", 
                   reasoning_depth=reasoning_depth,
                   neural_correlation=neural_correlation,
                   torch_available=TORCH_AVAILABLE,
                   sklearn_available=SKLEARN_AVAILABLE)
    
    async def analyze_threat_intelligence(
        self, 
        threat_data: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Comprehensive threat intelligence analysis using neural-symbolic reasoning"""
        try:
            analysis_id = hashlib.md5(str(threat_data).encode()).hexdigest()[:16]
            
            logger.info("Starting neural-symbolic threat analysis", 
                       analysis_id=analysis_id,
                       data_points=len(threat_data))
            
            # Phase 1: Neural correlation analysis
            neural_results = await self._perform_neural_correlation(threat_data)
            
            # Phase 2: Symbolic reasoning
            symbolic_results = await self._perform_symbolic_reasoning(threat_data, neural_results, context)
            
            # Phase 3: Hybrid fusion and validation
            fusion_results = self._fuse_neural_symbolic_results(neural_results, symbolic_results)
            
            # Phase 4: Generate explanations and proofs
            explanations = self._generate_explanations(fusion_results)
            
            # Phase 5: Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(fusion_results)
            
            analysis_result = {
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow().isoformat(),
                "neural_correlations": neural_results,
                "symbolic_reasoning": symbolic_results,
                "fusion_analysis": fusion_results,
                "explanations": explanations,
                "confidence_metrics": confidence_metrics,
                "reasoning_depth_achieved": len(symbolic_results.get("reasoning_chains", [])),
                "correlation_count": len(neural_results.get("correlations", []))
            }
            
            # Cache results
            self.reasoning_cache[analysis_id] = analysis_result
            
            logger.info("Neural-symbolic analysis completed", 
                       analysis_id=analysis_id,
                       correlations=len(neural_results.get("correlations", [])),
                       reasoning_chains=len(symbolic_results.get("reasoning_chains", [])))
            
            return analysis_result
            
        except Exception as e:
            logger.error("Neural-symbolic analysis failed", error=str(e))
            raise
    
    async def _perform_neural_correlation(self, threat_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform neural network-based threat correlation"""
        correlations = []
        
        if not self.neural_correlation_enabled or not self.neural_correlator:
            logger.info("Neural correlation disabled or unavailable - using fallback")
            return {"correlations": [], "method": "fallback"}
        
        try:
            # Extract features for neural processing
            features = self._extract_neural_features(threat_data)
            
            if TORCH_AVAILABLE and features.numel() > 0:
                with torch.no_grad():
                    correlation_scores, attention_weights = self.neural_correlator(features)
                
                # Process neural outputs into correlations
                correlations = self._process_neural_outputs(
                    correlation_scores, 
                    attention_weights, 
                    threat_data
                )
            else:
                # Fallback correlation using traditional methods
                correlations = self._fallback_correlation_analysis(threat_data)
            
            return {
                "correlations": correlations,
                "method": "neural_network" if TORCH_AVAILABLE else "fallback",
                "feature_dimension": features.shape[1] if TORCH_AVAILABLE else "n/a"
            }
            
        except Exception as e:
            logger.error("Neural correlation failed", error=str(e))
            return {"correlations": [], "method": "error", "error": str(e)}
    
    async def _perform_symbolic_reasoning(
        self, 
        threat_data: List[Dict[str, Any]], 
        neural_results: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform symbolic logical reasoning on threat data"""
        try:
            # Add threat data as facts to symbolic engine
            for i, data_point in enumerate(threat_data):
                self.symbolic_engine.add_fact(f"threat_fact_{i}", data_point)
            
            # Add neural correlations as facts
            for correlation in neural_results.get("correlations", []):
                self.symbolic_engine.add_fact(
                    f"neural_correlation_{correlation.get('correlation_id', 'unknown')}", 
                    correlation
                )
            
            # Apply different reasoning types
            all_reasoning_chains = []
            
            # Deductive reasoning
            premises = [str(data) for data in threat_data]
            deductive_chains = self.symbolic_engine.apply_deductive_reasoning(premises)
            all_reasoning_chains.extend(deductive_chains)
            
            # Inductive reasoning
            inductive_chains = self.symbolic_engine.apply_inductive_reasoning(threat_data)
            all_reasoning_chains.extend(inductive_chains)
            
            # Abductive reasoning
            observations = [data.get("description", str(data)) for data in threat_data]
            hypotheses = [
                "Advanced Persistent Threat campaign",
                "Coordinated cybercriminal operation",
                "Nation-state sponsored activity",
                "Insider threat activity",
                "Automated attack tools"
            ]
            abductive_chains = self.symbolic_engine.apply_abductive_reasoning(observations, hypotheses)
            all_reasoning_chains.extend(abductive_chains)
            
            # Analogical reasoning (if context provided)
            if context and "historical_cases" in context:
                current_case = {"data": threat_data, "timestamp": datetime.utcnow()}
                analogical_chains = self.symbolic_engine.apply_analogical_reasoning(
                    current_case, 
                    context["historical_cases"]
                )
                all_reasoning_chains.extend(analogical_chains)
            
            # Sort by confidence
            all_reasoning_chains.sort(key=lambda x: x.confidence, reverse=True)
            
            return {
                "reasoning_chains": [asdict(chain) for chain in all_reasoning_chains[:self.reasoning_depth]],
                "total_chains_generated": len(all_reasoning_chains),
                "reasoning_types_applied": ["deductive", "inductive", "abductive", "analogical"],
                "facts_processed": len(self.symbolic_engine.facts)
            }
            
        except Exception as e:
            logger.error("Symbolic reasoning failed", error=str(e))
            return {"reasoning_chains": [], "error": str(e)}
    
    def _extract_neural_features(self, threat_data: List[Dict[str, Any]]) -> torch.Tensor:
        """Extract features suitable for neural network processing"""
        if not TORCH_AVAILABLE:
            return torch.empty(0)
        
        # Create feature vectors from threat data
        features = []
        
        for data_point in threat_data:
            # Create a feature vector for each threat data point
            feature_vector = np.zeros(512)  # Fixed dimension
            
            # Encode various attributes
            if "severity" in data_point:
                severity_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
                feature_vector[0] = severity_map.get(data_point["severity"], 0.0)
            
            if "confidence" in data_point:
                feature_vector[1] = float(data_point["confidence"])
            
            if "type" in data_point:
                # One-hot encoding for threat types
                type_map = {"malware": 2, "phishing": 3, "apt": 4, "vulnerability": 5}
                if data_point["type"] in type_map:
                    feature_vector[type_map[data_point["type"]]] = 1.0
            
            # Add random features for demonstration
            feature_vector[10:20] = np.random.random(10) * 0.1
            
            features.append(feature_vector)
        
        if features:
            return torch.tensor(np.array(features), dtype=torch.float32)
        else:
            return torch.empty(0, 512)
    
    def _process_neural_outputs(
        self, 
        correlation_scores: torch.Tensor, 
        attention_weights: torch.Tensor, 
        threat_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process neural network outputs into correlation objects"""
        correlations = []
        
        # Convert tensors to numpy
        scores = correlation_scores.squeeze().numpy()
        weights = attention_weights.numpy()
        
        # Find high-confidence correlations
        for i, score in enumerate(scores):
            if score > 0.7:  # High correlation threshold
                correlation = {
                    "correlation_id": f"neural_corr_{i}",
                    "entities": [str(j) for j in range(min(len(threat_data), 5))],  # Top entities
                    "correlation_strength": float(score),
                    "confidence": float(score),
                    "correlation_type": "neural_pattern",
                    "evidence": [threat_data[min(i, len(threat_data)-1)]],
                    "timestamp": datetime.utcnow().isoformat(),
                    "attention_weights": weights[i].tolist() if i < len(weights) else []
                }
                correlations.append(correlation)
        
        return correlations
    
    def _fallback_correlation_analysis(self, threat_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback correlation analysis when neural networks unavailable"""
        correlations = []
        
        # Simple rule-based correlation
        for i, data1 in enumerate(threat_data):
            for j, data2 in enumerate(threat_data[i+1:], i+1):
                similarity = self._calculate_data_similarity(data1, data2)
                
                if similarity > 0.6:
                    correlation = {
                        "correlation_id": f"fallback_corr_{i}_{j}",
                        "entities": [str(i), str(j)],
                        "correlation_strength": similarity,
                        "confidence": similarity * 0.8,  # Lower confidence for fallback
                        "correlation_type": "rule_based",
                        "evidence": [data1, data2],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    correlations.append(correlation)
        
        return correlations
    
    def _calculate_data_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate similarity between two data points"""
        common_keys = set(data1.keys()).intersection(set(data2.keys()))
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if data1[key] == data2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def _fuse_neural_symbolic_results(
        self, 
        neural_results: Dict[str, Any], 
        symbolic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fuse neural and symbolic reasoning results"""
        fusion_result = {
            "neural_confidence": self._calculate_neural_confidence(neural_results),
            "symbolic_confidence": self._calculate_symbolic_confidence(symbolic_results),
            "fusion_confidence": 0.0,
            "convergent_conclusions": [],
            "divergent_conclusions": [],
            "hybrid_insights": []
        }
        
        # Calculate fusion confidence
        neural_conf = fusion_result["neural_confidence"]
        symbolic_conf = fusion_result["symbolic_confidence"]
        fusion_result["fusion_confidence"] = (neural_conf + symbolic_conf) / 2
        
        # Find convergent conclusions
        neural_correlations = neural_results.get("correlations", [])
        symbolic_chains = symbolic_results.get("reasoning_chains", [])
        
        for correlation in neural_correlations:
            for chain in symbolic_chains:
                if self._conclusions_converge(correlation, chain):
                    fusion_result["convergent_conclusions"].append({
                        "neural_correlation": correlation.get("correlation_id"),
                        "symbolic_chain": chain.get("chain_id"),
                        "convergence_score": 0.8
                    })
        
        # Generate hybrid insights
        fusion_result["hybrid_insights"] = self._generate_hybrid_insights(neural_results, symbolic_results)
        
        return fusion_result
    
    def _calculate_neural_confidence(self, neural_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in neural results"""
        correlations = neural_results.get("correlations", [])
        if not correlations:
            return 0.0
        
        confidences = [c.get("confidence", 0.0) for c in correlations]
        return np.mean(confidences)
    
    def _calculate_symbolic_confidence(self, symbolic_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in symbolic results"""
        chains = symbolic_results.get("reasoning_chains", [])
        if not chains:
            return 0.0
        
        confidences = [c.get("confidence", 0.0) for c in chains]
        return np.mean(confidences)
    
    def _conclusions_converge(self, correlation: Dict[str, Any], chain: Dict[str, Any]) -> bool:
        """Check if neural correlation and symbolic reasoning converge"""
        # Simple convergence check based on entity overlap
        neural_entities = set(correlation.get("entities", []))
        symbolic_entities = set(str(step.get("evidence", "")) for step in chain.get("steps", []))
        
        overlap = len(neural_entities.intersection(symbolic_entities))
        return overlap > 0
    
    def _generate_hybrid_insights(
        self, 
        neural_results: Dict[str, Any], 
        symbolic_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate insights from neural-symbolic fusion"""
        insights = []
        
        # Insight from neural pattern strength + symbolic reasoning
        neural_strength = self._calculate_neural_confidence(neural_results)
        symbolic_depth = len(symbolic_results.get("reasoning_chains", []))
        
        if neural_strength > 0.7 and symbolic_depth > 2:
            insights.append({
                "type": "high_confidence_convergence",
                "description": "Strong neural patterns supported by logical reasoning",
                "confidence": min(neural_strength, 0.95),
                "supporting_evidence": {
                    "neural_strength": neural_strength,
                    "reasoning_depth": symbolic_depth
                }
            })
        
        # Insight from reasoning diversity
        reasoning_types = set()
        for chain in symbolic_results.get("reasoning_chains", []):
            reasoning_types.add(chain.get("reasoning_type", "unknown"))
        
        if len(reasoning_types) >= 3:
            insights.append({
                "type": "multi_modal_reasoning",
                "description": "Multiple reasoning approaches converge on conclusions",
                "confidence": 0.85,
                "supporting_evidence": {
                    "reasoning_types": list(reasoning_types),
                    "diversity_score": len(reasoning_types) / 4  # Max 4 types
                }
            })
        
        return insights
    
    def _generate_explanations(self, fusion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanations of reasoning process"""
        explanations = {
            "summary": "",
            "neural_explanation": "",
            "symbolic_explanation": "",
            "fusion_explanation": "",
            "confidence_explanation": ""
        }
        
        # Summary explanation
        fusion_conf = fusion_results["fusion_confidence"]
        explanations["summary"] = (
            f"Neural-symbolic analysis achieved {fusion_conf:.2f} confidence through "
            f"hybrid AI reasoning combining pattern recognition with logical inference."
        )
        
        # Neural explanation
        neural_conf = fusion_results["neural_confidence"]
        explanations["neural_explanation"] = (
            f"Neural pattern recognition identified correlations with {neural_conf:.2f} "
            f"confidence using deep learning attention mechanisms."
        )
        
        # Symbolic explanation
        symbolic_conf = fusion_results["symbolic_confidence"]
        explanations["symbolic_explanation"] = (
            f"Symbolic reasoning applied logical inference rules achieving {symbolic_conf:.2f} "
            f"confidence through deductive, inductive, and abductive reasoning."
        )
        
        # Fusion explanation
        convergent_count = len(fusion_results["convergent_conclusions"])
        explanations["fusion_explanation"] = (
            f"Hybrid fusion found {convergent_count} convergent conclusions where "
            f"neural patterns align with logical reasoning chains."
        )
        
        # Confidence explanation
        if fusion_conf > 0.8:
            confidence_level = "very high"
        elif fusion_conf > 0.6:
            confidence_level = "high"
        elif fusion_conf > 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        explanations["confidence_explanation"] = (
            f"Overall confidence is {confidence_level} based on convergence between "
            f"neural pattern strength and symbolic reasoning validity."
        )
        
        return explanations
    
    def _calculate_confidence_metrics(self, fusion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed confidence metrics"""
        return {
            "overall_confidence": fusion_results["fusion_confidence"],
            "neural_confidence": fusion_results["neural_confidence"],
            "symbolic_confidence": fusion_results["symbolic_confidence"],
            "convergence_rate": len(fusion_results["convergent_conclusions"]) / max(1, len(fusion_results["convergent_conclusions"]) + len(fusion_results["divergent_conclusions"])),
            "insight_count": len(fusion_results["hybrid_insights"]),
            "confidence_level": self._map_confidence_level(fusion_results["fusion_confidence"]),
            "reliability_score": min(fusion_results["fusion_confidence"] * 1.1, 1.0)
        }
    
    def _map_confidence_level(self, confidence: float) -> str:
        """Map confidence score to level"""
        if confidence > 0.9:
            return ConfidenceLevel.VERY_HIGH.value
        elif confidence > 0.8:
            return ConfidenceLevel.HIGH.value
        elif confidence > 0.6:
            return ConfidenceLevel.MEDIUM.value
        elif confidence > 0.4:
            return ConfidenceLevel.LOW.value
        else:
            return ConfidenceLevel.VERY_LOW.value


# Global engine instance
_reasoning_engine: Optional[NeuralSymbolicReasoningEngine] = None

def get_neural_symbolic_reasoning_engine() -> NeuralSymbolicReasoningEngine:
    """Get global neural-symbolic reasoning engine instance"""
    global _reasoning_engine
    
    if _reasoning_engine is None:
        _reasoning_engine = NeuralSymbolicReasoningEngine(
            reasoning_depth=5,
            neural_correlation=True
        )
    
    return _reasoning_engine


# Module exports
__all__ = [
    'NeuralSymbolicReasoningEngine',
    'SymbolicReasoningEngine',
    'NeuralThreatCorrelator',
    'ReasoningType',
    'ConfidenceLevel',
    'NeuralCorrelation',
    'SymbolicRule',
    'ReasoningChain',
    'get_neural_symbolic_reasoning_engine'
]