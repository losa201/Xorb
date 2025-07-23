#!/usr/bin/env python3
"""
Graph-Based Attack Path Prediction for Xorb 2.0

This module implements advanced graph algorithms to predict attack paths,
identify high-value targets, and optimize agent deployment based on network
topology and vulnerability relationships. Uses Neo4j for graph storage and
NetworkX for complex graph analysis.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import networkx as nx

try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available. Install with: pip install neo4j")

try:
    import numpy as np
    from sklearn.cluster import DBSCAN, SpectralClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available for advanced graph analytics")

from .atom import KnowledgeAtom, AtomType
from .models import AtomModel


class PathType(Enum):
    """Types of attack paths"""
    DIRECT = "direct"
    LATERAL = "lateral"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class NodeType(Enum):
    """Types of nodes in attack graph"""
    VULNERABILITY = "vulnerability"
    EXPLOIT = "exploit"
    ASSET = "asset"
    CREDENTIAL = "credential"
    TECHNIQUE = "technique"
    TOOL = "tool"
    INDICATOR = "indicator"


@dataclass
class AttackNode:
    """Node in attack graph"""
    node_id: str
    node_type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    severity: float = 0.5  # 0.0 to 1.0
    exploitability: float = 0.5  # 0.0 to 1.0
    impact_score: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AttackEdge:
    """Edge in attack graph"""
    source_id: str
    target_id: str
    relationship_type: str
    probability: float = 0.5  # Probability of successful transition
    difficulty: float = 0.5  # Difficulty of exploitation (0.0 = easy, 1.0 = hard)
    time_estimate_hours: float = 1.0
    prerequisites: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    evidence_strength: float = 0.5


@dataclass
class AttackPath:
    """Complete attack path"""
    path_id: str
    nodes: List[AttackNode]
    edges: List[AttackEdge]
    path_type: PathType
    total_probability: float
    total_difficulty: float
    estimated_time_hours: float
    criticality_score: float
    detection_likelihood: float
    mitigation_coverage: float
    path_length: int
    entry_points: List[str]
    target_assets: List[str]


@dataclass
class GraphMetrics:
    """Graph analysis metrics"""
    node_count: int
    edge_count: int
    avg_clustering_coefficient: float
    avg_path_length: float
    network_diameter: int
    centrality_analysis: Dict[str, Dict[str, float]]
    community_structure: Dict[str, List[str]]
    vulnerability_density: float
    exploit_coverage: float


class GraphAttackPredictor:
    """
    Advanced graph-based attack path prediction and analysis
    """
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        self.logger = logging.getLogger(__name__)
        
        # Graph storage
        self.neo4j_driver = None
        self.memory_graph = nx.MultiDiGraph()  # In-memory graph for fast analysis
        
        # Analysis caches
        self.path_cache: Dict[str, List[AttackPath]] = {}
        self.centrality_cache: Dict[str, Dict[str, float]] = {}
        self.community_cache: Dict[str, List[str]] = {}
        
        # Machine learning models for path scoring
        self.path_classifier = None
        self.risk_predictor = None
        
        # Configuration
        self.max_path_length = 10
        self.max_paths_per_query = 50
        self.cache_ttl_minutes = 30
        
        # Performance metrics
        self.prediction_metrics = {
            'total_paths_analyzed': 0,
            'successful_predictions': 0,
            'false_positives': 0,
            'analysis_time_avg_ms': 0.0,
            'cache_hit_ratio': 0.0
        }
        
        self.logger.info("Graph Attack Predictor initialized")
    
    async def initialize(self):
        """Initialize Neo4j connection and graph structures"""
        
        if NEO4J_AVAILABLE:
            try:
                self.neo4j_driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                
                # Test connection
                async with self.neo4j_driver.session() as session:
                    result = await session.run("RETURN 1 as test")
                    await result.consume()
                
                self.logger.info("Connected to Neo4j for graph storage")
                
                # Initialize graph schema
                await self._initialize_graph_schema()
                
            except Exception as e:
                self.logger.warning(f"Neo4j connection failed: {e}. Using in-memory graph only.")
                self.neo4j_driver = None
        
        # Initialize ML models if available
        if ML_AVAILABLE:
            await self._initialize_ml_models()
        
        self.logger.info("Graph Attack Predictor initialization complete")
    
    async def _initialize_graph_schema(self):
        """Initialize Neo4j graph schema and constraints"""
        
        if not self.neo4j_driver:
            return
        
        schema_queries = [
            # Create constraints
            "CREATE CONSTRAINT asset_id IF NOT EXISTS FOR (a:Asset) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT vulnerability_id IF NOT EXISTS FOR (v:Vulnerability) REQUIRE v.id IS UNIQUE", 
            "CREATE CONSTRAINT exploit_id IF NOT EXISTS FOR (e:Exploit) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT technique_id IF NOT EXISTS FOR (t:Technique) REQUIRE t.id IS UNIQUE",
            
            # Create indexes for performance
            "CREATE INDEX asset_severity IF NOT EXISTS FOR (a:Asset) ON (a.severity)",
            "CREATE INDEX vulnerability_cvss IF NOT EXISTS FOR (v:Vulnerability) ON (v.cvss_score)",
            "CREATE INDEX exploit_difficulty IF NOT EXISTS FOR (e:Exploit) ON (e.difficulty)",
            "CREATE INDEX technique_impact IF NOT EXISTS FOR (t:Technique) ON (t.impact_score)",
            
            # Create compound indexes for common queries
            "CREATE INDEX path_analysis IF NOT EXISTS FOR (n) ON (n.severity, n.exploitability, n.confidence)"
        ]
        
        async with self.neo4j_driver.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    self.logger.debug(f"Schema query failed (may already exist): {e}")
        
        self.logger.info("Graph schema initialized")
    
    async def _initialize_ml_models(self):
        """Initialize machine learning models for enhanced prediction"""
        
        if not ML_AVAILABLE:
            return
        
        # In production, these would be pre-trained models
        # For now, we'll initialize placeholder models
        
        self.logger.info("ML models initialized for graph analysis")
    
    async def build_attack_graph(self, knowledge_atoms: List[KnowledgeAtom]) -> int:
        """
        Build attack graph from knowledge atoms
        
        Args:
            knowledge_atoms: List of knowledge atoms to process
            
        Returns:
            Number of nodes added to graph
        """
        
        nodes_added = 0
        edges_added = 0
        
        # Process atoms by type
        vulnerability_atoms = [a for a in knowledge_atoms if a.atom_type == AtomType.VULNERABILITY]
        exploit_atoms = [a for a in knowledge_atoms if a.atom_type == AtomType.EXPLOIT]
        intelligence_atoms = [a for a in knowledge_atoms if a.atom_type == AtomType.INTELLIGENCE]
        target_atoms = [a for a in knowledge_atoms if a.atom_type == AtomType.TARGET]
        
        # Add vulnerability nodes
        for atom in vulnerability_atoms:
            node = await self._atom_to_attack_node(atom, NodeType.VULNERABILITY)
            await self._add_node_to_graph(node)
            nodes_added += 1
        
        # Add exploit nodes
        for atom in exploit_atoms:
            node = await self._atom_to_attack_node(atom, NodeType.EXPLOIT)
            await self._add_node_to_graph(node)
            nodes_added += 1
        
        # Add asset nodes from targets
        for atom in target_atoms:
            node = await self._atom_to_attack_node(atom, NodeType.ASSET)
            await self._add_node_to_graph(node)
            nodes_added += 1
        
        # Add technique nodes from intelligence
        for atom in intelligence_atoms:
            node = await self._atom_to_attack_node(atom, NodeType.TECHNIQUE)
            await self._add_node_to_graph(node)
            nodes_added += 1
        
        # Build relationships between nodes
        edges_added = await self._build_graph_relationships(knowledge_atoms)
        
        # Update centrality and community caches
        await self._update_graph_analysis_caches()
        
        self.logger.info(f"Built attack graph: {nodes_added} nodes, {edges_added} edges")
        return nodes_added
    
    async def _atom_to_attack_node(self, atom: KnowledgeAtom, node_type: NodeType) -> AttackNode:
        """Convert knowledge atom to attack graph node"""
        
        # Extract properties based on atom content
        properties = atom.content.copy() if isinstance(atom.content, dict) else {}
        
        # Calculate severity based on atom type and content
        severity = atom.confidence  # Base severity from confidence
        if 'cvss_score' in properties:
            severity = max(severity, properties['cvss_score'] / 10.0)
        elif 'severity' in properties:
            severity_map = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 1.0}
            severity = max(severity, severity_map.get(properties['severity'].lower(), 0.5))
        
        # Calculate exploitability
        exploitability = 0.5  # Default
        if node_type == NodeType.VULNERABILITY:
            # Higher CVSS score = higher exploitability
            if 'cvss_score' in properties:
                exploitability = min(1.0, properties['cvss_score'] / 8.0)  # Scale to 0-1
        elif node_type == NodeType.EXPLOIT:
            # Exploit availability = high exploitability
            exploitability = 0.8
        
        # Calculate impact score
        impact_score = severity  # Default to severity
        if 'impact_score' in properties:
            impact_score = properties['impact_score']
        elif node_type == NodeType.ASSET:
            # Asset criticality affects impact
            criticality = properties.get('criticality', 'medium')
            criticality_map = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 1.0}
            impact_score = criticality_map.get(criticality.lower(), 0.6)
        
        return AttackNode(
            node_id=atom.id,
            node_type=node_type,
            name=atom.title,
            properties=properties,
            severity=severity,
            exploitability=exploitability,
            impact_score=impact_score,
            confidence=atom.confidence,
            last_updated=atom.updated_at or atom.created_at
        )
    
    async def _add_node_to_graph(self, node: AttackNode):
        """Add node to both Neo4j and in-memory graphs"""
        
        # Add to in-memory graph
        self.memory_graph.add_node(
            node.node_id,
            node_type=node.node_type.value,
            name=node.name,
            severity=node.severity,
            exploitability=node.exploitability,
            impact_score=node.impact_score,
            confidence=node.confidence,
            **node.properties
        )
        
        # Add to Neo4j if available
        if self.neo4j_driver:
            await self._add_node_to_neo4j(node)
    
    async def _add_node_to_neo4j(self, node: AttackNode):
        """Add node to Neo4j database"""
        
        query = f"""
        MERGE (n:{node.node_type.value.title()} {{id: $node_id}})
        SET n.name = $name,
            n.severity = $severity,
            n.exploitability = $exploitability,
            n.impact_score = $impact_score,
            n.confidence = $confidence,
            n.last_updated = datetime($last_updated),
            n.properties = $properties
        RETURN n.id
        """
        
        async with self.neo4j_driver.session() as session:
            await session.run(query, {
                'node_id': node.node_id,
                'name': node.name,
                'severity': node.severity,
                'exploitability': node.exploitability,
                'impact_score': node.impact_score,
                'confidence': node.confidence,
                'last_updated': node.last_updated.isoformat(),
                'properties': node.properties
            })
    
    async def _build_graph_relationships(self, knowledge_atoms: List[KnowledgeAtom]) -> int:
        """Build relationships between graph nodes"""
        
        edges_added = 0
        
        # Build relationships based on atom relationships
        for atom in knowledge_atoms:
            if not atom.related_atoms:
                continue
            
            for related in atom.related_atoms:
                # Create edge based on relationship type
                edge = AttackEdge(
                    source_id=atom.id,
                    target_id=related.target_atom_id,
                    relationship_type=related.relationship_type,
                    probability=related.strength,
                    difficulty=1.0 - related.strength,  # Inverse of strength
                    time_estimate_hours=self._estimate_transition_time(related.relationship_type),
                    evidence_strength=related.strength
                )
                
                await self._add_edge_to_graph(edge)
                edges_added += 1
        
        # Infer additional relationships based on content analysis
        edges_added += await self._infer_attack_relationships(knowledge_atoms)
        
        return edges_added
    
    def _estimate_transition_time(self, relationship_type: str) -> float:
        """Estimate time required for attack transition"""
        
        time_estimates = {
            'exploits': 2.0,      # 2 hours to exploit vulnerability
            'enables': 1.0,       # 1 hour to enable technique
            'requires': 0.5,      # 30 minutes to gather prerequisites
            'leads_to': 3.0,      # 3 hours for lateral movement
            'escalates_to': 4.0,  # 4 hours for privilege escalation
            'persists_via': 1.5,  # 1.5 hours to establish persistence
            'exfiltrates_via': 2.5, # 2.5 hours for data exfiltration
            'similar_to': 0.1     # Very fast for similar techniques
        }
        
        return time_estimates.get(relationship_type, 2.0)
    
    async def _infer_attack_relationships(self, knowledge_atoms: List[KnowledgeAtom]) -> int:
        """Infer additional attack relationships using heuristics"""
        
        edges_added = 0
        
        # Group atoms by type for relationship inference
        vulns = {a.id: a for a in knowledge_atoms if a.atom_type == AtomType.VULNERABILITY}
        exploits = {a.id: a for a in knowledge_atoms if a.atom_type == AtomType.EXPLOIT}
        targets = {a.id: a for a in knowledge_atoms if a.atom_type == AtomType.TARGET}
        
        # Infer vulnerability -> exploit relationships
        for vuln_id, vuln in vulns.items():
            for exploit_id, exploit in exploits.items():
                if await self._should_link_vuln_exploit(vuln, exploit):
                    probability = await self._calculate_exploit_probability(vuln, exploit)
                    
                    edge = AttackEdge(
                        source_id=vuln_id,
                        target_id=exploit_id,
                        relationship_type='exploitable_by',
                        probability=probability,
                        difficulty=1.0 - probability,
                        time_estimate_hours=2.0,
                        evidence_strength=0.7  # Inferred relationship
                    )
                    
                    await self._add_edge_to_graph(edge)
                    edges_added += 1
        
        # Infer exploit -> target relationships
        for exploit_id, exploit in exploits.items():
            for target_id, target in targets.items():
                if await self._should_link_exploit_target(exploit, target):
                    probability = await self._calculate_target_probability(exploit, target)
                    
                    edge = AttackEdge(
                        source_id=exploit_id,
                        target_id=target_id,
                        relationship_type='targets',
                        probability=probability,
                        difficulty=1.0 - probability,
                        time_estimate_hours=1.5,
                        evidence_strength=0.6
                    )
                    
                    await self._add_edge_to_graph(edge)
                    edges_added += 1
        
        return edges_added
    
    async def _should_link_vuln_exploit(self, vuln: KnowledgeAtom, exploit: KnowledgeAtom) -> bool:
        """Determine if vulnerability and exploit should be linked"""
        
        vuln_content = str(vuln.content).lower()
        exploit_content = str(exploit.content).lower()
        
        # Check for common indicators
        common_terms = set(vuln_content.split()) & set(exploit_content.split())
        
        # Look for CVE, product names, or technique similarities
        if len(common_terms) > 3:  # Arbitrary threshold
            return True
        
        # Check for CVE references
        if 'cve-' in vuln_content and 'cve-' in exploit_content:
            return True
        
        # Check for product/service matches
        products = ['apache', 'nginx', 'windows', 'linux', 'mysql', 'postgresql']
        vuln_products = [p for p in products if p in vuln_content]
        exploit_products = [p for p in products if p in exploit_content]
        
        return bool(set(vuln_products) & set(exploit_products))
    
    async def _calculate_exploit_probability(self, vuln: KnowledgeAtom, exploit: KnowledgeAtom) -> float:
        """Calculate probability of successful exploitation"""
        
        base_probability = 0.5
        
        # Higher confidence = higher probability
        confidence_factor = (vuln.confidence + exploit.confidence) / 2.0
        
        # Recent vulnerabilities are more likely exploitable
        age_factor = 1.0
        if vuln.created_at:
            days_old = (datetime.utcnow() - vuln.created_at).days
            age_factor = max(0.3, 1.0 - (days_old / 365.0))  # Decay over a year
        
        # Calculate final probability
        probability = base_probability * confidence_factor * age_factor
        return max(0.1, min(0.9, probability))
    
    async def _should_link_exploit_target(self, exploit: KnowledgeAtom, target: KnowledgeAtom) -> bool:
        """Determine if exploit and target should be linked"""
        
        exploit_content = str(exploit.content).lower()
        target_content = str(target.content).lower()
        
        # Check for technology stack matches
        technologies = ['web', 'database', 'network', 'windows', 'linux', 'cloud']
        
        for tech in technologies:
            if tech in exploit_content and tech in target_content:
                return True
        
        return False
    
    async def _calculate_target_probability(self, exploit: KnowledgeAtom, target: KnowledgeAtom) -> float:
        """Calculate probability of successful targeting"""
        
        base_probability = 0.4  # Lower than exploitation
        
        # Factor in exploit and target confidence
        confidence_factor = (exploit.confidence + target.confidence) / 2.0
        
        # Consider target accessibility (mock calculation)
        accessibility_factor = 0.7  # Most targets are accessible
        
        probability = base_probability * confidence_factor * accessibility_factor
        return max(0.1, min(0.8, probability))
    
    async def _add_edge_to_graph(self, edge: AttackEdge):
        """Add edge to both in-memory and Neo4j graphs"""
        
        # Add to in-memory graph
        self.memory_graph.add_edge(
            edge.source_id,
            edge.target_id,
            relationship_type=edge.relationship_type,
            probability=edge.probability,
            difficulty=edge.difficulty,
            time_estimate_hours=edge.time_estimate_hours,
            evidence_strength=edge.evidence_strength,
            prerequisites=edge.prerequisites,
            mitigations=edge.mitigations
        )
        
        # Add to Neo4j if available
        if self.neo4j_driver:
            await self._add_edge_to_neo4j(edge)
    
    async def _add_edge_to_neo4j(self, edge: AttackEdge):
        """Add edge to Neo4j database"""
        
        query = """
        MATCH (source {id: $source_id})
        MATCH (target {id: $target_id})
        MERGE (source)-[r:ATTACK_RELATIONSHIP {type: $relationship_type}]->(target)
        SET r.probability = $probability,
            r.difficulty = $difficulty,
            r.time_estimate_hours = $time_estimate_hours,
            r.evidence_strength = $evidence_strength,
            r.prerequisites = $prerequisites,
            r.mitigations = $mitigations,
            r.last_updated = datetime()
        RETURN r
        """
        
        async with self.neo4j_driver.session() as session:
            await session.run(query, {
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'relationship_type': edge.relationship_type,
                'probability': edge.probability,
                'difficulty': edge.difficulty,
                'time_estimate_hours': edge.time_estimate_hours,
                'evidence_strength': edge.evidence_strength,
                'prerequisites': edge.prerequisites,
                'mitigations': edge.mitigations
            })
    
    async def predict_attack_paths(self, 
                                 entry_points: List[str],
                                 target_assets: List[str],
                                 max_paths: int = 10,
                                 max_length: int = 8) -> List[AttackPath]:
        """
        Predict optimal attack paths from entry points to target assets
        
        Args:
            entry_points: List of node IDs representing potential entry points
            target_assets: List of node IDs representing target assets
            max_paths: Maximum number of paths to return
            max_length: Maximum path length to consider
            
        Returns:
            List of predicted attack paths ranked by criticality
        """
        
        # Check cache first
        cache_key = self._generate_cache_key(entry_points, target_assets, max_paths, max_length)
        if cache_key in self.path_cache:
            cache_time = self.path_cache[cache_key][0].last_updated if self.path_cache[cache_key] else datetime.utcnow()
            if datetime.utcnow() - cache_time < timedelta(minutes=self.cache_ttl_minutes):
                self.prediction_metrics['cache_hit_ratio'] += 1
                return self.path_cache[cache_key]
        
        start_time = datetime.utcnow()
        predicted_paths = []
        
        # Find paths using different algorithms
        paths_found = await self._find_attack_paths_multi_algorithm(
            entry_points, target_assets, max_paths, max_length
        )
        
        # Score and rank paths
        for path_nodes in paths_found:
            attack_path = await self._create_attack_path(path_nodes, entry_points, target_assets)
            if attack_path:
                predicted_paths.append(attack_path)
        
        # Rank by criticality score
        predicted_paths.sort(key=lambda p: p.criticality_score, reverse=True)
        predicted_paths = predicted_paths[:max_paths]
        
        # Cache results
        self.path_cache[cache_key] = predicted_paths
        
        # Update metrics
        analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.prediction_metrics['total_paths_analyzed'] += len(predicted_paths)
        self.prediction_metrics['analysis_time_avg_ms'] = (
            (self.prediction_metrics['analysis_time_avg_ms'] + analysis_time) / 2
        )
        
        self.logger.info(f"Predicted {len(predicted_paths)} attack paths in {analysis_time:.1f}ms")
        return predicted_paths
    
    def _generate_cache_key(self, entry_points: List[str], target_assets: List[str], 
                          max_paths: int, max_length: int) -> str:
        """Generate cache key for path prediction"""
        
        key_data = {
            'entry_points': sorted(entry_points),
            'target_assets': sorted(target_assets),
            'max_paths': max_paths,
            'max_length': max_length
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _find_attack_paths_multi_algorithm(self, 
                                               entry_points: List[str],
                                               target_assets: List[str],
                                               max_paths: int,
                                               max_length: int) -> List[List[str]]:
        """Find attack paths using multiple algorithms"""
        
        all_paths = []
        
        # Algorithm 1: Simple shortest paths
        shortest_paths = await self._find_shortest_paths(entry_points, target_assets, max_length)
        all_paths.extend(shortest_paths)
        
        # Algorithm 2: High-probability paths
        probability_paths = await self._find_high_probability_paths(entry_points, target_assets, max_length)
        all_paths.extend(probability_paths)
        
        # Algorithm 3: Minimum difficulty paths
        easy_paths = await self._find_minimum_difficulty_paths(entry_points, target_assets, max_length)
        all_paths.extend(easy_paths)
        
        # Algorithm 4: High-impact paths
        impact_paths = await self._find_high_impact_paths(entry_points, target_assets, max_length)
        all_paths.extend(impact_paths)
        
        # Remove duplicates and limit results
        unique_paths = []
        seen_paths = set()
        
        for path in all_paths:
            path_signature = tuple(path)
            if path_signature not in seen_paths:
                seen_paths.add(path_signature)
                unique_paths.append(path)
                
                if len(unique_paths) >= max_paths * 2:  # Get extra for ranking
                    break
        
        return unique_paths
    
    async def _find_shortest_paths(self, entry_points: List[str], target_assets: List[str], max_length: int) -> List[List[str]]:
        """Find shortest paths between entry points and targets"""
        
        paths = []
        
        for entry in entry_points:
            for target in target_assets:
                if entry not in self.memory_graph or target not in self.memory_graph:
                    continue
                
                try:
                    # Find shortest path
                    path = nx.shortest_path(self.memory_graph, entry, target)
                    if len(path) <= max_length:
                        paths.append(path)
                        
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    async def _find_high_probability_paths(self, entry_points: List[str], target_assets: List[str], max_length: int) -> List[List[str]]:
        """Find paths with highest success probability"""
        
        paths = []
        
        # Create weighted graph based on probability
        weighted_graph = nx.DiGraph()
        
        for edge in self.memory_graph.edges(data=True):
            source, target, data = edge
            # Use negative log probability as weight (higher probability = lower weight)
            probability = data.get('probability', 0.5)
            weight = -np.log(max(0.001, probability)) if ML_AVAILABLE else 1.0 - probability
            weighted_graph.add_edge(source, target, weight=weight)
        
        for entry in entry_points:
            for target in target_assets:
                if entry not in weighted_graph or target not in weighted_graph:
                    continue
                
                try:
                    path = nx.shortest_path(weighted_graph, entry, target, weight='weight')
                    if len(path) <= max_length:
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    async def _find_minimum_difficulty_paths(self, entry_points: List[str], target_assets: List[str], max_length: int) -> List[List[str]]:
        """Find paths with minimum difficulty"""
        
        paths = []
        
        # Create weighted graph based on difficulty
        difficulty_graph = nx.DiGraph()
        
        for edge in self.memory_graph.edges(data=True):
            source, target, data = edge
            difficulty = data.get('difficulty', 0.5)
            difficulty_graph.add_edge(source, target, weight=difficulty)
        
        for entry in entry_points:
            for target in target_assets:
                if entry not in difficulty_graph or target not in difficulty_graph:
                    continue
                
                try:
                    path = nx.shortest_path(difficulty_graph, entry, target, weight='weight')
                    if len(path) <= max_length:
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    async def _find_high_impact_paths(self, entry_points: List[str], target_assets: List[str], max_length: int) -> List[List[str]]:
        """Find paths leading to high-impact targets"""
        
        paths = []
        
        # Sort targets by impact score
        target_impacts = []
        for target in target_assets:
            if target in self.memory_graph:
                impact = self.memory_graph.nodes[target].get('impact_score', 0.5)
                target_impacts.append((target, impact))
        
        # Focus on high-impact targets first
        target_impacts.sort(key=lambda x: x[1], reverse=True)
        high_impact_targets = [t[0] for t in target_impacts[:len(target_impacts)//2 + 1]]
        
        for entry in entry_points:
            for target in high_impact_targets:
                try:
                    # Find all simple paths (up to max_length)
                    all_simple_paths = nx.all_simple_paths(
                        self.memory_graph, entry, target, cutoff=max_length
                    )
                    
                    # Take first few paths to avoid excessive computation
                    for i, path in enumerate(all_simple_paths):
                        if i >= 3:  # Limit paths per entry-target pair
                            break
                        paths.append(path)
                        
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    async def _create_attack_path(self, path_nodes: List[str], entry_points: List[str], target_assets: List[str]) -> Optional[AttackPath]:
        """Create AttackPath object from node sequence"""
        
        if len(path_nodes) < 2:
            return None
        
        # Build nodes and edges for the path
        nodes = []
        edges = []
        
        for node_id in path_nodes:
            if node_id in self.memory_graph:
                node_data = self.memory_graph.nodes[node_id]
                
                # Create AttackNode from graph data
                node = AttackNode(
                    node_id=node_id,
                    node_type=NodeType(node_data.get('node_type', 'asset')),
                    name=node_data.get('name', node_id),
                    properties=node_data.copy(),
                    severity=node_data.get('severity', 0.5),
                    exploitability=node_data.get('exploitability', 0.5),
                    impact_score=node_data.get('impact_score', 0.5),
                    confidence=node_data.get('confidence', 0.5)
                )
                nodes.append(node)
        
        # Build edges
        for i in range(len(path_nodes) - 1):
            source_id = path_nodes[i]
            target_id = path_nodes[i + 1]
            
            if self.memory_graph.has_edge(source_id, target_id):
                edge_data = self.memory_graph.get_edge_data(source_id, target_id, 0)  # Get first edge
                
                edge = AttackEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=edge_data.get('relationship_type', 'unknown'),
                    probability=edge_data.get('probability', 0.5),
                    difficulty=edge_data.get('difficulty', 0.5),
                    time_estimate_hours=edge_data.get('time_estimate_hours', 1.0),
                    evidence_strength=edge_data.get('evidence_strength', 0.5)
                )
                edges.append(edge)
        
        # Calculate path metrics
        total_probability = 1.0
        total_difficulty = 0.0
        estimated_time = 0.0
        
        for edge in edges:
            total_probability *= edge.probability
            total_difficulty += edge.difficulty
            estimated_time += edge.time_estimate_hours
        
        # Calculate criticality score
        criticality_score = await self._calculate_path_criticality(nodes, edges, total_probability)
        
        # Determine path type
        path_type = await self._classify_path_type(nodes, edges)
        
        # Calculate detection likelihood
        detection_likelihood = await self._calculate_detection_likelihood(nodes, edges)
        
        # Calculate mitigation coverage
        mitigation_coverage = await self._calculate_mitigation_coverage(edges)
        
        path_id = hashlib.md5(str(path_nodes).encode()).hexdigest()
        
        return AttackPath(
            path_id=path_id,
            nodes=nodes,
            edges=edges,
            path_type=path_type,
            total_probability=total_probability,
            total_difficulty=total_difficulty / len(edges) if edges else 0.0,
            estimated_time_hours=estimated_time,
            criticality_score=criticality_score,
            detection_likelihood=detection_likelihood,
            mitigation_coverage=mitigation_coverage,
            path_length=len(nodes),
            entry_points=[n for n in path_nodes if n in entry_points],
            target_assets=[n for n in path_nodes if n in target_assets]
        )
    
    async def _calculate_path_criticality(self, nodes: List[AttackNode], edges: List[AttackEdge], probability: float) -> float:
        """Calculate overall criticality score for attack path"""
        
        # Factors affecting criticality:
        # 1. Probability of success
        # 2. Impact of target assets
        # 3. Difficulty of detection
        # 4. Severity of vulnerabilities exploited
        
        probability_factor = probability
        
        # Average impact of nodes
        impact_factor = sum(node.impact_score for node in nodes) / len(nodes) if nodes else 0.0
        
        # Average severity of vulnerabilities
        severity_factor = sum(node.severity for node in nodes if node.node_type == NodeType.VULNERABILITY) / max(1, len([n for n in nodes if n.node_type == NodeType.VULNERABILITY]))
        
        # Exploitability factor
        exploitability_factor = sum(node.exploitability for node in nodes) / len(nodes) if nodes else 0.0
        
        # Combine factors (weighted average)
        criticality = (
            probability_factor * 0.3 +
            impact_factor * 0.3 +
            severity_factor * 0.2 +
            exploitability_factor * 0.2
        )
        
        return min(1.0, max(0.0, criticality))
    
    async def _classify_path_type(self, nodes: List[AttackNode], edges: List[AttackEdge]) -> PathType:
        """Classify the type of attack path"""
        
        # Analyze relationship types to determine path type
        relationship_types = [edge.relationship_type for edge in edges]
        
        if any('escalates' in rel for rel in relationship_types):
            return PathType.PRIVILEGE_ESCALATION
        elif any('lateral' in rel for rel in relationship_types):
            return PathType.LATERAL
        elif any('persist' in rel for rel in relationship_types):
            return PathType.PERSISTENCE
        elif any('exfiltrat' in rel for rel in relationship_types):
            return PathType.EXFILTRATION
        elif len(nodes) <= 3:
            return PathType.DIRECT
        else:
            return PathType.IMPACT
    
    async def _calculate_detection_likelihood(self, nodes: List[AttackNode], edges: List[AttackEdge]) -> float:
        """Calculate likelihood of path detection"""
        
        # Base detection likelihood
        base_detection = 0.3
        
        # Factors that increase detection
        visibility_factors = []
        
        for node in nodes:
            # High-profile targets are more monitored
            if node.node_type == NodeType.ASSET and node.impact_score > 0.7:
                visibility_factors.append(0.2)
            
            # Recent vulnerabilities are more monitored
            if node.node_type == NodeType.VULNERABILITY and node.confidence > 0.8:
                visibility_factors.append(0.1)
        
        # Path length affects detection (longer paths harder to detect completely)
        length_factor = 1.0 / (1.0 + len(nodes) * 0.1)
        
        detection_likelihood = base_detection + sum(visibility_factors) * length_factor
        return min(0.9, max(0.1, detection_likelihood))
    
    async def _calculate_mitigation_coverage(self, edges: List[AttackEdge]) -> float:
        """Calculate coverage of available mitigations"""
        
        total_edges = len(edges)
        if total_edges == 0:
            return 0.0
        
        mitigated_edges = sum(1 for edge in edges if edge.mitigations)
        return mitigated_edges / total_edges
    
    async def _update_graph_analysis_caches(self):
        """Update cached graph analysis results"""
        
        if len(self.memory_graph.nodes) == 0:
            return
        
        # Update centrality cache
        try:
            self.centrality_cache = {
                'betweenness': dict(nx.betweenness_centrality(self.memory_graph)),
                'closeness': dict(nx.closeness_centrality(self.memory_graph)),
                'pagerank': dict(nx.pagerank(self.memory_graph))
            }
        except Exception as e:
            self.logger.debug(f"Centrality calculation failed: {e}")
        
        # Update community detection cache
        if ML_AVAILABLE and len(self.memory_graph.nodes) > 5:
            try:
                # Convert to undirected for community detection
                undirected = self.memory_graph.to_undirected()
                communities = list(nx.community.greedy_modularity_communities(undirected))
                
                self.community_cache = {}
                for i, community in enumerate(communities):
                    self.community_cache[f'community_{i}'] = list(community)
                    
            except Exception as e:
                self.logger.debug(f"Community detection failed: {e}")
    
    async def get_graph_metrics(self) -> GraphMetrics:
        """Get comprehensive graph analysis metrics"""
        
        if len(self.memory_graph.nodes) == 0:
            return GraphMetrics(
                node_count=0, edge_count=0, avg_clustering_coefficient=0.0,
                avg_path_length=0.0, network_diameter=0, centrality_analysis={},
                community_structure={}, vulnerability_density=0.0, exploit_coverage=0.0
            )
        
        # Basic metrics
        node_count = len(self.memory_graph.nodes)
        edge_count = len(self.memory_graph.edges)
        
        # Clustering coefficient
        try:
            avg_clustering = nx.average_clustering(self.memory_graph.to_undirected())
        except:
            avg_clustering = 0.0
        
        # Average path length and diameter
        try:
            if nx.is_connected(self.memory_graph.to_undirected()):
                avg_path_length = nx.average_shortest_path_length(self.memory_graph.to_undirected())
                diameter = nx.diameter(self.memory_graph.to_undirected())
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(self.memory_graph.to_undirected()), key=len)
                subgraph = self.memory_graph.subgraph(largest_cc).to_undirected()
                avg_path_length = nx.average_shortest_path_length(subgraph)
                diameter = nx.diameter(subgraph)
        except:
            avg_path_length = 0.0
            diameter = 0
        
        # Calculate specialized metrics
        vulnerability_nodes = [n for n, d in self.memory_graph.nodes(data=True) if d.get('node_type') == 'vulnerability']
        exploit_nodes = [n for n, d in self.memory_graph.nodes(data=True) if d.get('node_type') == 'exploit']
        
        vulnerability_density = len(vulnerability_nodes) / node_count if node_count > 0 else 0.0
        exploit_coverage = len(exploit_nodes) / max(1, len(vulnerability_nodes)) if vulnerability_nodes else 0.0
        
        return GraphMetrics(
            node_count=node_count,
            edge_count=edge_count,
            avg_clustering_coefficient=avg_clustering,
            avg_path_length=avg_path_length,
            network_diameter=diameter,
            centrality_analysis=self.centrality_cache.copy(),
            community_structure=self.community_cache.copy(),
            vulnerability_density=vulnerability_density,
            exploit_coverage=exploit_coverage
        )
    
    async def get_high_value_targets(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify high-value targets based on graph analysis"""
        
        targets = []
        
        # Get centrality scores
        centrality_scores = self.centrality_cache.get('pagerank', {})
        
        for node_id, node_data in self.memory_graph.nodes(data=True):
            if node_data.get('node_type') == 'asset':
                
                # Calculate composite score
                impact_score = node_data.get('impact_score', 0.5)
                severity_score = node_data.get('severity', 0.5)
                centrality_score = centrality_scores.get(node_id, 0.0)
                
                # Weight factors
                composite_score = (
                    impact_score * 0.4 +
                    severity_score * 0.3 +
                    centrality_score * 0.3
                )
                
                targets.append({
                    'node_id': node_id,
                    'name': node_data.get('name', node_id),
                    'composite_score': composite_score,
                    'impact_score': impact_score,
                    'severity_score': severity_score,
                    'centrality_score': centrality_score,
                    'properties': node_data.copy()
                })
        
        # Sort by composite score and return top N
        targets.sort(key=lambda x: x['composite_score'], reverse=True)
        return targets[:top_n]
    
    async def shutdown(self):
        """Shutdown graph predictor and close connections"""
        
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        
        self.logger.info("Graph Attack Predictor shutdown complete")


if __name__ == "__main__":
    async def main():
        # Example usage
        predictor = GraphAttackPredictor()
        await predictor.initialize()
        
        # Mock knowledge atoms for testing
        from .atom import KnowledgeAtom, AtomType
        
        test_atoms = [
            KnowledgeAtom(
                id="vuln_001",
                atom_type=AtomType.VULNERABILITY,
                title="SQL Injection in Login Form",
                content={"cvss_score": 8.5, "severity": "high"},
                confidence=0.9
            ),
            KnowledgeAtom(
                id="exploit_001", 
                atom_type=AtomType.EXPLOIT,
                title="SQLi Exploit Tool",
                content={"difficulty": "medium", "availability": "public"},
                confidence=0.8
            ),
            KnowledgeAtom(
                id="target_001",
                atom_type=AtomType.TARGET,
                title="Web Application Server",
                content={"criticality": "high", "technologies": ["mysql", "php"]},
                confidence=0.7
            )
        ]
        
        # Build graph
        nodes_added = await predictor.build_attack_graph(test_atoms)
        print(f"Built graph with {nodes_added} nodes")
        
        # Predict attack paths
        if nodes_added > 0:
            paths = await predictor.predict_attack_paths(
                entry_points=["vuln_001"],
                target_assets=["target_001"],
                max_paths=5
            )
            
            print(f"Found {len(paths)} attack paths")
            for i, path in enumerate(paths):
                print(f"Path {i+1}: {path.path_length} nodes, "
                      f"criticality: {path.criticality_score:.2f}, "
                      f"probability: {path.total_probability:.2f}")
        
        # Get graph metrics
        metrics = await predictor.get_graph_metrics()
        print(f"Graph metrics: {metrics.node_count} nodes, {metrics.edge_count} edges")
        
        # Get high-value targets
        targets = await predictor.get_high_value_targets(5)
        print(f"High-value targets: {len(targets)}")
        
        await predictor.shutdown()
    
    asyncio.run(main())