#!/usr/bin/env python3

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import ServiceUnavailable, ClientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j not available. Install with: pip install neo4j")

try:
    import networkx as nx
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    GRAPH_ANALYTICS_AVAILABLE = True
except ImportError:
    GRAPH_ANALYTICS_AVAILABLE = False
    logging.warning("Graph analytics not available. Install with: pip install networkx scikit-learn")

from .atom import KnowledgeAtom, AtomType, AtomRelationship
from .core import KnowledgeFabric


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    atom_type: str
    labels: List[str]
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    properties: Dict[str, Any]
    created_at: datetime


@dataclass
class AttackPath:
    """Represents a potential attack path through the knowledge graph."""
    path_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_score: float
    complexity_score: float
    success_probability: float
    metadata: Dict[str, Any]


@dataclass
class KnowledgeCluster:
    """Represents a cluster of related knowledge atoms."""
    cluster_id: str
    center_node: GraphNode
    member_nodes: List[GraphNode]
    cluster_score: float
    topic_keywords: List[str]
    metadata: Dict[str, Any]


class Neo4jKnowledgeGraph:
    """Neo4j-based knowledge graph implementation for XORB."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "xorb_graph_2024"):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize connection to Neo4j database."""
        if not NEO4J_AVAILABLE:
            raise RuntimeError("Neo4j not available. Install with: pip install neo4j")
        
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # Test connection
            await self.verify_connectivity()
            
            # Create indexes and constraints
            await self.setup_database_schema()
            
            self.logger.info("Neo4j Knowledge Graph initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j: {e}")
            raise
    
    async def verify_connectivity(self):
        """Verify database connectivity."""
        async with self.driver.session() as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            if record["test"] != 1:
                raise RuntimeError("Neo4j connectivity test failed")
    
    async def setup_database_schema(self):
        """Setup database indexes and constraints."""
        async with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT atom_id_unique IF NOT EXISTS FOR (a:Atom) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT vulnerability_cve_unique IF NOT EXISTS FOR (v:Vulnerability) REQUIRE v.cve_id IS UNIQUE",
                "CREATE CONSTRAINT technique_id_unique IF NOT EXISTS FOR (t:Technique) REQUIRE t.technique_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except ClientError as e:
                    if "already exists" not in str(e).lower():
                        self.logger.warning(f"Failed to create constraint: {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX atom_type_index IF NOT EXISTS FOR (a:Atom) ON (a.atom_type)",
                "CREATE INDEX atom_confidence_index IF NOT EXISTS FOR (a:Atom) ON (a.confidence)",
                "CREATE INDEX atom_created_index IF NOT EXISTS FOR (a:Atom) ON (a.created_at)",
                "CREATE INDEX vulnerability_severity_index IF NOT EXISTS FOR (v:Vulnerability) ON (v.severity)",
                "CREATE INDEX technique_tactic_index IF NOT EXISTS FOR (t:Technique) ON (t.tactic)"
            ]
            
            for index in indexes:
                try:
                    await session.run(index)
                except ClientError as e:
                    if "already exists" not in str(e).lower():
                        self.logger.warning(f"Failed to create index: {e}")
    
    async def store_atom(self, atom: KnowledgeAtom) -> GraphNode:
        """Store a knowledge atom as a graph node."""
        async with self.driver.session() as session:
            # Determine node labels based on atom type
            labels = ["Atom", atom.atom_type.value.title()]
            
            # Prepare properties
            properties = {
                "id": atom.id,
                "atom_type": atom.atom_type.value,
                "confidence": atom.confidence,
                "success_rate": atom.success_rate,
                "usage_count": atom.usage_count,
                "created_at": atom.created_at.isoformat(),
                "updated_at": atom.updated_at.isoformat() if atom.updated_at else None,
                "content_summary": self._extract_content_summary(atom.content),
                "source_count": len(atom.sources),
                "validation_count": len(atom.validation_results)
            }
            
            # Add type-specific properties
            if atom.atom_type == AtomType.VULNERABILITY:
                properties.update(self._extract_vulnerability_properties(atom))
            elif atom.atom_type == AtomType.TECHNIQUE:
                properties.update(self._extract_technique_properties(atom))
            elif atom.atom_type == AtomType.EXPLOIT:
                properties.update(self._extract_exploit_properties(atom))
            
            # Create or update node
            query = f"""
            MERGE (a:{':'.join(labels)} {{id: $id}})
            SET a += $properties, a.last_updated = datetime()
            RETURN a
            """
            
            result = await session.run(query, id=atom.id, properties=properties)
            record = await result.single()
            
            if record:
                # Store relationships
                for relationship in atom.related_atoms:
                    await self._store_relationship(session, atom.id, relationship)
                
                return GraphNode(
                    id=atom.id,
                    atom_type=atom.atom_type.value,
                    labels=labels,
                    properties=properties,
                    confidence=atom.confidence,
                    created_at=atom.created_at,
                    updated_at=atom.updated_at or datetime.utcnow()
                )
            
            return None
    
    async def _store_relationship(self, session: AsyncSession, source_id: str, relationship: AtomRelationship):
        """Store a relationship between atoms."""
        query = """
        MATCH (source:Atom {id: $source_id})
        MATCH (target:Atom {id: $target_id})
        MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
        SET r.strength = $strength,
            r.created_at = $created_at,
            r.metadata = $metadata
        """
        
        await session.run(
            query,
            source_id=source_id,
            target_id=relationship.target_atom_id,
            rel_type=relationship.relationship_type,
            strength=relationship.strength,
            created_at=relationship.created_at.isoformat(),
            metadata=json.dumps(relationship.metadata or {})
        )
    
    def _extract_content_summary(self, content: Any) -> str:
        """Extract a summary from atom content."""
        if isinstance(content, dict):
            summary_parts = []
            if 'title' in content:
                summary_parts.append(content['title'])
            if 'description' in content:
                desc = str(content['description'])[:200]
                summary_parts.append(desc)
            return " | ".join(summary_parts)
        return str(content)[:200]
    
    def _extract_vulnerability_properties(self, atom: KnowledgeAtom) -> Dict[str, Any]:
        """Extract vulnerability-specific properties."""
        content = atom.content if isinstance(atom.content, dict) else {}
        return {
            "cve_id": content.get("cve_id"),
            "severity": content.get("severity", "unknown"),
            "cvss_score": content.get("cvss_score", 0.0),
            "affected_systems": json.dumps(content.get("affected_systems", [])),
            "mitigation": content.get("mitigation", ""),
            "exploit_available": content.get("exploit_available", False)
        }
    
    def _extract_technique_properties(self, atom: KnowledgeAtom) -> Dict[str, Any]:
        """Extract technique-specific properties."""
        content = atom.content if isinstance(atom.content, dict) else {}
        return {
            "technique_id": content.get("technique_id"),
            "tactic": content.get("tactic", "unknown"),
            "platform": json.dumps(content.get("platform", [])),
            "detection_difficulty": content.get("detection_difficulty", "medium"),
            "prerequisites": json.dumps(content.get("prerequisites", [])),
            "tools_required": json.dumps(content.get("tools_required", []))
        }
    
    def _extract_exploit_properties(self, atom: KnowledgeAtom) -> Dict[str, Any]:
        """Extract exploit-specific properties."""
        content = atom.content if isinstance(atom.content, dict) else {}
        return {
            "exploit_type": content.get("exploit_type", "unknown"),
            "complexity": content.get("complexity", "medium"),
            "reliability": content.get("reliability", 0.5),
            "target_vulnerability": content.get("target_vulnerability"),
            "payload_type": content.get("payload_type"),
            "stealth_rating": content.get("stealth_rating", 0.5)
        }
    
    async def find_attack_paths(self, start_atom_id: str, end_atom_id: str, 
                               max_depth: int = 5) -> List[AttackPath]:
        """Find potential attack paths between two atoms."""
        async with self.driver.session() as session:
            query = """
            MATCH path = (start:Atom {id: $start_id})-[:RELATES_TO*1..$max_depth]-(end:Atom {id: $end_id})
            WHERE ALL(r IN relationships(path) WHERE r.strength > 0.3)
            RETURN path, 
                   length(path) as path_length,
                   reduce(score = 1.0, r IN relationships(path) | score * r.strength) as path_score
            ORDER BY path_score DESC, path_length ASC
            LIMIT 10
            """
            
            result = await session.run(
                query, 
                start_id=start_atom_id, 
                end_id=end_atom_id, 
                max_depth=max_depth
            )
            
            attack_paths = []
            async for record in result:
                path = record["path"]
                path_score = record["path_score"]
                path_length = record["path_length"]
                
                # Extract nodes and edges from path
                nodes = []
                edges = []
                
                for i, node in enumerate(path.nodes):
                    graph_node = GraphNode(
                        id=node["id"],
                        atom_type=node["atom_type"],
                        labels=list(node.labels),
                        properties=dict(node),
                        confidence=node.get("confidence", 0.5),
                        created_at=datetime.fromisoformat(node["created_at"]),
                        updated_at=datetime.fromisoformat(node.get("updated_at", node["created_at"]))
                    )
                    nodes.append(graph_node)
                
                for relationship in path.relationships:
                    graph_edge = GraphEdge(
                        source_id=relationship.start_node["id"],
                        target_id=relationship.end_node["id"],
                        relationship_type=relationship.type,
                        strength=relationship.get("strength", 0.5),
                        properties=dict(relationship),
                        created_at=datetime.fromisoformat(relationship.get("created_at", datetime.utcnow().isoformat()))
                    )
                    edges.append(graph_edge)
                
                # Calculate additional metrics
                complexity_score = self._calculate_path_complexity(nodes, edges)
                success_probability = self._calculate_success_probability(nodes, edges)
                
                attack_path = AttackPath(
                    path_id=hashlib.md5(f"{start_atom_id}_{end_atom_id}_{path_length}_{path_score}".encode()).hexdigest()[:16],
                    nodes=nodes,
                    edges=edges,
                    total_score=path_score,
                    complexity_score=complexity_score,
                    success_probability=success_probability,
                    metadata={
                        "path_length": path_length,
                        "avg_confidence": sum(n.confidence for n in nodes) / len(nodes),
                        "technique_count": len([n for n in nodes if n.atom_type == "technique"]),
                        "vulnerability_count": len([n for n in nodes if n.atom_type == "vulnerability"])
                    }
                )
                
                attack_paths.append(attack_path)
            
            return attack_paths
    
    def _calculate_path_complexity(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> float:
        """Calculate the complexity score of an attack path."""
        if not nodes:
            return 1.0
        
        # Factors that increase complexity
        complexity_factors = []
        
        # Number of steps
        complexity_factors.append(len(nodes) / 10.0)
        
        # Types of techniques involved
        technique_types = set()
        for node in nodes:
            if node.atom_type == "technique":
                tactic = node.properties.get("tactic", "unknown")
                technique_types.add(tactic)
        
        complexity_factors.append(len(technique_types) / 5.0)  # Normalize by max expected tactics
        
        # Average detection difficulty
        detection_scores = []
        for node in nodes:
            if node.atom_type == "technique":
                difficulty = node.properties.get("detection_difficulty", "medium")
                score = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(difficulty, 0.6)
                detection_scores.append(score)
        
        if detection_scores:
            complexity_factors.append(sum(detection_scores) / len(detection_scores))
        
        return min(1.0, sum(complexity_factors) / len(complexity_factors))
    
    def _calculate_success_probability(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> float:
        """Calculate the success probability of an attack path."""
        if not nodes or not edges:
            return 0.0
        
        # Start with the average confidence of nodes
        node_confidence = sum(n.confidence for n in nodes) / len(nodes)
        
        # Factor in edge strengths
        edge_strength = sum(e.strength for e in edges) / len(edges)
        
        # Consider path length (longer paths are less likely to succeed)
        length_penalty = max(0.1, 1.0 - (len(nodes) - 2) * 0.1)
        
        # Consider technique reliability
        technique_reliability = []
        for node in nodes:
            if node.atom_type == "technique":
                complexity = node.properties.get("complexity", "medium")
                reliability = {"low": 0.9, "medium": 0.7, "high": 0.5}.get(complexity, 0.7)
                technique_reliability.append(reliability)
        
        if technique_reliability:
            avg_reliability = sum(technique_reliability) / len(technique_reliability)
        else:
            avg_reliability = 0.7
        
        # Combine all factors
        success_prob = node_confidence * edge_strength * length_penalty * avg_reliability
        
        return min(1.0, max(0.0, success_prob))
    
    async def discover_knowledge_clusters(self, min_cluster_size: int = 3) -> List[KnowledgeCluster]:
        """Discover clusters of related knowledge atoms."""
        async with self.driver.session() as session:
            # Use community detection algorithm
            query = """
            CALL gds.graph.project.cypher(
                'knowledge-graph',
                'MATCH (a:Atom) RETURN id(a) AS id, a.atom_type AS type, a.confidence AS confidence',
                'MATCH (a:Atom)-[r:RELATES_TO]->(b:Atom) RETURN id(a) AS source, id(b) AS target, r.strength AS weight'
            )
            YIELD graphName, nodeCount, relationshipCount
            """
            
            try:
                await session.run(query)
                
                # Run Louvain community detection
                community_query = """
                CALL gds.louvain.stream('knowledge-graph', {relationshipWeightProperty: 'weight'})
                YIELD nodeId, communityId
                MATCH (a:Atom) WHERE id(a) = nodeId
                RETURN a.id as atom_id, a.atom_type as type, a.confidence as confidence, 
                       communityId, a.content_summary as summary
                ORDER BY communityId, confidence DESC
                """
                
                result = await session.run(community_query)
                
                # Group atoms by community
                communities = {}
                async for record in result:
                    community_id = record["communityId"]
                    if community_id not in communities:
                        communities[community_id] = []
                    
                    communities[community_id].append({
                        "atom_id": record["atom_id"],
                        "type": record["type"],
                        "confidence": record["confidence"],
                        "summary": record["summary"]
                    })
                
                # Create clusters from communities
                clusters = []
                for community_id, atoms in communities.items():
                    if len(atoms) >= min_cluster_size:
                        # Find center node (highest confidence)
                        center_atom = max(atoms, key=lambda x: x["confidence"])
                        
                        # Extract topic keywords
                        summaries = [atom["summary"] for atom in atoms if atom["summary"]]
                        keywords = self._extract_keywords_from_summaries(summaries)
                        
                        cluster = KnowledgeCluster(
                            cluster_id=f"cluster_{community_id}",
                            center_node=GraphNode(
                                id=center_atom["atom_id"],
                                atom_type=center_atom["type"],
                                labels=["Atom", center_atom["type"].title()],
                                properties={"confidence": center_atom["confidence"]},
                                confidence=center_atom["confidence"],
                                created_at=datetime.utcnow(),
                                updated_at=datetime.utcnow()
                            ),
                            member_nodes=[],  # Would populate with full node details if needed
                            cluster_score=sum(atom["confidence"] for atom in atoms) / len(atoms),
                            topic_keywords=keywords,
                            metadata={
                                "atom_count": len(atoms),
                                "avg_confidence": sum(atom["confidence"] for atom in atoms) / len(atoms),
                                "atom_types": list(set(atom["type"] for atom in atoms))
                            }
                        )
                        
                        clusters.append(cluster)
                
                # Clean up graph
                await session.run("CALL gds.graph.drop('knowledge-graph')")
                
                return clusters
                
            except Exception as e:
                self.logger.error(f"Community detection failed: {e}")
                # Fallback to simple clustering
                return await self._simple_clustering(session, min_cluster_size)
    
    def _extract_keywords_from_summaries(self, summaries: List[str]) -> List[str]:
        """Extract keywords from cluster summaries."""
        if not summaries:
            return []
        
        # Simple keyword extraction (in production, use more sophisticated NLP)
        all_text = " ".join(summaries).lower()
        
        # Common security keywords
        security_keywords = [
            "vulnerability", "exploit", "injection", "authentication", "authorization",
            "xss", "csrf", "sql", "rce", "lfi", "rfi", "privilege", "escalation",
            "buffer", "overflow", "dos", "ddos", "malware", "backdoor", "trojan",
            "phishing", "social", "engineering", "reconnaissance", "enumeration",
            "brute", "force", "dictionary", "password", "credential", "session",
            "token", "encryption", "decryption", "hash", "cryptography"
        ]
        
        found_keywords = []
        for keyword in security_keywords:
            if keyword in all_text:
                found_keywords.append(keyword)
        
        return found_keywords[:10]  # Return top 10
    
    async def _simple_clustering(self, session: AsyncSession, min_cluster_size: int) -> List[KnowledgeCluster]:
        """Fallback simple clustering method."""
        # Simple clustering based on atom types and confidence
        query = """
        MATCH (a:Atom)
        RETURN a.id as id, a.atom_type as type, a.confidence as confidence, 
               a.content_summary as summary
        ORDER BY a.atom_type, a.confidence DESC
        """
        
        result = await session.run(query)
        
        atoms_by_type = {}
        async for record in result:
            atom_type = record["type"]
            if atom_type not in atoms_by_type:
                atoms_by_type[atom_type] = []
            
            atoms_by_type[atom_type].append({
                "id": record["id"],
                "type": record["type"],
                "confidence": record["confidence"],
                "summary": record["summary"]
            })
        
        clusters = []
        for atom_type, atoms in atoms_by_type.items():
            if len(atoms) >= min_cluster_size:
                center_atom = atoms[0]  # Highest confidence due to ordering
                
                cluster = KnowledgeCluster(
                    cluster_id=f"simple_cluster_{atom_type}",
                    center_node=GraphNode(
                        id=center_atom["id"],
                        atom_type=center_atom["type"],
                        labels=["Atom", center_atom["type"].title()],
                        properties={"confidence": center_atom["confidence"]},
                        confidence=center_atom["confidence"],
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    ),
                    member_nodes=[],
                    cluster_score=sum(atom["confidence"] for atom in atoms) / len(atoms),
                    topic_keywords=[atom_type],
                    metadata={
                        "atom_count": len(atoms),
                        "clustering_method": "simple_type_based"
                    }
                )
                
                clusters.append(cluster)
        
        return clusters
    
    async def get_atom_neighbors(self, atom_id: str, max_distance: int = 2) -> Dict[str, Any]:
        """Get neighboring atoms and their relationships."""
        async with self.driver.session() as session:
            query = """
            MATCH path = (center:Atom {id: $atom_id})-[:RELATES_TO*1..$max_distance]-(neighbor:Atom)
            RETURN neighbor, relationships(path) as rels, length(path) as distance
            ORDER BY distance, neighbor.confidence DESC
            LIMIT 50
            """
            
            result = await session.run(query, atom_id=atom_id, max_distance=max_distance)
            
            neighbors = {}
            async for record in result:
                neighbor = record["neighbor"]
                distance = record["distance"]
                
                if distance not in neighbors:
                    neighbors[distance] = []
                
                neighbor_node = GraphNode(
                    id=neighbor["id"],
                    atom_type=neighbor["atom_type"],
                    labels=list(neighbor.labels),
                    properties=dict(neighbor),
                    confidence=neighbor.get("confidence", 0.5),
                    created_at=datetime.fromisoformat(neighbor["created_at"]),
                    updated_at=datetime.fromisoformat(neighbor.get("updated_at", neighbor["created_at"]))
                )
                
                neighbors[distance].append(neighbor_node)
            
            return neighbors
    
    async def close(self):
        """Close the database connection."""
        if self.driver:
            await self.driver.close()


class GraphKnowledgeAnalyzer:
    """Advanced analytics for knowledge graph data."""
    
    def __init__(self, graph: Neo4jKnowledgeGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)
    
    async def analyze_attack_surface(self, target_atoms: List[str]) -> Dict[str, Any]:
        """Analyze the attack surface for given target atoms."""
        attack_surface = {
            "total_vulnerabilities": 0,
            "critical_paths": [],
            "entry_points": [],
            "high_value_targets": [],
            "risk_score": 0.0,
            "recommendations": []
        }
        
        try:
            async with self.graph.driver.session() as session:
                # Find all vulnerabilities connected to targets
                vuln_query = """
                MATCH (target:Atom)
                WHERE target.id IN $target_ids
                MATCH (target)-[:RELATES_TO*1..3]-(vuln:Vulnerability)
                RETURN vuln, target.id as target_id
                ORDER BY vuln.cvss_score DESC
                """
                
                result = await session.run(vuln_query, target_ids=target_atoms)
                
                vulnerabilities = []
                async for record in result:
                    vuln = record["vuln"]
                    vulnerabilities.append({
                        "id": vuln["id"],
                        "cvss_score": vuln.get("cvss_score", 0.0),
                        "severity": vuln.get("severity", "unknown"),
                        "target_id": record["target_id"]
                    })
                
                attack_surface["total_vulnerabilities"] = len(vulnerabilities)
                
                # Calculate risk score
                if vulnerabilities:
                    avg_cvss = sum(v["cvss_score"] for v in vulnerabilities) / len(vulnerabilities)
                    critical_count = len([v for v in vulnerabilities if v["cvss_score"] >= 9.0])
                    high_count = len([v for v in vulnerabilities if 7.0 <= v["cvss_score"] < 9.0])
                    
                    risk_score = (avg_cvss / 10.0) * 0.6 + (critical_count / len(vulnerabilities)) * 0.4
                    attack_surface["risk_score"] = min(1.0, risk_score)
                
                # Find critical attack paths
                for target_id in target_atoms:
                    paths = await self.graph.find_attack_paths("initial_access", target_id, max_depth=4)
                    critical_paths = [p for p in paths if p.total_score > 0.7]
                    attack_surface["critical_paths"].extend(critical_paths)
                
                # Generate recommendations
                attack_surface["recommendations"] = self._generate_security_recommendations(vulnerabilities)
        
        except Exception as e:
            self.logger.error(f"Attack surface analysis failed: {e}")
        
        return attack_surface
    
    def _generate_security_recommendations(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate security recommendations based on vulnerabilities."""
        recommendations = []
        
        # High CVSS score recommendations
        critical_vulns = [v for v in vulnerabilities if v["cvss_score"] >= 9.0]
        if critical_vulns:
            recommendations.append({
                "priority": "critical",
                "action": "patch_critical_vulnerabilities",
                "description": f"Immediately patch {len(critical_vulns)} critical vulnerabilities",
                "affected_count": len(critical_vulns)
            })
        
        # Severity-based recommendations
        severity_counts = {}
        for vuln in vulnerabilities:
            severity = vuln["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts.get("high", 0) > 5:
            recommendations.append({
                "priority": "high",
                "action": "vulnerability_assessment",
                "description": "Conduct comprehensive vulnerability assessment",
                "affected_count": severity_counts["high"]
            })
        
        return recommendations
    
    async def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify gaps in the knowledge graph."""
        gaps = []
        
        try:
            async with self.graph.driver.session() as session:
                # Find isolated atoms
                isolation_query = """
                MATCH (a:Atom)
                WHERE NOT (a)-[:RELATES_TO]-()
                RETURN a.id as id, a.atom_type as type, a.confidence as confidence
                ORDER BY a.confidence DESC
                LIMIT 20
                """
                
                result = await session.run(isolation_query)
                isolated_atoms = []
                async for record in result:
                    isolated_atoms.append({
                        "id": record["id"],
                        "type": record["type"],
                        "confidence": record["confidence"]
                    })
                
                if isolated_atoms:
                    gaps.append({
                        "gap_type": "isolated_atoms",
                        "description": f"{len(isolated_atoms)} atoms have no relationships",
                        "severity": "medium",
                        "recommendations": ["Link isolated atoms to related concepts", "Validate atom relevance"]
                    })
                
                # Find technique gaps
                technique_query = """
                MATCH (v:Vulnerability)
                WHERE NOT (v)-[:RELATES_TO]->(:Technique)
                RETURN count(v) as vulns_without_techniques
                """
                
                result = await session.run(technique_query)
                record = await result.single()
                vulns_without_techniques = record["vulns_without_techniques"]
                
                if vulns_without_techniques > 0:
                    gaps.append({
                        "gap_type": "missing_techniques",
                        "description": f"{vulns_without_techniques} vulnerabilities lack associated techniques",
                        "severity": "high",
                        "recommendations": ["Research attack techniques for vulnerabilities", "Update knowledge base"]
                    })
        
        except Exception as e:
            self.logger.error(f"Knowledge gap analysis failed: {e}")
        
        return gaps


class GraphKnowledgeFabric(KnowledgeFabric):
    """Enhanced knowledge fabric with graph capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = Neo4jKnowledgeGraph()
        self.analyzer = GraphKnowledgeAnalyzer(self.graph)
        
    async def initialize(self):
        """Initialize both traditional and graph knowledge systems."""
        await super().initialize()
        
        try:
            await self.graph.initialize()
            self.logger.info("Graph knowledge fabric initialized")
        except Exception as e:
            self.logger.warning(f"Graph initialization failed, continuing without graph features: {e}")
    
    async def store_atom(self, atom: KnowledgeAtom) -> str:
        """Store atom in both traditional and graph storage."""
        # Store in traditional fabric
        atom_id = await super().store_atom(atom)
        
        # Store in graph if available
        try:
            if self.graph.driver:
                await self.graph.store_atom(atom)
        except Exception as e:
            self.logger.warning(f"Graph storage failed: {e}")
        
        return atom_id
    
    async def find_attack_paths(self, start_atom_id: str, end_atom_id: str, max_depth: int = 5) -> List[AttackPath]:
        """Find attack paths using graph analysis."""
        if not self.graph.driver:
            return []
        
        return await self.graph.find_attack_paths(start_atom_id, end_atom_id, max_depth)
    
    async def discover_knowledge_clusters(self, min_cluster_size: int = 3) -> List[KnowledgeCluster]:
        """Discover knowledge clusters using graph algorithms."""
        if not self.graph.driver:
            return []
        
        return await self.graph.discover_knowledge_clusters(min_cluster_size)
    
    async def analyze_attack_surface(self, target_atoms: List[str]) -> Dict[str, Any]:
        """Analyze attack surface for given targets."""
        if not self.analyzer:
            return {}
        
        return await self.analyzer.analyze_attack_surface(target_atoms)
    
    async def close(self):
        """Close all connections."""
        await super().close()
        if self.graph:
            await self.graph.close()