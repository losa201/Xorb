#!/usr/bin/env python3

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, 
        FieldCondition, MatchValue, SearchRequest
    )
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Vector search will be disabled.")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence transformers not available. Using fallback embeddings.")

import numpy as np
from .core import KnowledgeFabric
from .atom import KnowledgeAtom, AtomType, Source


class VectorEmbeddingService:
    """Service for generating embeddings from text"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.logger = logging.getLogger(__name__)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            self.logger.warning("Sentence transformers not available. Using fallback.")
            return
        
        try:
            # Use CPU-optimized model
            self.model = SentenceTransformer(self.model_name)
            
            # Set to CPU explicitly to avoid GPU memory issues
            if torch.cuda.is_available():
                self.model = self.model.to('cpu')
            
            self.logger.info(f"Loaded embedding model: {self.model_name}")
            
            # Verify dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dimension = len(test_embedding[0])
            self.logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    async def encode_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self.model:
            return self._fallback_embedding(text)
        
        try:
            # Generate embedding
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return self._fallback_embedding(text)

    async def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        if not self.model:
            return [self._fallback_embedding(text) for text in texts]
        
        try:
            embeddings = self.model.encode(texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {e}")
            return [self._fallback_embedding(text) for text in texts]

    def _fallback_embedding(self, text: str) -> List[float]:
        """Simple hash-based fallback embedding"""
        # Create a deterministic but diverse embedding based on text hash
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % (2**32))
        
        # Generate random vector and normalize
        vector = np.random.normal(0, 1, self.embedding_dimension)
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0


class VectorStore:
    """Qdrant-based vector storage for knowledge atoms"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client = None
        self.collection_name = "xorb_knowledge"
        self.logger = logging.getLogger(__name__)
        
        if QDRANT_AVAILABLE:
            self._initialize_client()
        else:
            self.logger.warning("Qdrant not available. Vector search disabled.")

    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            
            # Test connection
            collections = self.client.get_collections()
            self.logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            
            # Create collection if it doesn't exist
            self._ensure_collection_exists()
            
        except Exception as e:
            self.logger.warning(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def _ensure_collection_exists(self):
        """Ensure the knowledge collection exists"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Dimension for all-MiniLM-L6-v2
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created collection: {self.collection_name}")
            else:
                self.logger.info(f"Collection already exists: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to ensure collection exists: {e}")
            self.client = None

    async def add_atom_vector(self, atom: KnowledgeAtom, embedding: List[float]) -> bool:
        """Add atom with its vector embedding"""
        if not self.client:
            return False
        
        try:
            # Create point for Qdrant
            point = PointStruct(
                id=atom.id,
                vector=embedding,
                payload={
                    "atom_type": atom.atom_type.value,
                    "title": atom.title,
                    "confidence": atom.confidence,
                    "predictive_score": atom.predictive_score,
                    "tags": list(atom.tags),
                    "created_at": atom.created_at.isoformat(),
                    "updated_at": atom.updated_at.isoformat(),
                    "content_preview": str(atom.content)[:500],  # First 500 chars for filtering
                    "source_count": len(atom.sources),
                    "usage_count": atom.usage_count,
                    "success_rate": atom.success_rate
                }
            )
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            self.logger.debug(f"Added vector for atom {atom.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add atom vector {atom.id}: {e}")
            return False

    async def search_similar(self, 
                           query_embedding: List[float], 
                           limit: int = 10,
                           atom_type: Optional[AtomType] = None,
                           min_confidence: float = 0.0,
                           tags: Optional[List[str]] = None,
                           score_threshold: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """Search for similar atoms using vector similarity"""
        if not self.client:
            return []
        
        try:
            # Build filter conditions
            filter_conditions = []
            
            if atom_type:
                filter_conditions.append(
                    FieldCondition(
                        key="atom_type",
                        match=MatchValue(value=atom_type.value)
                    )
                )
            
            if min_confidence > 0:
                filter_conditions.append(
                    FieldCondition(
                        key="confidence",
                        range={
                            "gte": min_confidence
                        }
                    )
                )
            
            # Create filter
            filter_obj = None
            if filter_conditions:
                filter_obj = Filter(must=filter_conditions)
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_obj,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append((
                    hit.id,
                    hit.score,
                    hit.payload
                ))
            
            self.logger.debug(f"Found {len(results)} similar atoms")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

    async def get_atom_neighbors(self, atom_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get atoms similar to a specific atom"""
        if not self.client:
            return []
        
        try:
            # Get the atom's vector
            atom_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[atom_id],
                with_vectors=True
            )
            
            if not atom_points:
                return []
            
            atom_vector = atom_points[0].vector
            
            # Search for similar atoms (excluding self)
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=atom_vector,
                limit=limit + 1,  # +1 because result includes the query atom
                score_threshold=0.3
            )
            
            # Filter out the original atom and format results
            neighbors = []
            for hit in search_result:
                if hit.id != atom_id:  # Exclude self
                    neighbors.append((hit.id, hit.score))
            
            return neighbors[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get neighbors for {atom_id}: {e}")
            return []

    async def delete_atom_vector(self, atom_id: str) -> bool:
        """Delete atom vector from store"""
        if not self.client:
            return False
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[atom_id]
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete atom vector {atom_id}: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get vector collection statistics"""
        if not self.client:
            return {"error": "Qdrant not available"}
        
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "total_vectors": info.points_count,
                "vector_dimension": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.value,
                "indexed_vectors": info.indexed_vectors_count,
                "collection_status": info.status.value
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}


class VectorKnowledgeFabric(KnowledgeFabric):
    """Enhanced knowledge fabric with vector search capabilities"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 database_url: str = "sqlite+aiosqlite:///./xorb_knowledge.db",
                 vector_host: str = "localhost",
                 vector_port: int = 6333):
        
        super().__init__(redis_url, database_url)
        
        self.embedding_service = VectorEmbeddingService()
        self.vector_store = VectorStore(vector_host, vector_port)
        
        # Enhanced search capabilities
        self.semantic_search_enabled = QDRANT_AVAILABLE and EMBEDDINGS_AVAILABLE

    async def initialize(self):
        """Initialize the enhanced knowledge fabric"""
        await super().initialize()
        
        if self.semantic_search_enabled:
            self.logger.info("Vector knowledge fabric initialized with semantic search")
        else:
            self.logger.warning("Vector search disabled - missing dependencies")

    async def add_atom(self, atom: KnowledgeAtom) -> str:
        """Add atom with vector embedding"""
        # Add to regular storage first
        atom_id = await super().add_atom(atom)
        
        # Generate and store vector embedding
        if self.semantic_search_enabled:
            try:
                # Create text representation for embedding
                text_content = self._atom_to_text(atom)
                
                # Generate embedding
                embedding = await self.embedding_service.encode_text(text_content)
                
                # Store in vector database
                await self.vector_store.add_atom_vector(atom, embedding)
                
                self.logger.debug(f"Added vector embedding for atom {atom_id}")
                
            except Exception as e:
                self.logger.warning(f"Failed to add vector for atom {atom_id}: {e}")
        
        return atom_id

    async def semantic_search(self, 
                            query: str,
                            limit: int = 10,
                            atom_type: Optional[AtomType] = None,
                            min_confidence: float = 0.0,
                            tags: Optional[List[str]] = None,
                            score_threshold: float = 0.7) -> List[KnowledgeAtom]:
        """Search atoms using semantic similarity"""
        
        if not self.semantic_search_enabled:
            self.logger.warning("Semantic search not available, falling back to regular search")
            return await self.search_atoms(query, atom_type, tags, min_confidence, limit)
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.encode_text(query)
            
            # Search for similar vectors
            vector_results = await self.vector_store.search_similar(
                query_embedding,
                limit=limit,
                atom_type=atom_type,
                min_confidence=min_confidence,
                tags=tags,
                score_threshold=score_threshold
            )
            
            # Retrieve full atoms
            atoms = []
            for atom_id, similarity_score, payload in vector_results:
                atom = await self.get_atom(atom_id)
                if atom:
                    # Enhance atom with similarity score
                    atom.metadata = getattr(atom, 'metadata', {})
                    atom.metadata['similarity_score'] = similarity_score
                    atoms.append(atom)
            
            self.logger.info(f"Semantic search for '{query}' found {len(atoms)} results")
            return atoms
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            # Fallback to regular search
            return await self.search_atoms(query, atom_type, tags, min_confidence, limit)

    async def find_related_atoms(self, atom: KnowledgeAtom, limit: int = 5) -> List[Tuple[KnowledgeAtom, float]]:
        """Find atoms semantically related to the given atom"""
        if not self.semantic_search_enabled:
            return []
        
        try:
            # Get similar atom IDs with scores
            neighbors = await self.vector_store.get_atom_neighbors(atom.id, limit)
            
            # Retrieve full atoms
            related_atoms = []
            for neighbor_id, similarity_score in neighbors:
                neighbor_atom = await self.get_atom(neighbor_id)
                if neighbor_atom:
                    related_atoms.append((neighbor_atom, similarity_score))
            
            return related_atoms
            
        except Exception as e:
            self.logger.error(f"Failed to find related atoms for {atom.id}: {e}")
            return []

    async def cluster_atoms(self, 
                          atom_type: Optional[AtomType] = None,
                          min_cluster_size: int = 3) -> Dict[str, List[KnowledgeAtom]]:
        """Cluster atoms by semantic similarity"""
        if not self.semantic_search_enabled:
            return {}
        
        try:
            # Get atoms to cluster
            if atom_type:
                atoms = await self.search_atoms(atom_type=atom_type, max_results=1000)
            else:
                atoms = await self.get_high_value_atoms(limit=500)
            
            if len(atoms) < min_cluster_size:
                return {}
            
            # Generate embeddings for all atoms
            texts = [self._atom_to_text(atom) for atom in atoms]
            embeddings = await self.embedding_service.encode_batch(texts)
            
            # Simple clustering using similarity threshold
            clusters = {}
            processed = set()
            
            for i, atom in enumerate(atoms):
                if atom.id in processed:
                    continue
                
                # Start new cluster
                cluster_id = f"cluster_{len(clusters)}"
                cluster_atoms = [atom]
                processed.add(atom.id)
                
                # Find similar atoms for this cluster
                for j, other_atom in enumerate(atoms):
                    if i != j and other_atom.id not in processed:
                        similarity = self.embedding_service.calculate_similarity(
                            embeddings[i], embeddings[j]
                        )
                        
                        if similarity > 0.8:  # High similarity threshold
                            cluster_atoms.append(other_atom)
                            processed.add(other_atom.id)
                
                # Only keep clusters with minimum size
                if len(cluster_atoms) >= min_cluster_size:
                    clusters[cluster_id] = cluster_atoms
            
            self.logger.info(f"Created {len(clusters)} semantic clusters")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {}

    async def suggest_knowledge_gaps(self, recent_campaigns: List[str]) -> List[Dict[str, Any]]:
        """Analyze recent campaigns to suggest knowledge gaps"""
        try:
            # Get recent findings from campaigns
            recent_findings = []
            for campaign_id in recent_campaigns:
                # Get campaign findings (simplified - would query campaign results)
                findings = await self._get_campaign_findings(campaign_id)
                recent_findings.extend(findings)
            
            if not recent_findings:
                return []
            
            # Analyze what types of knowledge are missing
            gaps = []
            
            # Find techniques that failed
            failed_techniques = [f for f in recent_findings if not f.get('success', True)]
            
            if failed_techniques and self.semantic_search_enabled:
                for failed in failed_techniques[:5]:  # Analyze top 5 failures
                    # Search for alternative techniques
                    similar_atoms = await self.semantic_search(
                        failed.get('description', ''),
                        atom_type=AtomType.TECHNIQUE,
                        limit=3
                    )
                    
                    if len(similar_atoms) < 2:  # Few alternatives available
                        gaps.append({
                            'gap_type': 'technique_alternatives',
                            'description': f"Limited alternatives for: {failed.get('title', 'Unknown')}",
                            'severity': 'medium',
                            'suggested_research': f"Research more techniques for {failed.get('category', 'general')} testing"
                        })
            
            # Find underrepresented attack vectors
            attack_vectors = {}
            for atom in await self.search_atoms(atom_type=AtomType.TECHNIQUE, max_results=200):
                vector_type = atom.content.get('attack_vector', 'unknown')
                attack_vectors[vector_type] = attack_vectors.get(vector_type, 0) + 1
            
            # Identify sparse areas
            avg_count = sum(attack_vectors.values()) / len(attack_vectors) if attack_vectors else 0
            for vector_type, count in attack_vectors.items():
                if count < avg_count * 0.5:  # Less than half average
                    gaps.append({
                        'gap_type': 'attack_vector_coverage',
                        'description': f"Low coverage for {vector_type} attack vector",
                        'severity': 'low',
                        'suggested_research': f"Expand {vector_type} techniques and payloads"
                    })
            
            return gaps[:10]  # Return top 10 gaps
            
        except Exception as e:
            self.logger.error(f"Failed to suggest knowledge gaps: {e}")
            return []

    def _atom_to_text(self, atom: KnowledgeAtom) -> str:
        """Convert atom to text representation for embedding"""
        text_parts = [atom.title]
        
        # Add content description
        if 'description' in atom.content:
            text_parts.append(atom.content['description'])
        
        # Add attack vectors or techniques
        for key in ['attack_vectors', 'techniques', 'payloads']:
            if key in atom.content and isinstance(atom.content[key], list):
                text_parts.extend(atom.content[key])
        
        # Add tags
        if atom.tags:
            text_parts.append(' '.join(atom.tags))
        
        return ' '.join(text_parts)

    async def _get_campaign_findings(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Get findings from a campaign (mock implementation)"""
        # In production, this would query the campaign results
        return [
            {
                'title': 'SQL Injection Test',
                'description': 'Attempted SQL injection on login form',
                'success': True,
                'category': 'web_app'
            },
            {
                'title': 'XSS Payload Test',
                'description': 'Cross-site scripting payload failed',
                'success': False,
                'category': 'web_app'
            }
        ]

    async def get_vector_fabric_stats(self) -> Dict[str, Any]:
        """Get enhanced knowledge fabric statistics"""
        base_stats = await self.analyze_knowledge_gaps()
        
        vector_stats = {}
        if self.semantic_search_enabled:
            vector_stats = await self.vector_store.get_collection_stats()
        
        return {
            **base_stats,
            'semantic_search_enabled': self.semantic_search_enabled,
            'embedding_model': self.embedding_service.model_name,
            'embedding_dimension': self.embedding_service.embedding_dimension,
            'vector_stats': vector_stats
        }


# Utility functions

async def migrate_to_vector_fabric(old_fabric: KnowledgeFabric, new_fabric: VectorKnowledgeFabric):
    """Migrate existing knowledge fabric to vector-enhanced version"""
    logging.info("Starting migration to vector knowledge fabric")
    
    # Get all atoms from old fabric
    atoms = await old_fabric.get_high_value_atoms(limit=10000)  # Get all atoms
    
    migrated_count = 0
    failed_count = 0
    
    for atom in atoms:
        try:
            await new_fabric.add_atom(atom)
            migrated_count += 1
            
            if migrated_count % 100 == 0:
                logging.info(f"Migrated {migrated_count} atoms...")
                
        except Exception as e:
            logging.error(f"Failed to migrate atom {atom.id}: {e}")
            failed_count += 1
    
    logging.info(f"Migration completed: {migrated_count} succeeded, {failed_count} failed")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_vector_search():
        """Demo vector search capabilities"""
        fabric = VectorKnowledgeFabric()
        await fabric.initialize()
        
        # Add some sample atoms
        sample_atoms = [
            KnowledgeAtom(
                id="",
                atom_type=AtomType.VULNERABILITY,
                title="SQL Injection in Login Form",
                content={
                    "description": "Classic SQL injection vulnerability in authentication",
                    "attack_vectors": ["web application", "database"],
                    "severity": "high"
                },
                tags={"sql_injection", "web_app", "authentication"}
            ),
            KnowledgeAtom(
                id="",
                atom_type=AtomType.TECHNIQUE,
                title="XSS via Reflected Input",
                content={
                    "description": "Cross-site scripting through reflected user input",
                    "attack_vectors": ["web application", "client-side"],
                    "payloads": ["<script>alert(1)</script>", "javascript:alert(1)"]
                },
                tags={"xss", "web_app", "reflected"}
            ),
            KnowledgeAtom(
                id="",
                atom_type=AtomType.PAYLOAD,
                title="NoSQL Injection Payloads",
                content={
                    "description": "Payloads for testing NoSQL database injection",
                    "payloads": ["{'$ne': null}", "';return 'a'=='a' && ''=='"],
                    "attack_vectors": ["nosql", "database"]
                },
                tags={"nosql", "injection", "database"}
            )
        ]
        
        # Add atoms
        for atom in sample_atoms:
            atom_id = await fabric.add_atom(atom)
            print(f"Added atom: {atom_id}")
        
        # Test semantic search
        print("\n=== Semantic Search Demo ===")
        
        search_queries = [
            "database injection vulnerabilities",
            "client side scripting attacks",
            "authentication bypass techniques"
        ]
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            results = await fabric.semantic_search(query, limit=3)
            
            for atom in results:
                similarity = atom.metadata.get('similarity_score', 0)
                print(f"  - {atom.title} (similarity: {similarity:.3f})")
        
        # Test clustering
        print("\n=== Clustering Demo ===")
        clusters = await fabric.cluster_atoms(min_cluster_size=2)
        
        for cluster_id, cluster_atoms in clusters.items():
            print(f"\n{cluster_id}: {len(cluster_atoms)} atoms")
            for atom in cluster_atoms:
                print(f"  - {atom.title}")
        
        # Show stats
        print("\n=== Vector Fabric Stats ===")
        stats = await fabric.get_vector_fabric_stats()
        print(json.dumps(stats, indent=2))
        
        await fabric.shutdown()
    
    if "--demo" in sys.argv:
        asyncio.run(demo_vector_search())
    elif "--install-deps" in sys.argv:
        print("To install vector search dependencies:")
        print("pip install qdrant-client sentence-transformers torch")
        print("docker run -p 6333:6333 qdrant/qdrant")
    else:
        print("Vector Knowledge Fabric")
        print("Usage:")
        print("  python vector_fabric.py --demo        # Run demo")
        print("  python vector_fabric.py --install-deps # Show installation instructions")