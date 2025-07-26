"""
Enhanced Knowledge Atoms with Embedding Support
Extends the base knowledge atoms with semantic embedding capabilities
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import hashlib

from .embedding_service import get_embedding_service, EmbeddingResult
from ..logging import get_logger

log = get_logger(__name__)

class AtomType(Enum):
    """Enhanced atom types with embedding-specific categories"""
    VULNERABILITY = "vulnerability"
    TARGET = "target"
    ASSET = "asset"
    TECHNIQUE = "technique"
    INDICATOR = "indicator"
    INTELLIGENCE = "intelligence"
    EMBEDDING = "embedding"  # New type for semantic content
    SEMANTIC_CLUSTER = "semantic_cluster"  # Groups of related content
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Graph relationships

@dataclass
class SemanticAtom:
    """Knowledge atom enhanced with embedding capabilities"""
    
    # Core atom properties
    id: str
    atom_type: AtomType
    content: str
    confidence: float = 0.8
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Embedding properties
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Semantic relationships
    semantic_clusters: Set[str] = field(default_factory=set)
    similar_atoms: List[tuple] = field(default_factory=list)  # [(atom_id, similarity_score)]
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.id:
            # Generate ID from content hash
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            self.id = f"{self.atom_type.value}_{content_hash}"
    
    async def generate_embedding(
        self,
        force_regenerate: bool = False,
        input_type: str = "passage"
    ) -> bool:
        """Generate embedding for this atom's content"""
        
        if self.embedding and not force_regenerate:
            log.debug("Embedding already exists for atom", atom_id=self.id)
            return True
        
        try:
            embedding_service = get_embedding_service()
            
            result = await embedding_service.embed_text(
                text=self.content,
                input_type=input_type,
                metadata={
                    "atom_id": self.id,
                    "atom_type": self.atom_type.value,
                    "tags": list(self.tags),
                    **self.metadata
                }
            )
            
            self.embedding = result.embedding
            self.embedding_model = result.model
            self.embedding_metadata = {
                "generated_at": result.timestamp.isoformat(),
                "input_type": input_type,
                "embedding_dimension": len(result.embedding),
                **result.metadata
            }
            self.updated_at = datetime.now(timezone.utc)
            
            log.info("Embedding generated for atom",
                    atom_id=self.id,
                    embedding_dimension=len(self.embedding),
                    model=self.embedding_model)
            
            return True
            
        except Exception as e:
            log.error("Failed to generate embedding for atom",
                     atom_id=self.id,
                     error=str(e))
            return False
    
    def compute_similarity(
        self,
        other: 'SemanticAtom',
        metric: str = "cosine"
    ) -> Optional[float]:
        """Compute semantic similarity with another atom"""
        
        if not self.embedding or not other.embedding:
            log.warning("Cannot compute similarity - missing embeddings",
                       self_has_embedding=bool(self.embedding),
                       other_has_embedding=bool(other.embedding))
            return None
        
        embedding_service = get_embedding_service()
        return embedding_service.compute_similarity(
            self.embedding,
            other.embedding,
            metric=metric
        )
    
    def add_similar_atom(
        self,
        atom_id: str,
        similarity_score: float,
        max_similar: int = 10
    ):
        """Add a similar atom reference"""
        
        # Remove existing reference if it exists
        self.similar_atoms = [
            (aid, score) for aid, score in self.similar_atoms
            if aid != atom_id
        ]
        
        # Add new reference
        self.similar_atoms.append((atom_id, similarity_score))
        
        # Sort by similarity (descending) and keep top N
        self.similar_atoms.sort(key=lambda x: x[1], reverse=True)
        self.similar_atoms = self.similar_atoms[:max_similar]
        
        self.updated_at = datetime.now(timezone.utc)
    
    def add_to_cluster(self, cluster_id: str):
        """Add this atom to a semantic cluster"""
        self.semantic_clusters.add(cluster_id)
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_from_cluster(self, cluster_id: str):
        """Remove this atom from a semantic cluster"""
        self.semantic_clusters.discard(cluster_id)
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary representation"""
        return {
            "id": self.id,
            "atom_type": self.atom_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
            "embedding_metadata": self.embedding_metadata,
            "semantic_clusters": list(self.semantic_clusters),
            "similar_atoms": self.similar_atoms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticAtom':
        """Create atom from dictionary representation"""
        
        return cls(
            id=data["id"],
            atom_type=AtomType(data["atom_type"]),
            content=data["content"],
            confidence=data.get("confidence", 0.8),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            embedding=data.get("embedding"),
            embedding_model=data.get("embedding_model"),
            embedding_metadata=data.get("embedding_metadata", {}),
            semantic_clusters=set(data.get("semantic_clusters", [])),
            similar_atoms=data.get("similar_atoms", [])
        )

@dataclass
class SemanticCluster:
    """Represents a cluster of semantically similar atoms"""
    
    id: str
    name: str
    description: str
    atom_ids: Set[str] = field(default_factory=set)
    centroid_embedding: Optional[List[float]] = None
    cluster_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_atom(self, atom_id: str):
        """Add an atom to this cluster"""
        self.atom_ids.add(atom_id)
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_atom(self, atom_id: str):
        """Remove an atom from this cluster"""
        self.atom_ids.discard(atom_id)
        self.updated_at = datetime.now(timezone.utc)
    
    async def compute_centroid(self, atoms: List[SemanticAtom]):
        """Compute the centroid embedding for this cluster"""
        
        cluster_atoms = [atom for atom in atoms if atom.id in self.atom_ids]
        
        if not cluster_atoms:
            log.warning("No atoms found for cluster centroid computation", cluster_id=self.id)
            return
        
        # Filter atoms with embeddings
        atoms_with_embeddings = [atom for atom in cluster_atoms if atom.embedding]
        
        if not atoms_with_embeddings:
            log.warning("No atoms with embeddings in cluster", cluster_id=self.id)
            return
        
        # Compute centroid as mean of embeddings
        import numpy as np
        
        embeddings = np.array([atom.embedding for atom in atoms_with_embeddings])
        centroid = np.mean(embeddings, axis=0)
        
        self.centroid_embedding = centroid.tolist()
        self.cluster_metadata.update({
            "atom_count": len(atoms_with_embeddings),
            "centroid_computed_at": datetime.now(timezone.utc).isoformat(),
            "embedding_dimension": len(self.centroid_embedding)
        })
        self.updated_at = datetime.now(timezone.utc)
        
        log.info("Cluster centroid computed",
                cluster_id=self.id,
                atom_count=len(atoms_with_embeddings),
                embedding_dimension=len(self.centroid_embedding))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "atom_ids": list(self.atom_ids),
            "centroid_embedding": self.centroid_embedding,
            "cluster_metadata": self.cluster_metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticCluster':
        """Create cluster from dictionary representation"""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            atom_ids=set(data.get("atom_ids", [])),
            centroid_embedding=data.get("centroid_embedding"),
            cluster_metadata=data.get("cluster_metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )

class SemanticKnowledgeFabric:
    """Enhanced knowledge fabric with semantic capabilities"""
    
    def __init__(self):
        self.atoms: Dict[str, SemanticAtom] = {}
        self.clusters: Dict[str, SemanticCluster] = {}
        self.embedding_service = get_embedding_service()
    
    async def add_atom(
        self,
        content: str,
        atom_type: AtomType,
        confidence: float = 0.8,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True
    ) -> SemanticAtom:
        """Add a new semantic atom to the fabric"""
        
        atom = SemanticAtom(
            id="",  # Will be generated in __post_init__
            atom_type=atom_type,
            content=content,
            confidence=confidence,
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        if generate_embedding:
            await atom.generate_embedding()
        
        self.atoms[atom.id] = atom
        
        log.info("Semantic atom added to fabric",
                atom_id=atom.id,
                atom_type=atom_type.value,
                has_embedding=bool(atom.embedding))
        
        return atom
    
    async def find_similar_atoms(
        self,
        query_atom: SemanticAtom,
        top_k: int = 10,
        threshold: float = 0.7,
        atom_types: Optional[List[AtomType]] = None
    ) -> List[tuple]:
        """Find atoms similar to the query atom"""
        
        if not query_atom.embedding:
            log.warning("Query atom has no embedding", atom_id=query_atom.id)
            return []
        
        similar_atoms = []
        
        for atom in self.atoms.values():
            # Skip self
            if atom.id == query_atom.id:
                continue
            
            # Filter by type if specified
            if atom_types and atom.atom_type not in atom_types:
                continue
            
            # Skip atoms without embeddings
            if not atom.embedding:
                continue
            
            # Compute similarity
            similarity = self.embedding_service.compute_similarity(
                query_atom.embedding,
                atom.embedding
            )
            
            if similarity >= threshold:
                similar_atoms.append((atom, similarity))
        
        # Sort by similarity (descending) and return top_k
        similar_atoms.sort(key=lambda x: x[1], reverse=True)
        return similar_atoms[:top_k]
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.7,
        atom_types: Optional[List[AtomType]] = None
    ) -> List[tuple]:
        """Perform semantic search over the knowledge fabric"""
        
        # Generate embedding for query
        result = await self.embedding_service.embed_text(query, input_type="query")
        
        similar_atoms = []
        
        for atom in self.atoms.values():
            # Filter by type if specified
            if atom_types and atom.atom_type not in atom_types:
                continue
            
            # Skip atoms without embeddings
            if not atom.embedding:
                continue
            
            # Compute similarity
            similarity = self.embedding_service.compute_similarity(
                result.embedding,
                atom.embedding
            )
            
            if similarity >= threshold:
                similar_atoms.append((atom, similarity))
        
        # Sort by similarity (descending) and return top_k
        similar_atoms.sort(key=lambda x: x[1], reverse=True)
        
        log.info("Semantic search completed",
                query_length=len(query),
                results_found=len(similar_atoms),
                top_similarity=similar_atoms[0][1] if similar_atoms else 0.0)
        
        return similar_atoms[:top_k]
    
    async def create_semantic_clusters(
        self,
        num_clusters: int = 5,
        atom_types: Optional[List[AtomType]] = None
    ) -> List[SemanticCluster]:
        """Create semantic clusters from atoms"""
        
        # Filter atoms
        target_atoms = []
        for atom in self.atoms.values():
            if atom_types and atom.atom_type not in atom_types:
                continue
            if not atom.embedding:
                continue
            target_atoms.append(atom)
        
        if len(target_atoms) < num_clusters:
            log.warning("Not enough atoms for clustering",
                       atom_count=len(target_atoms),
                       requested_clusters=num_clusters)
            return []
        
        # Extract texts and generate clusters
        texts = [atom.content for atom in target_atoms]
        text_clusters = await self.embedding_service.cluster_texts(
            texts, num_clusters=num_clusters
        )
        
        # Create SemanticCluster objects
        clusters = []
        for cluster_id, cluster_texts in text_clusters.items():
            cluster = SemanticCluster(
                id=f"cluster_{cluster_id}",
                name=f"Semantic Cluster {cluster_id}",
                description=f"Automatically generated cluster containing {len(cluster_texts)} atoms"
            )
            
            # Add atoms to cluster
            for text in cluster_texts:
                # Find atom with this text
                for atom in target_atoms:
                    if atom.content == text:
                        cluster.add_atom(atom.id)
                        atom.add_to_cluster(cluster.id)
                        break
            
            # Compute centroid
            await cluster.compute_centroid(target_atoms)
            
            clusters.append(cluster)
            self.clusters[cluster.id] = cluster
        
        log.info("Semantic clusters created",
                num_clusters=len(clusters),
                total_atoms_clustered=sum(len(c.atom_ids) for c in clusters))
        
        return clusters
    
    def get_cluster_atoms(self, cluster_id: str) -> List[SemanticAtom]:
        """Get all atoms in a specific cluster"""
        
        if cluster_id not in self.clusters:
            return []
        
        cluster = self.clusters[cluster_id]
        return [self.atoms[atom_id] for atom_id in cluster.atom_ids if atom_id in self.atoms]
    
    def get_fabric_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic knowledge fabric"""
        
        atoms_with_embeddings = sum(1 for atom in self.atoms.values() if atom.embedding)
        
        atom_type_counts = {}
        for atom in self.atoms.values():
            atom_type = atom.atom_type.value
            atom_type_counts[atom_type] = atom_type_counts.get(atom_type, 0) + 1
        
        return {
            "total_atoms": len(self.atoms),
            "atoms_with_embeddings": atoms_with_embeddings,
            "embedding_coverage": atoms_with_embeddings / len(self.atoms) if self.atoms else 0,
            "total_clusters": len(self.clusters),
            "atom_type_distribution": atom_type_counts,
            "cache_stats": self.embedding_service.get_cache_stats()
        }

# Global semantic knowledge fabric instance
_semantic_fabric = None

def get_semantic_fabric() -> SemanticKnowledgeFabric:
    """Get the global semantic knowledge fabric instance"""
    global _semantic_fabric
    
    if _semantic_fabric is None:
        _semantic_fabric = SemanticKnowledgeFabric()
    
    return _semantic_fabric