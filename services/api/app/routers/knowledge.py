"""
Knowledge Fabric API Router
Provides semantic knowledge management with NVIDIA embeddings
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set
from enum import Enum
import time
from prometheus_client import Counter, Histogram
import structlog

from ..deps import has_role
from xorb_core.logging import get_logger
from xorb_core.knowledge_fabric.enhanced_atoms import (
    get_semantic_fabric, 
    SemanticAtom, 
    SemanticCluster,
    AtomType
)

# Initialize logger
log = get_logger(__name__)

# Prometheus metrics
knowledge_operations_total = Counter(
    'xorb_knowledge_operations_total',
    'Total knowledge fabric operations',
    ['operation', 'status']
)
knowledge_operation_duration_seconds = Histogram(
    'xorb_knowledge_operation_duration_seconds',
    'Time spent on knowledge operations',
    ['operation']
)

router = APIRouter()

# Pydantic models
class AtomTypeEnum(str, Enum):
    VULNERABILITY = "vulnerability"
    TARGET = "target"
    ASSET = "asset"
    TECHNIQUE = "technique"
    INDICATOR = "indicator"
    INTELLIGENCE = "intelligence"
    EMBEDDING = "embedding"
    SEMANTIC_CLUSTER = "semantic_cluster"
    KNOWLEDGE_GRAPH = "knowledge_graph"

class CreateAtomRequest(BaseModel):
    content: str = Field(..., description="Content of the knowledge atom")
    atom_type: AtomTypeEnum = Field(..., description="Type of the atom")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    tags: Set[str] = Field(default_factory=set, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    generate_embedding: bool = Field(default=True, description="Generate embedding for the atom")

class AtomResponse(BaseModel):
    id: str
    atom_type: str
    content: str
    confidence: float
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    has_embedding: bool
    embedding_model: Optional[str]
    semantic_clusters: List[str]
    similar_atoms_count: int

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    atom_types: Optional[List[AtomTypeEnum]] = Field(default=None, description="Filter by atom types")

class SearchResult(BaseModel):
    atom: AtomResponse
    similarity_score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_seconds: float

class ClusterRequest(BaseModel):
    num_clusters: int = Field(default=5, ge=2, le=20, description="Number of clusters to create")
    atom_types: Optional[List[AtomTypeEnum]] = Field(default=None, description="Filter by atom types")

class ClusterResponse(BaseModel):
    id: str
    name: str
    description: str
    atom_count: int
    centroid_available: bool
    created_at: str

class SimilarityRequest(BaseModel):
    atom_id1: str = Field(..., description="First atom ID")
    atom_id2: str = Field(..., description="Second atom ID")
    metric: str = Field(default="cosine", description="Similarity metric")

class FabricStatsResponse(BaseModel):
    total_atoms: int
    atoms_with_embeddings: int
    embedding_coverage: float
    total_clusters: int
    atom_type_distribution: Dict[str, int]
    cache_stats: Dict[str, Any]

# Initialize semantic fabric
semantic_fabric = get_semantic_fabric()

@router.post("/atoms", response_model=AtomResponse)
async def create_atom(
    request: CreateAtomRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(has_role("user"))
):
    """Create a new semantic knowledge atom"""
    
    start_time = time.time()
    
    try:
        log.info("Creating semantic atom", 
                content_length=len(request.content),
                atom_type=request.atom_type.value,
                generate_embedding=request.generate_embedding)
        
        # Convert enum to AtomType
        atom_type = AtomType(request.atom_type.value)
        
        atom = await semantic_fabric.add_atom(
            content=request.content,
            atom_type=atom_type,
            confidence=request.confidence,
            tags=request.tags,
            metadata=request.metadata,
            generate_embedding=request.generate_embedding
        )
        
        # Record metrics
        duration = time.time() - start_time
        knowledge_operations_total.labels(operation="create_atom", status="success").inc()
        knowledge_operation_duration_seconds.labels(operation="create_atom").observe(duration)
        
        # Background task for similar atom discovery
        def find_similar_atoms():
            if atom.embedding:
                try:
                    similar_atoms = semantic_fabric.find_similar_atoms(
                        atom, top_k=5, threshold=0.8
                    )
                    for similar_atom, score in similar_atoms:
                        atom.add_similar_atom(similar_atom.id, score)
                        similar_atom.add_similar_atom(atom.id, score)
                    
                    log.info("Similar atoms discovered",
                            atom_id=atom.id,
                            similar_count=len(similar_atoms))
                except Exception as e:
                    log.error("Failed to discover similar atoms", 
                             atom_id=atom.id, error=str(e))
        
        background_tasks.add_task(find_similar_atoms)
        
        response = AtomResponse(
            id=atom.id,
            atom_type=atom.atom_type.value,
            content=atom.content,
            confidence=atom.confidence,
            tags=list(atom.tags),
            metadata=atom.metadata,
            created_at=atom.created_at.isoformat(),
            updated_at=atom.updated_at.isoformat(),
            has_embedding=bool(atom.embedding),
            embedding_model=atom.embedding_model,
            semantic_clusters=list(atom.semantic_clusters),
            similar_atoms_count=len(atom.similar_atoms)
        )
        
        log.info("Semantic atom created successfully",
                atom_id=atom.id,
                duration=duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        knowledge_operations_total.labels(operation="create_atom", status="error").inc()
        
        log.error("Failed to create semantic atom", error=str(e), duration=duration)
        raise HTTPException(status_code=500, detail=f"Failed to create atom: {str(e)}")

@router.get("/atoms/{atom_id}", response_model=AtomResponse)
async def get_atom(
    atom_id: str,
    current_user: dict = Depends(has_role("user"))
):
    """Get a specific semantic atom by ID"""
    
    if atom_id not in semantic_fabric.atoms:
        raise HTTPException(status_code=404, detail="Atom not found")
    
    atom = semantic_fabric.atoms[atom_id]
    
    return AtomResponse(
        id=atom.id,
        atom_type=atom.atom_type.value,
        content=atom.content,
        confidence=atom.confidence,
        tags=list(atom.tags),
        metadata=atom.metadata,
        created_at=atom.created_at.isoformat(),
        updated_at=atom.updated_at.isoformat(),
        has_embedding=bool(atom.embedding),
        embedding_model=atom.embedding_model,
        semantic_clusters=list(atom.semantic_clusters),
        similar_atoms_count=len(atom.similar_atoms)
    )

@router.get("/atoms", response_model=List[AtomResponse])
async def list_atoms(
    atom_type: Optional[AtomTypeEnum] = None,
    has_embedding: Optional[bool] = None,
    limit: int = Field(default=50, ge=1, le=1000),
    offset: int = Field(default=0, ge=0),
    current_user: dict = Depends(has_role("user"))
):
    """List semantic atoms with optional filtering"""
    
    atoms = list(semantic_fabric.atoms.values())
    
    # Apply filters
    if atom_type:
        atoms = [atom for atom in atoms if atom.atom_type.value == atom_type.value]
    
    if has_embedding is not None:
        atoms = [atom for atom in atoms if bool(atom.embedding) == has_embedding]
    
    # Apply pagination
    total_atoms = len(atoms)
    atoms = atoms[offset:offset + limit]
    
    responses = []
    for atom in atoms:
        responses.append(AtomResponse(
            id=atom.id,
            atom_type=atom.atom_type.value,
            content=atom.content,
            confidence=atom.confidence,
            tags=list(atom.tags),
            metadata=atom.metadata,
            created_at=atom.created_at.isoformat(),
            updated_at=atom.updated_at.isoformat(),
            has_embedding=bool(atom.embedding),
            embedding_model=atom.embedding_model,
            semantic_clusters=list(atom.semantic_clusters),
            similar_atoms_count=len(atom.similar_atoms)
        ))
    
    return responses

@router.post("/search", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    current_user: dict = Depends(has_role("user"))
):
    """Perform semantic search over the knowledge fabric"""
    
    start_time = time.time()
    
    try:
        log.info("Performing semantic search",
                query_length=len(request.query),
                top_k=request.top_k,
                threshold=request.threshold)
        
        # Convert enum atom types
        atom_types = None
        if request.atom_types:
            atom_types = [AtomType(at.value) for at in request.atom_types]
        
        # Perform search
        results = await semantic_fabric.semantic_search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            atom_types=atom_types
        )
        
        # Convert results
        search_results = []
        for atom, similarity in results:
            atom_response = AtomResponse(
                id=atom.id,
                atom_type=atom.atom_type.value,
                content=atom.content,
                confidence=atom.confidence,
                tags=list(atom.tags),
                metadata=atom.metadata,
                created_at=atom.created_at.isoformat(),
                updated_at=atom.updated_at.isoformat(),
                has_embedding=bool(atom.embedding),
                embedding_model=atom.embedding_model,
                semantic_clusters=list(atom.semantic_clusters),
                similar_atoms_count=len(atom.similar_atoms)
            )
            
            search_results.append(SearchResult(
                atom=atom_response,
                similarity_score=similarity
            ))
        
        duration = time.time() - start_time
        knowledge_operations_total.labels(operation="semantic_search", status="success").inc()
        knowledge_operation_duration_seconds.labels(operation="semantic_search").observe(duration)
        
        response = SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_seconds=duration
        )
        
        log.info("Semantic search completed",
                results_found=len(search_results),
                duration=duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        knowledge_operations_total.labels(operation="semantic_search", status="error").inc()
        
        log.error("Semantic search failed", error=str(e), duration=duration)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/clusters", response_model=List[ClusterResponse])
async def create_clusters(
    request: ClusterRequest,
    current_user: dict = Depends(has_role("user"))
):
    """Create semantic clusters from atoms"""
    
    start_time = time.time()
    
    try:
        log.info("Creating semantic clusters",
                num_clusters=request.num_clusters,
                atom_types=request.atom_types)
        
        # Convert enum atom types
        atom_types = None
        if request.atom_types:
            atom_types = [AtomType(at.value) for at in request.atom_types]
        
        clusters = await semantic_fabric.create_semantic_clusters(
            num_clusters=request.num_clusters,
            atom_types=atom_types
        )
        
        # Convert to response format
        cluster_responses = []
        for cluster in clusters:
            cluster_responses.append(ClusterResponse(
                id=cluster.id,
                name=cluster.name,
                description=cluster.description,
                atom_count=len(cluster.atom_ids),
                centroid_available=bool(cluster.centroid_embedding),
                created_at=cluster.created_at.isoformat()
            ))
        
        duration = time.time() - start_time
        knowledge_operations_total.labels(operation="create_clusters", status="success").inc()
        knowledge_operation_duration_seconds.labels(operation="create_clusters").observe(duration)
        
        log.info("Semantic clusters created",
                clusters_created=len(clusters),
                duration=duration)
        
        return cluster_responses
        
    except Exception as e:
        duration = time.time() - start_time
        knowledge_operations_total.labels(operation="create_clusters", status="error").inc()
        
        log.error("Cluster creation failed", error=str(e), duration=duration)
        raise HTTPException(status_code=500, detail=f"Cluster creation failed: {str(e)}")

@router.get("/clusters/{cluster_id}/atoms", response_model=List[AtomResponse])
async def get_cluster_atoms(
    cluster_id: str,
    current_user: dict = Depends(has_role("user"))
):
    """Get all atoms in a specific cluster"""
    
    if cluster_id not in semantic_fabric.clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    atoms = semantic_fabric.get_cluster_atoms(cluster_id)
    
    responses = []
    for atom in atoms:
        responses.append(AtomResponse(
            id=atom.id,
            atom_type=atom.atom_type.value,
            content=atom.content,
            confidence=atom.confidence,
            tags=list(atom.tags),
            metadata=atom.metadata,
            created_at=atom.created_at.isoformat(),
            updated_at=atom.updated_at.isoformat(),
            has_embedding=bool(atom.embedding),
            embedding_model=atom.embedding_model,
            semantic_clusters=list(atom.semantic_clusters),
            similar_atoms_count=len(atom.similar_atoms)
        ))
    
    return responses

@router.post("/similarity")
async def compute_atom_similarity(
    request: SimilarityRequest,
    current_user: dict = Depends(has_role("user"))
):
    """Compute similarity between two atoms"""
    
    if request.atom_id1 not in semantic_fabric.atoms:
        raise HTTPException(status_code=404, detail="First atom not found")
    
    if request.atom_id2 not in semantic_fabric.atoms:
        raise HTTPException(status_code=404, detail="Second atom not found")
    
    atom1 = semantic_fabric.atoms[request.atom_id1]
    atom2 = semantic_fabric.atoms[request.atom_id2]
    
    if not atom1.embedding or not atom2.embedding:
        raise HTTPException(
            status_code=400,
            detail="Both atoms must have embeddings for similarity computation"
        )
    
    similarity = atom1.compute_similarity(atom2, metric=request.metric)
    
    return {
        "atom_id1": request.atom_id1,
        "atom_id2": request.atom_id2,
        "similarity_score": similarity,
        "metric": request.metric,
        "atom1_content": atom1.content[:100] + "..." if len(atom1.content) > 100 else atom1.content,
        "atom2_content": atom2.content[:100] + "..." if len(atom2.content) > 100 else atom2.content
    }

@router.get("/stats", response_model=FabricStatsResponse)
async def get_fabric_stats(
    current_user: dict = Depends(has_role("user"))
):
    """Get knowledge fabric statistics"""
    
    stats = semantic_fabric.get_fabric_stats()
    
    return FabricStatsResponse(
        total_atoms=stats["total_atoms"],
        atoms_with_embeddings=stats["atoms_with_embeddings"],
        embedding_coverage=stats["embedding_coverage"],
        total_clusters=stats["total_clusters"],
        atom_type_distribution=stats["atom_type_distribution"],
        cache_stats=stats["cache_stats"]
    )

@router.post("/atoms/{atom_id}/embedding")
async def generate_atom_embedding(
    atom_id: str,
    force_regenerate: bool = Field(default=False, description="Force regeneration if embedding exists"),
    current_user: dict = Depends(has_role("user"))
):
    """Generate or regenerate embedding for an atom"""
    
    if atom_id not in semantic_fabric.atoms:
        raise HTTPException(status_code=404, detail="Atom not found")
    
    atom = semantic_fabric.atoms[atom_id]
    
    try:
        success = await atom.generate_embedding(force_regenerate=force_regenerate)
        
        if success:
            knowledge_operations_total.labels(operation="generate_embedding", status="success").inc()
            return {
                "atom_id": atom_id,
                "embedding_generated": True,
                "embedding_model": atom.embedding_model,
                "embedding_dimension": len(atom.embedding) if atom.embedding else 0
            }
        else:
            knowledge_operations_total.labels(operation="generate_embedding", status="error").inc()
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
            
    except Exception as e:
        knowledge_operations_total.labels(operation="generate_embedding", status="error").inc()
        log.error("Embedding generation failed", atom_id=atom_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@router.delete("/atoms/{atom_id}")
async def delete_atom(
    atom_id: str,
    current_user: dict = Depends(has_role("admin"))
):
    """Delete a semantic atom"""
    
    if atom_id not in semantic_fabric.atoms:
        raise HTTPException(status_code=404, detail="Atom not found")
    
    # Remove from all clusters
    atom = semantic_fabric.atoms[atom_id]
    for cluster_id in list(atom.semantic_clusters):
        if cluster_id in semantic_fabric.clusters:
            semantic_fabric.clusters[cluster_id].remove_atom(atom_id)
    
    # Remove similar atom references
    for other_atom in semantic_fabric.atoms.values():
        other_atom.similar_atoms = [
            (aid, score) for aid, score in other_atom.similar_atoms
            if aid != atom_id
        ]
    
    # Delete the atom
    del semantic_fabric.atoms[atom_id]
    
    knowledge_operations_total.labels(operation="delete_atom", status="success").inc()
    
    log.info("Semantic atom deleted", atom_id=atom_id)
    
    return {"message": "Atom deleted successfully", "atom_id": atom_id}