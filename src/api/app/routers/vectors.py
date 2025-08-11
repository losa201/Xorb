"""Vector search and similarity API routes."""
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..auth.dependencies import require_permissions
from ..auth.models import Permission, UserClaims
from ..middleware.tenant_context import require_tenant_context
from ..infrastructure.vector_store import get_vector_store
from ..security.input_validation import validate_pagination
from ..infrastructure.observability import get_metrics_collector, add_trace_context
import structlog
import time

logger = structlog.get_logger("vectors_api")
router = APIRouter(prefix="/api/vectors", tags=["Vector Search"])


class VectorAddRequest(BaseModel):
    """Request to add vector to store."""
    vector: List[float] = Field(..., min_items=1, max_items=2048)
    source_type: str = Field(..., min_length=1, max_length=50)
    source_id: UUID
    content_hash: str = Field(..., min_length=1, max_length=64)
    embedding_model: str = Field(..., min_length=1, max_length=100)
    metadata: Optional[Dict] = Field(default_factory=dict)


class VectorSearchRequest(BaseModel):
    """Request for vector similarity search."""
    query_vector: List[float] = Field(..., min_items=1, max_items=2048)
    limit: int = Field(default=10, ge=1, le=100)
    source_type: Optional[str] = Field(default=None, max_length=50)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    ef_search: Optional[int] = Field(default=None, ge=1, le=1000)


class TextSearchRequest(BaseModel):
    """Request for text-based similarity search."""
    query_text: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    source_type: Optional[str] = Field(default=None, max_length=50)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


@router.post("/add", response_model=Dict)
async def add_vector(
    request: Request,
    vector_request: VectorAddRequest,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_WRITE))
):
    """Add vector to the vector store."""
    tenant_id = require_tenant_context(request)
    vector_store = get_vector_store()
    
    try:
        vector_id = await vector_store.add_vector(
            vector=vector_request.vector,
            tenant_id=tenant_id,
            source_type=vector_request.source_type,
            source_id=vector_request.source_id,
            content_hash=vector_request.content_hash,
            embedding_model=vector_request.embedding_model,
            metadata=vector_request.metadata
        )
        
        logger.info(
            "Vector added",
            vector_id=str(vector_id),
            tenant_id=str(tenant_id),
            source_type=vector_request.source_type,
            source_id=str(vector_request.source_id),
            user_id=current_user.sub
        )
        
        add_trace_context(
            operation="add_vector",
            vector_id=str(vector_id),
            source_type=vector_request.source_type,
            tenant_id=str(tenant_id)
        )
        
        return {
            "vector_id": str(vector_id),
            "status": "added",
            "tenant_id": str(tenant_id)
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to add vector",
            tenant_id=str(tenant_id),
            source_type=vector_request.source_type,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add vector"
        )


@router.post("/search", response_model=List[Dict])
async def search_similar_vectors(
    request: Request,
    search_request: VectorSearchRequest,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_READ))
):
    """Search for similar vectors."""
    tenant_id = require_tenant_context(request)
    vector_store = get_vector_store()
    
    start_time = time.time()
    
    try:
        results = await vector_store.search_similar(
            query_vector=search_request.query_vector,
            tenant_id=tenant_id,
            limit=search_request.limit,
            source_type=search_request.source_type,
            similarity_threshold=search_request.similarity_threshold,
            ef_search=search_request.ef_search
        )
        
        duration = time.time() - start_time
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_vector_search(duration, len(results))
        
        logger.info(
            "Vector search completed",
            tenant_id=str(tenant_id),
            result_count=len(results),
            duration_ms=round(duration * 1000, 2),
            similarity_threshold=search_request.similarity_threshold,
            user_id=current_user.sub
        )
        
        add_trace_context(
            operation="vector_search",
            result_count=len(results),
            duration_ms=duration * 1000,
            tenant_id=str(tenant_id)
        )
        
        return results
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Vector search failed",
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector search failed"
        )


@router.post("/search-text", response_model=List[Dict])
async def search_by_text(
    request: Request,
    search_request: TextSearchRequest,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_READ))
):
    """Search vectors by text query."""
    tenant_id = require_tenant_context(request)
    vector_store = get_vector_store()
    
    start_time = time.time()
    
    try:
        # Import embedding function (would be injected in production)
        from ..infrastructure.vector_store import openai_embedding_function
        
        results = await vector_store.search_by_text(
            query_text=search_request.query_text,
            tenant_id=tenant_id,
            embedding_function=openai_embedding_function,
            limit=search_request.limit,
            source_type=search_request.source_type,
            similarity_threshold=search_request.similarity_threshold
        )
        
        duration = time.time() - start_time
        
        logger.info(
            "Text search completed",
            tenant_id=str(tenant_id),
            query_text=search_request.query_text[:100],  # Truncate for logging
            result_count=len(results),
            duration_ms=round(duration * 1000, 2),
            user_id=current_user.sub
        )
        
        return results
        
    except Exception as e:
        logger.error(
            "Text search failed",
            tenant_id=str(tenant_id),
            query_text=search_request.query_text[:100],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text search failed"
        )


@router.get("/vector/{vector_id}", response_model=Dict)
async def get_vector(
    request: Request,
    vector_id: UUID,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_READ))
):
    """Get vector by ID."""
    tenant_id = require_tenant_context(request)
    vector_store = get_vector_store()
    
    try:
        vector_data = await vector_store.get_vector(vector_id, tenant_id)
        
        if not vector_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Vector not found"
            )
        
        return vector_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get vector",
            vector_id=str(vector_id),
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get vector"
        )


@router.delete("/vector/{vector_id}")
async def delete_vector(
    request: Request,
    vector_id: UUID,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_DELETE))
):
    """Delete vector by ID."""
    tenant_id = require_tenant_context(request)
    vector_store = get_vector_store()
    
    try:
        deleted = await vector_store.delete_vector(vector_id, tenant_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Vector not found"
            )
        
        logger.info(
            "Vector deleted",
            vector_id=str(vector_id),
            tenant_id=str(tenant_id),
            user_id=current_user.sub
        )
        
        return {"message": "Vector deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete vector",
            vector_id=str(vector_id),
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete vector"
        )


@router.delete("/source/{source_id}")
async def delete_vectors_by_source(
    request: Request,
    source_id: UUID,
    source_type: Optional[str] = None,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_DELETE))
):
    """Delete all vectors for a source."""
    tenant_id = require_tenant_context(request)
    vector_store = get_vector_store()
    
    try:
        deleted_count = await vector_store.delete_vectors_by_source(
            source_id=source_id,
            tenant_id=tenant_id,
            source_type=source_type
        )
        
        logger.info(
            "Vectors deleted by source",
            source_id=str(source_id),
            source_type=source_type,
            deleted_count=deleted_count,
            tenant_id=str(tenant_id),
            user_id=current_user.sub
        )
        
        return {
            "message": f"Deleted {deleted_count} vectors",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(
            "Failed to delete vectors by source",
            source_id=str(source_id),
            source_type=source_type,
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete vectors"
        )


@router.get("/stats", response_model=Dict)
async def get_vector_stats(
    request: Request,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_READ))
):
    """Get vector statistics for tenant."""
    tenant_id = require_tenant_context(request)
    vector_store = get_vector_store()
    
    try:
        stats = await vector_store.get_vector_stats(tenant_id)
        
        logger.info(
            "Retrieved vector stats",
            tenant_id=str(tenant_id),
            total_vectors=stats["total_vectors"],
            user_id=current_user.sub
        )
        
        return stats
        
    except Exception as e:
        logger.error(
            "Failed to get vector stats",
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get vector statistics"
        )


# Batch operations
@router.post("/batch-add", response_model=Dict)
async def batch_add_vectors(
    request: Request,
    vectors: List[VectorAddRequest],
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_WRITE))
):
    """Add multiple vectors in batch."""
    tenant_id = require_tenant_context(request)
    vector_store = get_vector_store()
    
    if len(vectors) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 100 vectors"
        )
    
    success_count = 0
    failed_count = 0
    results = []
    
    for i, vector_request in enumerate(vectors):
        try:
            vector_id = await vector_store.add_vector(
                vector=vector_request.vector,
                tenant_id=tenant_id,
                source_type=vector_request.source_type,
                source_id=vector_request.source_id,
                content_hash=vector_request.content_hash,
                embedding_model=vector_request.embedding_model,
                metadata=vector_request.metadata
            )
            
            results.append({
                "index": i,
                "vector_id": str(vector_id),
                "status": "success"
            })
            success_count += 1
            
        except Exception as e:
            results.append({
                "index": i,
                "status": "failed",
                "error": str(e)
            })
            failed_count += 1
    
    logger.info(
        "Batch vector add completed",
        tenant_id=str(tenant_id),
        total_vectors=len(vectors),
        success_count=success_count,
        failed_count=failed_count,
        user_id=current_user.sub
    )
    
    return {
        "total": len(vectors),
        "success": success_count,
        "failed": failed_count,
        "results": results
    }