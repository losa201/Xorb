"""
NVIDIA Embeddings API Router (Refactored for Clean Architecture)
Provides embedding generation using NVIDIA's embed-qa-4 model
"""

import time
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

from ..container import get_container
from ..services.interfaces import EmbeddingService
from ..domain.exceptions import DomainException
from ..dependencies import get_current_user, get_current_organization

# Try multiple import paths for compatibility
try:
    from packages.xorb_core.xorb_core.logging import get_logger
except ImportError:
    try:
        from xorb_core.logging import get_logger
    except ImportError:
        # Fallback logging
        import logging
        def get_logger(name):
            return logging.getLogger(name)

# Initialize logger
log = get_logger(__name__)

# Prometheus metrics
embedding_requests_total = Counter(
    'xorb_embedding_requests_total',
    'Total embedding requests',
    ['model', 'status']
)
embedding_duration_seconds = Histogram(
    'xorb_embedding_duration_seconds',
    'Time spent generating embeddings',
    ['model']
)
embedding_tokens_total = Counter(
    'xorb_embedding_tokens_total',
    'Total tokens processed for embeddings',
    ['model', 'input_type']
)
embedding_cache_hits_total = Counter(
    'xorb_embedding_cache_hits_total',
    'Total embedding cache hits',
    ['cache_type', 'model']
)

router = APIRouter()

# Pydantic models
class EmbeddingRequest(BaseModel):
    input: list[str] = Field(..., description="List of texts to embed", max_items=100)
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use")
    input_type: str | None = Field(default="query", description="Type of input text")
    truncate: str | None = Field(default="NONE", description="Truncation strategy")
    encoding_format: str = Field(default="float", description="Encoding format for embeddings")

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int]

# Service is now injected via dependency injection - no need for global instance

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    current_org=Depends(get_current_organization)
):
    """
    Generate embeddings for input texts using NVIDIA's embed-qa-4 model

    - **input**: List of texts to embed (max 100 items)
    - **model**: Embedding model to use (default: nvidia/embed-qa-4)
    - **input_type**: Type of input text (query, passage, classification, etc.)
    - **encoding_format**: Format for embeddings (float, base64)
    """

    try:
        container = get_container()
        embedding_service = container.get(EmbeddingService)

        # Generate embeddings using service
        result = await embedding_service.generate_embeddings(
            texts=request.input,
            model=request.model,
            input_type=request.input_type or "query",
            user=current_user,
            org=current_org
        )

        # Convert domain result to API response
        embedding_data = []
        for i, embedding in enumerate(result.embeddings):
            embedding_data.append(EmbeddingData(
                embedding=embedding,
                index=i
            ))

        response = EmbeddingResponse(
            data=embedding_data,
            model=result.model,
            usage=result.usage_stats
        )

        # Log usage in background
        def log_usage():
            log.info("Embedding usage recorded",
                    user=current_user.username,
                    num_texts=len(request.input),
                    model=request.model,
                    total_chars=sum(len(text) for text in request.input))

        background_tasks.add_task(log_usage)

        return response

    except DomainException as e:
        if "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise HTTPException(status_code=400, detail=str(e))
        elif "limit" in str(e).lower():
            raise HTTPException(status_code=429, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/embeddings/models")
async def list_embedding_models(
    current_user=Depends(get_current_user)
):
    """List available embedding models"""

    try:
        container = get_container()
        embedding_service = container.get(EmbeddingService)

        models = await embedding_service.get_available_models()

        return {"object": "list", "data": models}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="First text for comparison")
    text2: str = Field(..., description="Second text for comparison")
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use")

@router.post("/embeddings/similarity")
async def compute_similarity(
    request: SimilarityRequest,
    current_user=Depends(get_current_user)
):
    """Compute semantic similarity between two texts"""

    try:
        container = get_container()
        embedding_service = container.get(EmbeddingService)

        similarity = await embedding_service.compute_similarity(
            text1=request.text1,
            text2=request.text2,
            model=request.model,
            user=current_user
        )

        return {
            "similarity": similarity,
            "text1": request.text1,
            "text2": request.text2,
            "model": request.model
        }

    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class BatchEmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., description="Texts to embed", max_items=1000)
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use")
    batch_size: int = Field(default=50, description="Batch size for processing", le=100)
    input_type: str = Field(default="query", description="Input type for embeddings")

@router.post("/embeddings/batch")
async def batch_embeddings(
    request: BatchEmbeddingRequest,
    current_user=Depends(get_current_user)
):
    """Process large batches of texts for embedding generation"""

    if len(request.texts) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 texts per batch request")

    # Process in batches
    all_embeddings = []
    total_processed = 0

    for i in range(0, len(request.texts), request.batch_size):
        batch = request.texts[i:i + request.batch_size]

        log.info("Processing batch",
                batch_num=i // request.batch_size + 1,
                batch_size=len(batch),
                total_texts=len(request.texts))

        batch_result = await embedding_service.generate_embeddings(
            texts=batch,
            model=request.model,
            input_type=request.input_type
        )

        # Adjust indices for global position
        for embedding_data in batch_result.data:
            embedding_data.index = total_processed
            all_embeddings.append(embedding_data)
            total_processed += 1

    # Calculate total usage
    total_tokens = sum(len(text.split()) for text in request.texts)

    return EmbeddingResponse(
        data=all_embeddings,
        model=request.model,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    )

@router.get("/embeddings/health")
async def embedding_service_health():
    """Check embedding service health"""

    try:
        # Test with a simple embedding
        test_result = await embedding_service.generate_embeddings(
            texts=["health check"],
            model="nvidia/embed-qa-4"
        )

        return {
            "status": "healthy",
            "service": "nvidia-embeddings",
            "model": "nvidia/embed-qa-4",
            "test_embedding_dimension": len(test_result.data[0].embedding),
            "timestamp": time.time()
        }

    except Exception as e:
        log.error("Embedding service health check failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service unhealthy: {str(e)}"
        )
