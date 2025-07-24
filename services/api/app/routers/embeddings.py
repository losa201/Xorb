"""
NVIDIA Embeddings API Router
Provides embedding generation using NVIDIA's embed-qa-4 model
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
import time
from prometheus_client import Counter, Histogram
import structlog

from ..deps import has_role
from xorb_core.logging import get_logger

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

router = APIRouter()

# Pydantic models
class EmbeddingRequest(BaseModel):
    input: List[str] = Field(..., description="List of texts to embed", max_items=100)
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use")
    input_type: Optional[str] = Field(default="query", description="Type of input text")
    truncate: Optional[str] = Field(default="NONE", description="Truncation strategy")
    encoding_format: str = Field(default="float", description="Encoding format for embeddings")

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]

class EmbeddingService:
    """Service for generating embeddings using NVIDIA API"""
    
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY", "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")
        self.base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        log.info("NVIDIA Embedding Service initialized", 
                base_url=self.base_url, 
                api_key_present=bool(self.api_key))
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        model: str = "nvidia/embed-qa-4",
        input_type: str = "query",
        truncate: str = "NONE",
        encoding_format: str = "float"
    ) -> EmbeddingResponse:
        """Generate embeddings for input texts"""
        
        start_time = time.time()
        
        try:
            log.info("Generating embeddings", 
                    num_texts=len(texts), 
                    model=model, 
                    input_type=input_type)
            
            # Call NVIDIA API
            response = self.client.embeddings.create(
                input=texts,
                model=model,
                encoding_format=encoding_format,
                extra_body={
                    "input_type": input_type,
                    "truncate": truncate
                }
            )
            
            # Convert OpenAI response to our format
            embedding_data = []
            for i, embedding in enumerate(response.data):
                embedding_data.append(EmbeddingData(
                    embedding=embedding.embedding,
                    index=i
                ))
            
            # Calculate usage stats
            total_tokens = sum(len(text.split()) for text in texts)
            
            result = EmbeddingResponse(
                data=embedding_data,
                model=model,
                usage={
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            )
            
            # Record metrics
            duration = time.time() - start_time
            embedding_requests_total.labels(model=model, status="success").inc()
            embedding_duration_seconds.labels(model=model).observe(duration)
            
            log.info("Embeddings generated successfully", 
                    duration=duration,
                    num_embeddings=len(embedding_data),
                    model=model)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            embedding_requests_total.labels(model=model, status="error").inc()
            
            log.error("Failed to generate embeddings", 
                     error=str(e), 
                     duration=duration,
                     model=model)
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {str(e)}"
            )

# Initialize service
embedding_service = EmbeddingService()

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(has_role("user"))
):
    """
    Generate embeddings for input texts using NVIDIA's embed-qa-4 model
    
    - **input**: List of texts to embed (max 100 items)
    - **model**: Embedding model to use (default: nvidia/embed-qa-4)
    - **input_type**: Type of input text (query, passage, classification, etc.)
    - **truncate**: Truncation strategy (NONE, START, END)
    - **encoding_format**: Format for embeddings (float, base64)
    """
    
    # Validate input
    if not request.input:
        raise HTTPException(status_code=400, detail="Input texts cannot be empty")
    
    if len(request.input) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per request")
    
    # Check for empty strings
    if any(not text.strip() for text in request.input):
        raise HTTPException(status_code=400, detail="Input texts cannot be empty strings")
    
    # Generate embeddings
    result = await embedding_service.generate_embeddings(
        texts=request.input,
        model=request.model,
        input_type=request.input_type or "query",
        truncate=request.truncate or "NONE",
        encoding_format=request.encoding_format
    )
    
    # Log usage for analytics
    def log_usage():
        log.info("Embedding usage recorded", 
                user=current_user.get("username"),
                num_texts=len(request.input),
                model=request.model,
                total_chars=sum(len(text) for text in request.input))
    
    background_tasks.add_task(log_usage)
    
    return result

@router.get("/embeddings/models")
async def list_embedding_models(
    current_user: dict = Depends(has_role("user"))
):
    """List available embedding models"""
    
    models = [
        {
            "id": "nvidia/embed-qa-4",
            "object": "model",
            "created": 1699000000,
            "owned_by": "nvidia",
            "description": "High-quality embedding model optimized for Q&A tasks",
            "max_input_tokens": 32768,
            "embedding_dimension": 1024
        }
    ]
    
    return {"object": "list", "data": models}

@router.post("/embeddings/similarity")
async def compute_similarity(
    text1: str = Field(..., description="First text for comparison"),
    text2: str = Field(..., description="Second text for comparison"),
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use"),
    current_user: dict = Depends(has_role("user"))
):
    """Compute semantic similarity between two texts"""
    
    import numpy as np
    
    # Generate embeddings for both texts
    result = await embedding_service.generate_embeddings(
        texts=[text1, text2],
        model=model,
        input_type="query"
    )
    
    # Calculate cosine similarity
    emb1 = np.array(result.data[0].embedding)
    emb2 = np.array(result.data[1].embedding)
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    return {
        "similarity": float(similarity),
        "text1": text1,
        "text2": text2,
        "model": model,
        "embedding_dimensions": len(emb1)
    }

@router.post("/embeddings/batch")
async def batch_embeddings(
    texts: List[str] = Field(..., description="Texts to embed", max_items=1000),
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use"),
    batch_size: int = Field(default=50, description="Batch size for processing", le=100),
    input_type: str = Field(default="query", description="Input type for embeddings"),
    current_user: dict = Depends(has_role("user"))
):
    """Process large batches of texts for embedding generation"""
    
    if len(texts) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 texts per batch request")
    
    # Process in batches
    all_embeddings = []
    total_processed = 0
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        log.info("Processing batch", 
                batch_num=i // batch_size + 1,
                batch_size=len(batch),
                total_texts=len(texts))
        
        batch_result = await embedding_service.generate_embeddings(
            texts=batch,
            model=model,
            input_type=input_type
        )
        
        # Adjust indices for global position
        for embedding_data in batch_result.data:
            embedding_data.index = total_processed
            all_embeddings.append(embedding_data)
            total_processed += 1
    
    # Calculate total usage
    total_tokens = sum(len(text.split()) for text in texts)
    
    return EmbeddingResponse(
        data=all_embeddings,
        model=model,
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