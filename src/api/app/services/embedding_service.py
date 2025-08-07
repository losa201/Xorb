"""
Embedding service implementation
"""

import time
from typing import List, Dict, Any, Optional
from uuid import UUID

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from ..domain.entities import User, Organization, EmbeddingRequest, EmbeddingResult
from ..domain.exceptions import (
    EmbeddingGenerationFailed, ValidationError, ServiceUnavailable,
    ResourceLimitExceeded
)
from ..domain.repositories import EmbeddingRepository
from ..domain.value_objects import UsageStats
from .interfaces import EmbeddingService


class EmbeddingServiceImpl(EmbeddingService):
    """Implementation of embedding service"""
    
    def __init__(
        self,
        embedding_repository: EmbeddingRepository,
        nvidia_api_key: str,
        nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    ):
        self.embedding_repository = embedding_repository
        self.nvidia_api_key = nvidia_api_key
        self.nvidia_base_url = nvidia_base_url
        
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=nvidia_api_key,
                base_url=nvidia_base_url
            )
        else:
            self.client = None
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str,
        input_type: str,
        user: User,
        org: Organization
    ) -> EmbeddingResult:
        """Generate embeddings for texts"""
        
        # Create and validate request
        embedding_request = EmbeddingRequest.create(
            texts=texts,
            model=model,
            input_type=input_type,
            user_id=user.id,
            org_id=org.id
        )
        
        try:
            embedding_request.validate()
        except ValueError as e:
            raise ValidationError(str(e))
        
        # Save request
        await self.embedding_repository.save_request(embedding_request)
        
        start_time = time.time()
        
        try:
            if not OPENAI_AVAILABLE or not self.client:
                # Mock embedding generation for testing
                embeddings = [[0.1, 0.2, 0.3] * 100 for _ in texts]  # Mock 300-dim embeddings
                processing_time = int((time.time() - start_time) * 1000)
                total_tokens = sum(len(text.split()) for text in texts)
                
                usage_stats = UsageStats(
                    prompt_tokens=total_tokens,
                    total_tokens=total_tokens,
                    processing_time_ms=processing_time
                )
                
                result = EmbeddingResult.create(
                    request_id=embedding_request.id,
                    embeddings=embeddings,
                    model=model,
                    usage_stats=usage_stats.__dict__
                )
                
                await self.embedding_repository.save_result(result)
                embedding_request.mark_completed()
                await self.embedding_repository.save_request(embedding_request)
                
                return result
            
            # Call NVIDIA API
            response = self.client.embeddings.create(
                input=texts,
                model=model,
                encoding_format="float",
                extra_body={
                    "input_type": input_type,
                    "truncate": "NONE"
                }
            )
            
            # Extract embeddings
            embeddings = [embedding.embedding for embedding in response.data]
            
            # Calculate usage stats
            processing_time = int((time.time() - start_time) * 1000)
            total_tokens = sum(len(text.split()) for text in texts)
            
            usage_stats = UsageStats(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
                processing_time_ms=processing_time
            )
            
            # Create and save result
            result = EmbeddingResult.create(
                request_id=embedding_request.id,
                embeddings=embeddings,
                model=model,
                usage_stats=usage_stats.__dict__
            )
            
            await self.embedding_repository.save_result(result)
            
            # Mark request as completed
            embedding_request.mark_completed()
            await self.embedding_repository.save_request(embedding_request)
            
            return result
            
        except Exception as e:
            # Mark request as failed
            embedding_request.mark_failed()
            await self.embedding_repository.save_request(embedding_request)
            
            raise EmbeddingGenerationFailed(
                f"Failed to generate embeddings: {str(e)}",
                model=model
            )
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        model: str,
        user: User
    ) -> float:
        """Compute similarity between two texts"""
        
        if not text1.strip() or not text2.strip():
            raise ValidationError("Both texts must be non-empty")
        
        try:
            if not OPENAI_AVAILABLE or not self.client or not NUMPY_AVAILABLE:
                # Mock similarity computation for testing
                return 0.85  # Mock similarity score
            
            # Generate embeddings for both texts
            response = self.client.embeddings.create(
                input=[text1, text2],
                model=model,
                encoding_format="float",
                extra_body={
                    "input_type": "query",
                    "truncate": "NONE"
                }
            )
            
            # Extract embeddings
            emb1 = np.array(response.data[0].embedding)
            emb2 = np.array(response.data[1].embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            return float(similarity)
            
        except Exception as e:
            raise EmbeddingGenerationFailed(
                f"Failed to compute similarity: {str(e)}",
                model=model
            )
    
    async def batch_embeddings(
        self,
        texts: List[str],
        model: str,
        batch_size: int,
        input_type: str,
        user: User,
        org: Organization
    ) -> EmbeddingResult:
        """Process large batches of texts"""
        
        if len(texts) > 1000:
            raise ResourceLimitExceeded(
                "Maximum 1000 texts per batch request",
                resource_type="batch_texts",
                limit=1000
            )
        
        if batch_size > 100:
            raise ValidationError("Maximum batch size is 100")
        
        # Create overall request
        embedding_request = EmbeddingRequest.create(
            texts=texts,
            model=model,
            input_type=input_type,
            user_id=user.id,
            org_id=org.id
        )
        
        await self.embedding_repository.save_request(embedding_request)
        
        start_time = time.time()
        all_embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=model,
                    encoding_format="float",
                    extra_body={
                        "input_type": input_type,
                        "truncate": "NONE"
                    }
                )
                
                batch_embeddings = [emb.embedding for emb in response.data]
                all_embeddings.extend(batch_embeddings)
            
            # Calculate usage stats
            processing_time = int((time.time() - start_time) * 1000)
            total_tokens = sum(len(text.split()) for text in texts)
            
            usage_stats = UsageStats(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
                processing_time_ms=processing_time
            )
            
            # Create and save result
            result = EmbeddingResult.create(
                request_id=embedding_request.id,
                embeddings=all_embeddings,
                model=model,
                usage_stats=usage_stats.__dict__
            )
            
            await self.embedding_repository.save_result(result)
            
            # Mark request as completed
            embedding_request.mark_completed()
            await self.embedding_repository.save_request(embedding_request)
            
            return result
            
        except Exception as e:
            # Mark request as failed
            embedding_request.mark_failed()
            await self.embedding_repository.save_request(embedding_request)
            
            raise EmbeddingGenerationFailed(
                f"Failed to process batch embeddings: {str(e)}",
                model=model
            )
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available embedding models"""
        
        return [
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
    
    async def health_check(self) -> Dict[str, Any]:
        """Check embedding service health"""
        
        try:
            # Test with a simple embedding
            response = self.client.embeddings.create(
                input=["health check"],
                model="nvidia/embed-qa-4",
                encoding_format="float"
            )
            
            return {
                "status": "healthy",
                "service": "nvidia-embeddings",
                "model": "nvidia/embed-qa-4",
                "test_embedding_dimension": len(response.data[0].embedding),
                "timestamp": time.time()
            }
            
        except Exception as e:
            raise ServiceUnavailable(
                f"Embedding service unhealthy: {str(e)}",
                service="nvidia-embeddings"
            )