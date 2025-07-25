"""
gRPC Service Implementations

High-performance RPC services for internal communication and microservices.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import grpc

from ...application import (
    GenerateEmbeddingCommand,
    GenerateEmbeddingUseCase,
    KnowledgeApplicationService
)
from ..dependencies import get_knowledge_service

__all__ = [
    "EmbeddingGrpcService",
    "CampaignGrpcService"
]

logger = logging.getLogger(__name__)


class EmbeddingGrpcService:
    """gRPC service for embedding generation"""
    
    def __init__(self, knowledge_service: KnowledgeApplicationService) -> None:
        self._knowledge_service = knowledge_service
    
    async def GenerateEmbedding(self, request, context) -> any:
        """Generate embedding for text input"""
        
        try:
            # Extract request data (would use protobuf message in real implementation)
            text = getattr(request, 'text', '')
            model = getattr(request, 'model', 'nvidia/embed-qa-4')
            input_type = getattr(request, 'input_type', 'query')
            
            if not text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Text field is required')
                return None
            
            # Generate embedding using application service
            embedding = await self._knowledge_service._generate_embedding_use_case.execute(
                GenerateEmbeddingCommand(
                    text=text,
                    model=model,
                    input_type=input_type
                )
            )
            
            # Create response (would use protobuf message in real implementation)
            class EmbeddingResponse:
                def __init__(self, vector: List[float], dimension: int, model: str):
                    self.vector = vector
                    self.dimension = dimension
                    self.model = model
            
            return EmbeddingResponse(
                vector=list(embedding.vector),
                dimension=embedding.dimension,
                model=embedding.model
            )
            
        except Exception as e:
            logger.error("gRPC embedding generation failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Embedding generation failed: {str(e)}')
            return None
    
    async def GenerateBatchEmbeddings(self, request, context) -> any:
        """Generate embeddings for multiple texts"""
        
        try:
            # Extract batch request data
            texts = getattr(request, 'texts', [])
            model = getattr(request, 'model', 'nvidia/embed-qa-4')
            input_type = getattr(request, 'input_type', 'query')
            
            if not texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Texts field is required')
                return None
            
            # Generate embeddings for each text
            embeddings = []
            for text in texts:
                embedding = await self._knowledge_service._generate_embedding_use_case.execute(
                    GenerateEmbeddingCommand(
                        text=text,
                        model=model,
                        input_type=input_type
                    )
                )
                embeddings.append(embedding)
            
            # Create batch response
            class BatchEmbeddingResponse:
                def __init__(self, embeddings: List[any]):
                    self.embeddings = embeddings
                    self.count = len(embeddings)
            
            response_embeddings = []
            for emb in embeddings:
                class EmbeddingItem:
                    def __init__(self, vector: List[float], dimension: int, model: str):
                        self.vector = vector
                        self.dimension = dimension
                        self.model = model
                
                response_embeddings.append(EmbeddingItem(
                    vector=list(emb.vector),
                    dimension=emb.dimension,
                    model=emb.model
                ))
            
            return BatchEmbeddingResponse(embeddings=response_embeddings)
            
        except Exception as e:
            logger.error("gRPC batch embedding generation failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Batch embedding generation failed: {str(e)}')
            return None
    
    async def ComputeSimilarity(self, request, context) -> any:
        """Compute similarity between two embeddings"""
        
        try:
            # Extract embeddings from request
            embedding1_vector = getattr(request, 'embedding1', [])
            embedding2_vector = getattr(request, 'embedding2', [])
            metric = getattr(request, 'metric', 'cosine')
            
            if not embedding1_vector or not embedding2_vector:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Both embeddings are required')
                return None
            
            # Create embedding objects
            from ...domain import Embedding
            
            embedding1 = Embedding.from_list(vector=embedding1_vector)
            embedding2 = Embedding.from_list(vector=embedding2_vector)
            
            # Compute similarity
            if metric == 'cosine':
                similarity = embedding1.similarity_cosine(embedding2)
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f'Unsupported similarity metric: {metric}')
                return None
            
            # Create response
            class SimilarityResponse:
                def __init__(self, similarity: float, metric: str):
                    self.similarity = similarity
                    self.metric = metric
            
            return SimilarityResponse(similarity=similarity, metric=metric)
            
        except Exception as e:
            logger.error("gRPC similarity computation failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Similarity computation failed: {str(e)}')
            return None


class CampaignGrpcService:
    """gRPC service for campaign management (internal communication)"""
    
    def __init__(self) -> None:
        pass
    
    async def GetCampaignStatus(self, request, context) -> any:
        """Get campaign status for internal monitoring"""
        
        try:
            campaign_id = getattr(request, 'campaign_id', '')
            
            if not campaign_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Campaign ID is required')
                return None
            
            # This would fetch actual campaign status
            class CampaignStatusResponse:
                def __init__(self, campaign_id: str, status: str, progress: float):
                    self.campaign_id = campaign_id
                    self.status = status
                    self.progress = progress
                    self.agents_running = 0
                    self.findings_count = 0
            
            return CampaignStatusResponse(
                campaign_id=campaign_id,
                status="running",
                progress=0.75
            )
            
        except Exception as e:
            logger.error("gRPC campaign status failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Campaign status retrieval failed: {str(e)}')
            return None
    
    async def StreamCampaignEvents(self, request, context) -> any:
        """Stream campaign events for real-time monitoring"""
        
        try:
            campaign_id = getattr(request, 'campaign_id', '')
            
            if not campaign_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Campaign ID is required')
                return
            
            # This would stream real campaign events
            # For now, yield a sample event
            class CampaignEvent:
                def __init__(self, event_type: str, timestamp: str, data: dict):
                    self.event_type = event_type
                    self.timestamp = timestamp
                    self.data = data
            
            import json
            from datetime import datetime, timezone
            
            yield CampaignEvent(
                event_type="agent_started",
                timestamp=datetime.now(timezone.utc).isoformat(),
                data=json.dumps({"agent_id": "sample-agent", "campaign_id": campaign_id})
            )
            
        except Exception as e:
            logger.error("gRPC campaign event streaming failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Campaign event streaming failed: {str(e)}')
            return


# gRPC service factory functions
async def create_embedding_service() -> EmbeddingGrpcService:
    """Create embedding gRPC service with dependencies"""
    
    knowledge_service = await get_knowledge_service()
    return EmbeddingGrpcService(knowledge_service)


async def create_campaign_service() -> CampaignGrpcService:
    """Create campaign gRPC service with dependencies"""
    
    return CampaignGrpcService()