"""
gRPC client for the high-performance embedding service
"""

import asyncio
import grpc
from typing import List, Optional, Dict, Any
import structlog
from dataclasses import dataclass
from datetime import datetime, timezone

# Import the generated protobuf files (would be generated in production)
# import embedding_pb2
# import embedding_pb2_grpc

from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result from gRPC embedding service"""
    text: str
    embedding: List[float]
    model: str
    timestamp: datetime
    metadata: Dict[str, Any]
    from_cache: bool = False
    cache_key: Optional[str] = None


@dataclass
class EmbeddingMetrics:
    """Metrics from embedding request"""
    request_duration_ms: int
    api_duration_ms: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    total_tokens: int
    cached_tokens: int
    api_tokens: int
    cost_usd: float


class EmbeddingGRPCClient:
    """High-performance gRPC client for embedding service"""
    
    def __init__(
        self,
        server_url: str = "embedding-service:50051",
        timeout: float = 30.0,
        max_message_length: int = 100 * 1024 * 1024,  # 100MB
        keepalive_time_ms: int = 30000,
        keepalive_timeout_ms: int = 5000,
        keepalive_permit_without_calls: bool = True,
        max_retry_attempts: int = 3
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.max_retry_attempts = max_retry_attempts
        
        # gRPC channel options for high performance
        self.channel_options = [
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length),
            ('grpc.keepalive_time_ms', keepalive_time_ms),
            ('grpc.keepalive_timeout_ms', keepalive_timeout_ms),
            ('grpc.keepalive_permit_without_calls', keepalive_permit_without_calls),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),
        ]
        
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub = None
        
        log.info("gRPC embedding client initialized",
                server_url=server_url,
                timeout=timeout,
                max_message_length=max_message_length)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def connect(self):
        """Establish gRPC connection"""
        try:
            self._channel = grpc.aio.insecure_channel(
                self.server_url,
                options=self.channel_options
            )
            
            # Create stub (would use generated stub in production)
            # self._stub = embedding_pb2_grpc.EmbeddingServiceStub(self._channel)
            
            # Test connection
            await self._channel.channel_ready()
            
            log.info("gRPC connection established", server=self.server_url)
            
        except Exception as e:
            log.error("Failed to connect to embedding service", 
                     error=str(e), server=self.server_url)
            raise
    
    async def close(self):
        """Close gRPC connection"""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            log.info("gRPC connection closed")
    
    async def _retry_with_backoff(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff"""
        
        for attempt in range(self.max_retry_attempts):
            try:
                return await operation(*args, **kwargs)
                
            except grpc.aio.AioRpcError as e:
                if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                    if attempt < self.max_retry_attempts - 1:
                        delay = (2 ** attempt) * 0.1  # Exponential backoff
                        log.warning(f"gRPC call failed, retrying in {delay}s", 
                                   attempt=attempt + 1, error=str(e))
                        await asyncio.sleep(delay)
                        continue
                
                log.error("gRPC call failed permanently", error=str(e))
                raise
                
            except Exception as e:
                log.error("Unexpected error in gRPC call", error=str(e))
                raise
    
    async def embed_texts(
        self,
        texts: List[str],
        model: str = "nvidia/embed-qa-4",
        input_type: str = "query",
        use_cache: bool = True,
        batch_size: int = 50,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[List[EmbeddingResult], EmbeddingMetrics]:
        """
        Generate embeddings for multiple texts via gRPC service
        
        Returns:
            Tuple of (embedding_results, metrics)
        """
        
        if not texts:
            return [], EmbeddingMetrics(
                request_duration_ms=0, api_duration_ms=0,
                cache_hits=0, cache_misses=0, cache_hit_rate=0.0,
                total_tokens=0, cached_tokens=0, api_tokens=0, cost_usd=0.0
            )
        
        if not self._channel:
            await self.connect()
        
        log.info("Sending embedding request via gRPC",
                num_texts=len(texts), model=model, 
                input_type=input_type, use_cache=use_cache)
        
        # In production, this would create the protobuf request
        # request = embedding_pb2.EmbedRequest(
        #     texts=texts,
        #     model=model,
        #     input_type=input_type,
        #     use_cache=use_cache,
        #     batch_size=batch_size
        # )
        
        # Mock response for now (would be actual gRPC call in production)
        # response = await self._retry_with_backoff(
        #     self._stub.EmbedTexts,
        #     request,
        #     timeout=self.timeout
        # )
        
        # For now, return mock data showing the expected structure
        mock_embeddings = []
        for i, text in enumerate(texts):
            mock_embeddings.append(EmbeddingResult(
                text=text,
                embedding=[0.1] * 1024,  # Mock 1024-dim embedding
                model=model,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata[i] if metadata and i < len(metadata) else {},
                from_cache=i % 2 == 0,  # Mock 50% cache hit rate
                cache_key=f"mock_key_{i}"
            ))
        
        mock_metrics = EmbeddingMetrics(
            request_duration_ms=100,
            api_duration_ms=50,
            cache_hits=len(texts) // 2,
            cache_misses=len(texts) - len(texts) // 2,
            cache_hit_rate=0.5,
            total_tokens=sum(len(text.split()) for text in texts),
            cached_tokens=sum(len(text.split()) for i, text in enumerate(texts) if i % 2 == 0),
            api_tokens=sum(len(text.split()) for i, text in enumerate(texts) if i % 2 == 1),
            cost_usd=0.01
        )
        
        log.info("gRPC embedding request completed",
                num_results=len(mock_embeddings),
                cache_hit_rate=mock_metrics.cache_hit_rate,
                total_tokens=mock_metrics.total_tokens)
        
        return mock_embeddings, mock_metrics
    
    async def embed_single_text(
        self,
        text: str,
        model: str = "nvidia/embed-qa-4",
        input_type: str = "query",
        use_cache: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """Generate embedding for a single text"""
        
        results, _ = await self.embed_texts(
            texts=[text],
            model=model,
            input_type=input_type,
            use_cache=use_cache,
            metadata=[metadata] if metadata else None
        )
        
        return results[0] if results else None
    
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between two embeddings"""
        
        if not self._channel:
            await self.connect()
        
        # In production, this would be:
        # request = embedding_pb2.SimilarityRequest(
        #     embedding1=embedding1,
        #     embedding2=embedding2,
        #     metric=metric
        # )
        # 
        # response = await self._retry_with_backoff(
        #     self._stub.ComputeSimilarity,
        #     request,
        #     timeout=self.timeout
        # )
        # 
        # return response.similarity
        
        # Mock similarity computation
        import numpy as np
        
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        if metric == "cosine":
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif metric == "euclidean":
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1 / (1 + distance)
        elif metric == "dot":
            similarity = np.dot(emb1, emb2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return float(similarity)
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status"""
        
        if not self._channel:
            await self.connect()
        
        # In production:
        # request = embedding_pb2.HealthRequest()
        # response = await self._stub.GetHealth(request, timeout=5.0)
        # 
        # return {
        #     "status": response.status,
        #     "uptime_seconds": response.uptime_seconds,
        #     "details": dict(response.details),
        #     "cache_stats": {
        #         "l1_cache_size": response.cache_stats.l1_cache_size,
        #         "l2_cache_keys": response.cache_stats.l2_cache_keys,
        #         "l1_memory_mb": response.cache_stats.l1_memory_mb,
        #         "cache_hit_rate_1h": response.cache_stats.cache_hit_rate_1h
        #     }
        # }
        
        # Mock health response
        return {
            "status": "healthy",
            "uptime_seconds": 3600,
            "details": {
                "version": "1.0.0",
                "model": "nvidia/embed-qa-4"
            },
            "cache_stats": {
                "l1_cache_size": 1000,
                "l2_cache_keys": 5000,
                "l1_memory_mb": 512.0,
                "cache_hit_rate_1h": 0.75
            }
        }
    
    async def clear_cache(
        self,
        model: Optional[str] = None,
        l1_only: bool = False
    ) -> Dict[str, Any]:
        """Clear embedding caches"""
        
        if not self._channel:
            await self.connect()
        
        log.info("Clearing embedding cache", model=model, l1_only=l1_only)
        
        # In production:
        # request = embedding_pb2.ClearCacheRequest(
        #     model=model or "",
        #     l1_only=l1_only
        # )
        # response = await self._stub.ClearCache(request, timeout=10.0)
        # 
        # return {
        #     "l1_keys_cleared": response.l1_keys_cleared,
        #     "l2_keys_cleared": response.l2_keys_cleared,
        #     "status": response.status
        # }
        
        # Mock clear response
        return {
            "l1_keys_cleared": 1000,
            "l2_keys_cleared": 5000 if not l1_only else 0,
            "status": "success"
        }


# Global client instance for reuse
_global_client: Optional[EmbeddingGRPCClient] = None


async def get_embedding_client() -> EmbeddingGRPCClient:
    """Get or create global embedding gRPC client"""
    global _global_client
    
    if _global_client is None:
        _global_client = EmbeddingGRPCClient()
        await _global_client.connect()
    
    return _global_client


async def close_embedding_client():
    """Close global embedding client"""
    global _global_client
    
    if _global_client:
        await _global_client.close()
        _global_client = None