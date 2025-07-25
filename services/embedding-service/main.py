"""
High-performance gRPC Embedding Service
Optimized for EPYC architecture with 2-tier caching
"""

import asyncio
import time
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

import grpc
from grpc import aio
import redis.asyncio as redis
from cachetools import TTLCache
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Generated protobuf imports (would be generated from proto file)
import embedding_pb2
import embedding_pb2_grpc

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

log = structlog.get_logger(__name__)

# Prometheus metrics
embedding_requests_total = Counter(
    'embedding_service_requests_total',
    'Total embedding requests',
    ['model', 'status', 'cache_type']
)

embedding_duration_seconds = Histogram(
    'embedding_service_duration_seconds',
    'Time spent generating embeddings',
    ['model', 'operation']
)

embedding_tokens_total = Counter(
    'embedding_service_tokens_total',
    'Total tokens processed',
    ['model', 'input_type', 'source']
)

cache_operations_total = Counter(
    'embedding_service_cache_operations_total',
    'Cache operations',
    ['cache_type', 'operation', 'result']
)

active_connections = Gauge(
    'embedding_service_active_connections',
    'Number of active gRPC connections'
)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service"""
    nvidia_api_key: str = "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"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    redis_url: str = "redis://redis:6379/1"  # Use DB 1 for embedding service
    default_model: str = "nvidia/embed-qa-4"
    l1_cache_size: int = 10000
    l1_cache_ttl: int = 3600  # 1 hour
    l2_cache_ttl: int = 86400  # 24 hours
    batch_size: int = 50
    max_concurrent_requests: int = 32  # EPYC optimized
    grpc_port: int = 50051
    metrics_port: int = 9090


class EmbeddingServiceImpl(embedding_pb2_grpc.EmbeddingServiceServicer):
    """High-performance gRPC embedding service implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.start_time = time.time()
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=config.nvidia_api_key,
            base_url=config.nvidia_base_url
        )
        
        # Initialize caches
        self.l1_cache = TTLCache(
            maxsize=config.l1_cache_size,
            ttl=config.l1_cache_ttl
        )
        
        # Redis connection pool
        self.redis_pool = redis.ConnectionPool.from_url(
            config.redis_url,
            encoding="utf-8",
            decode_responses=False,
            max_connections=20
        )
        
        log.info("Embedding service initialized",
                model=config.default_model,
                l1_cache_size=config.l1_cache_size,
                redis_url=config.redis_url)
    
    def _generate_cache_key(self, text: str, model: str, input_type: str) -> str:
        """Generate SHA-1 cache key for deduplication"""
        content = f"{model}:{input_type}:{text}"
        return hashlib.sha1(content.encode('utf-8')).hexdigest()
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client from pool"""
        return redis.Redis(connection_pool=self.redis_pool)
    
    async def _embed_single_batch(
        self,
        texts: List[str],
        model: str,
        input_type: str
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for a single batch via API"""
        
        start_time = time.time()
        
        try:
            response = await self.openai_client.embeddings.create(
                input=texts,
                model=model,
                encoding_format="float",
                extra_body={
                    "input_type": input_type,
                    "truncate": "NONE"
                }
            )
            
            api_duration = time.time() - start_time
            embedding_duration_seconds.labels(
                model=model, operation="api_call"
            ).observe(api_duration)
            
            # Process results
            results = []
            for i, embedding_data in enumerate(response.data):
                results.append({
                    "text": texts[i],
                    "embedding": embedding_data.embedding,
                    "created_unix_ms": int(time.time() * 1000),
                    "from_cache": False,
                    "cache_key": self._generate_cache_key(texts[i], model, input_type)
                })
            
            # Track token usage
            total_tokens = sum(len(text.split()) for text in texts)
            embedding_tokens_total.labels(
                model=model, input_type=input_type, source="api"
            ).inc(total_tokens)
            
            log.info("API batch completed",
                    batch_size=len(texts),
                    api_duration=api_duration,
                    total_tokens=total_tokens)
            
            return results
            
        except Exception as e:
            log.error("API batch failed", error=str(e), batch_size=len(texts))
            raise
    
    async def _cache_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        model: str,
        input_type: str
    ):
        """Store embeddings in both L1 and L2 caches"""
        
        redis_client = await self._get_redis_client()
        
        for emb in embeddings:
            cache_key = emb["cache_key"]
            
            # Store in L1 cache
            self.l1_cache[cache_key] = emb
            cache_operations_total.labels(
                cache_type="l1", operation="set", result="success"
            ).inc()
            
            # Store in L2 cache (Redis)
            try:
                cache_data = {
                    "embedding": emb["embedding"],
                    "created_unix_ms": emb["created_unix_ms"],
                    "text": emb["text"]
                }
                
                await redis_client.setex(
                    cache_key,
                    self.config.l2_cache_ttl,
                    json.dumps(cache_data)
                )
                
                cache_operations_total.labels(
                    cache_type="l2", operation="set", result="success"
                ).inc()
                
            except Exception as e:
                log.warning("L2 cache store failed", error=str(e), cache_key=cache_key)
                cache_operations_total.labels(
                    cache_type="l2", operation="set", result="error"
                ).inc()
        
        await redis_client.aclose()
    
    async def _get_cached_embeddings(
        self,
        texts: List[str],
        model: str,
        input_type: str
    ) -> tuple[List[Optional[Dict[str, Any]]], List[str], List[int]]:
        """Get cached embeddings, return (results, uncached_texts, uncached_indices)"""
        
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []
        
        redis_client = await self._get_redis_client()
        
        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text, model, input_type)
            cached_result = None
            
            # Check L1 cache first
            if cache_key in self.l1_cache:
                cached_result = self.l1_cache[cache_key].copy()
                cached_result["from_cache"] = True
                cache_operations_total.labels(
                    cache_type="l1", operation="get", result="hit"
                ).inc()
                
            # Check L2 cache if not in L1
            elif redis_client:
                try:
                    cached_data = await redis_client.get(cache_key)
                    if cached_data:
                        cache_info = json.loads(cached_data)
                        cached_result = {
                            "text": text,
                            "embedding": cache_info["embedding"],
                            "created_unix_ms": cache_info["created_unix_ms"],
                            "from_cache": True,
                            "cache_key": cache_key
                        }
                        
                        # Store in L1 for next access
                        self.l1_cache[cache_key] = cached_result
                        
                        cache_operations_total.labels(
                            cache_type="l2", operation="get", result="hit"
                        ).inc()
                        
                except Exception as e:
                    log.warning("L2 cache read failed", error=str(e))
                    cache_operations_total.labels(
                        cache_type="l2", operation="get", result="error"
                    ).inc()
            
            if cached_result:
                results[i] = cached_result
                embedding_tokens_total.labels(
                    model=model, input_type=input_type, source="cache"
                ).inc(len(text.split()))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                cache_operations_total.labels(
                    cache_type="l1", operation="get", result="miss"
                ).inc()
        
        await redis_client.aclose()
        return results, uncached_texts, uncached_indices
    
    async def EmbedTexts(
        self,
        request: embedding_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_pb2.EmbedResponse:
        """Generate embeddings for multiple texts with caching"""
        
        start_time = time.time()
        active_connections.inc()
        
        try:
            # Validate request
            if not request.texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No texts provided")
                return embedding_pb2.EmbedResponse()
            
            model = request.model or self.config.default_model
            input_type = request.input_type or "query"
            use_cache = request.use_cache if hasattr(request, 'use_cache') else True
            batch_size = request.batch_size or self.config.batch_size
            
            log.info("Embedding request received",
                    num_texts=len(request.texts),
                    model=model,
                    input_type=input_type,
                    use_cache=use_cache)
            
            all_embeddings = []
            total_tokens = 0
            cache_hits = 0
            cache_misses = 0
            api_duration = 0
            
            if use_cache:
                # Get cached embeddings
                cached_results, uncached_texts, uncached_indices = await self._get_cached_embeddings(
                    list(request.texts), model, input_type
                )
                
                cache_hits = sum(1 for r in cached_results if r is not None)
                cache_misses = len(uncached_texts)
                
                # Add cached results
                for result in cached_results:
                    if result:
                        all_embeddings.append(result)
                
                # Process uncached texts in batches
                if uncached_texts:
                    api_start = time.time()
                    
                    for i in range(0, len(uncached_texts), batch_size):
                        batch = uncached_texts[i:i + batch_size]
                        batch_results = await self._embed_single_batch(
                            batch, model, input_type
                        )
                        
                        # Cache the new embeddings
                        await self._cache_embeddings(batch_results, model, input_type)
                        
                        # Add to results at correct positions
                        for j, result in enumerate(batch_results):
                            original_index = uncached_indices[i + j]
                            cached_results[original_index] = result
                    
                    api_duration = time.time() - api_start
                
                # Ensure results are in correct order
                all_embeddings = [r for r in cached_results if r is not None]
                
            else:
                # No caching - direct API calls
                api_start = time.time()
                texts_list = list(request.texts)
                
                for i in range(0, len(texts_list), batch_size):
                    batch = texts_list[i:i + batch_size]
                    batch_results = await self._embed_single_batch(
                        batch, model, input_type
                    )
                    all_embeddings.extend(batch_results)
                
                api_duration = time.time() - api_start
                cache_misses = len(texts_list)
            
            # Calculate metrics
            total_duration = time.time() - start_time
            total_tokens = sum(len(emb["text"].split()) for emb in all_embeddings)
            cached_tokens = sum(
                len(emb["text"].split()) for emb in all_embeddings 
                if emb.get("from_cache", False)
            )
            api_tokens = total_tokens - cached_tokens
            cache_hit_rate = cache_hits / len(request.texts) if request.texts else 0
            
            # Record metrics
            embedding_requests_total.labels(
                model=model, status="success", cache_type="mixed"
            ).inc()
            
            embedding_duration_seconds.labels(
                model=model, operation="total"
            ).observe(total_duration)
            
            # Build response
            response_embeddings = []
            for emb in all_embeddings:
                pb_embedding = embedding_pb2.Embedding(
                    text=emb["text"],
                    vector=emb["embedding"],
                    created_unix_ms=emb["created_unix_ms"],
                    cache_key=emb["cache_key"],
                    from_cache=emb.get("from_cache", False)
                )
                response_embeddings.append(pb_embedding)
            
            usage = embedding_pb2.EmbedUsage(
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                api_tokens=api_tokens,
                cost_usd=api_tokens * 0.0001  # Rough estimate
            )
            
            metrics = embedding_pb2.EmbedMetrics(
                request_duration_ms=int(total_duration * 1000),
                api_duration_ms=int(api_duration * 1000),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_hit_rate=cache_hit_rate
            )
            
            response = embedding_pb2.EmbedResponse(
                embeddings=response_embeddings,
                model=model,
                usage=usage,
                metrics=metrics
            )
            
            log.info("Embedding request completed",
                    total_duration=total_duration,
                    api_duration=api_duration,
                    cache_hit_rate=cache_hit_rate,
                    total_tokens=total_tokens)
            
            return response
            
        except Exception as e:
            log.error("Embedding request failed", error=str(e))
            embedding_requests_total.labels(
                model=model, status="error", cache_type="mixed"
            ).inc()
            
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Embedding generation failed: {str(e)}")
            return embedding_pb2.EmbedResponse()
            
        finally:
            active_connections.dec()
    
    async def ComputeSimilarity(
        self,
        request: embedding_pb2.SimilarityRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_pb2.SimilarityResponse:
        """Compute similarity between two embeddings"""
        
        import numpy as np
        
        try:
            if not request.embedding1 or not request.embedding2:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Both embeddings required")
                return embedding_pb2.SimilarityResponse()
            
            emb1 = np.array(request.embedding1)
            emb2 = np.array(request.embedding2)
            metric = request.metric or "cosine"
            
            if metric == "cosine":
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            elif metric == "euclidean":
                distance = np.linalg.norm(emb1 - emb2)
                similarity = 1 / (1 + distance)
            elif metric == "dot":
                similarity = np.dot(emb1, emb2)
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Unknown metric: {metric}")
                return embedding_pb2.SimilarityResponse()
            
            return embedding_pb2.SimilarityResponse(
                similarity=float(similarity),
                metric=metric
            )
            
        except Exception as e:
            log.error("Similarity computation failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return embedding_pb2.SimilarityResponse()
    
    async def GetHealth(
        self,
        request: embedding_pb2.HealthRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_pb2.HealthResponse:
        """Get service health and statistics"""
        
        try:
            uptime = int(time.time() - self.start_time)
            
            # Get cache stats
            redis_client = await self._get_redis_client()
            redis_keys = 0
            
            try:
                pattern = f"{self.config.default_model}:*"
                keys = await redis_client.keys(pattern)
                redis_keys = len(keys)
            except Exception as e:
                log.warning("Failed to get Redis stats", error=str(e))
            finally:
                await redis_client.aclose()
            
            l1_memory_mb = sum(
                len(str(v)) for v in self.l1_cache.values()
            ) / (1024 * 1024)
            
            cache_stats = embedding_pb2.CacheStats(
                l1_cache_size=len(self.l1_cache),
                l2_cache_keys=redis_keys,
                l1_memory_mb=l1_memory_mb,
                cache_hit_rate_1h=0.0  # Would need historical data
            )
            
            status = "healthy"
            details = {
                "version": "1.0.0",
                "model": self.config.default_model,
                "l1_cache_utilization": f"{len(self.l1_cache)}/{self.config.l1_cache_size}",
                "redis_connected": "true" if redis_keys >= 0 else "false"
            }
            
            return embedding_pb2.HealthResponse(
                status=status,
                details=details,
                uptime_seconds=uptime,
                cache_stats=cache_stats
            )
            
        except Exception as e:
            log.error("Health check failed", error=str(e))
            return embedding_pb2.HealthResponse(
                status="unhealthy",
                details={"error": str(e)},
                uptime_seconds=int(time.time() - self.start_time)
            )
    
    async def ClearCache(
        self,
        request: embedding_pb2.ClearCacheRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_pb2.ClearCacheResponse:
        """Clear embedding caches"""
        
        try:
            l1_cleared = 0
            l2_cleared = 0
            
            if not request.l1_only:
                # Clear Redis cache
                redis_client = await self._get_redis_client()
                
                try:
                    if request.model:
                        pattern = f"{request.model}:*"
                    else:
                        pattern = "*"
                    
                    keys = await redis_client.keys(pattern)
                    if keys:
                        l2_cleared = await redis_client.delete(*keys)
                        
                except Exception as e:
                    log.error("Redis cache clear failed", error=str(e))
                finally:
                    await redis_client.aclose()
            
            # Clear L1 cache
            if request.model:
                # Clear specific model keys
                keys_to_remove = [
                    k for k in self.l1_cache.keys() 
                    if k.startswith(f"{request.model}:")
                ]
                for key in keys_to_remove:
                    del self.l1_cache[key]
                l1_cleared = len(keys_to_remove)
            else:
                # Clear all
                l1_cleared = len(self.l1_cache)
                self.l1_cache.clear()
            
            log.info("Cache cleared",
                    l1_cleared=l1_cleared,
                    l2_cleared=l2_cleared,
                    model=request.model or "all")
            
            return embedding_pb2.ClearCacheResponse(
                l1_keys_cleared=l1_cleared,
                l2_keys_cleared=l2_cleared,
                status="success"
            )
            
        except Exception as e:
            log.error("Cache clear failed", error=str(e))
            return embedding_pb2.ClearCacheResponse(
                status=f"error: {str(e)}"
            )


async def serve():
    """Start the gRPC embedding service"""
    
    config = EmbeddingConfig()
    
    # Start Prometheus metrics server
    start_http_server(config.metrics_port)
    log.info(f"Metrics server started on port {config.metrics_port}")
    
    # Create gRPC server
    server = aio.server(
        futures.ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
    )
    
    # Add service
    embedding_service = EmbeddingServiceImpl(config)
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        embedding_service, server
    )
    
    # Configure server
    listen_addr = f'[::]:{config.grpc_port}'
    server.add_insecure_port(listen_addr)
    
    log.info(f"Starting gRPC embedding service on {listen_addr}")
    
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        log.info("Shutting down gRPC server...")
        await server.stop(5)


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    
    # Note: In production, you'd generate these from the proto file:
    # python -m grpc_tools.protoc --python_out=. --grpc_python_out=. proto/embedding.proto
    
    asyncio.run(serve())