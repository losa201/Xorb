"""
Embedding Service Integration for Knowledge Fabric
Integrates NVIDIA embeddings with the knowledge storage system
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
import structlog
import hashlib
import json
from cachetools import TTLCache
import redis.asyncio as redis
from prometheus_client import Counter

from openai import OpenAI
from ..logging import get_logger

log = get_logger(__name__)

# Prometheus metrics for cache performance
embedding_cache_hits_total = Counter(
    'xorb_embedding_cache_hits_total',
    'Total embedding cache hits',
    ['cache_type', 'model']
)

@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    text: str
    embedding: List[float]
    model: str
    timestamp: datetime
    metadata: Dict[str, Any]

class KnowledgeEmbeddingService:
    """Service for generating and managing embeddings in the knowledge fabric"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/embed-qa-4",
        cache_embeddings: bool = True,
        redis_url: str = "redis://redis:6379/0"
    ):
        self.api_key = api_key or "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"
        self.base_url = base_url
        self.model = model
        self.cache_embeddings = cache_embeddings
        self.redis_url = redis_url
        
        # Initialize OpenAI client for NVIDIA API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Local TTL cache as L1 (hot cache)
        self._local_cache = TTLCache(maxsize=10_000, ttl=24*60*60)
        
        # Redis connection for persistent cache (L2)
        self._redis = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=False)
        
        log.info("Knowledge Embedding Service initialized",
                model=self.model,
                base_url=self.base_url,
                cache_enabled=self.cache_embeddings)
    
    def _cache_key(self, text: str, input_type: str = "query") -> str:
        """Generate SHA-1 cache key for text and input type"""
        text_hash = hashlib.sha1(text.encode('utf-8')).hexdigest()
        return f"{self.model}:{input_type}:{text_hash}"
    
    async def embed_text(
        self, 
        text: str, 
        input_type: str = "query",
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """Generate embedding for a single text with SHA-1 deduplication"""
        
        cache_key = self._cache_key(text, input_type)
        
        # Check L1 cache (local) first
        if self.cache_embeddings and cache_key in self._local_cache:
            log.debug("Embedding L1 cache hit", text_length=len(text))
            embedding_cache_hits_total.labels(cache_type="l1", model=self.model).inc()
            return self._local_cache[cache_key]
        
        # Check L2 cache (Redis) 
        if self.cache_embeddings:
            try:
                cached_data = await self._redis.get(cache_key)
                if cached_data:
                    embedding_data = json.loads(cached_data)
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding_data['embedding'],
                        model=self.model,
                        timestamp=datetime.fromisoformat(embedding_data['timestamp']),
                        metadata=embedding_data.get('metadata', {})
                    )
                    # Store in L1 cache for next access
                    self._local_cache[cache_key] = result
                    log.debug("Embedding L2 cache hit", text_length=len(text))
                    embedding_cache_hits_total.labels(cache_type="l2", model=self.model).inc()
                    return result
            except Exception as e:
                log.warning("Redis cache read failed", error=str(e))
        
        try:
            log.debug("Generating embedding", 
                     text_length=len(text),
                     input_type=input_type,
                     model=self.model)
            
            response = self.client.embeddings.create(
                input=[text],
                model=self.model,
                encoding_format="float",
                extra_body={
                    "input_type": input_type,
                    "truncate": "NONE"
                }
            )
            
            embedding = response.data[0].embedding
            
            result = EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {}
            )
            
            # Cache the result in both L1 and L2
            if self.cache_embeddings:
                # Store in local cache
                self._local_cache[cache_key] = result
                
                # Store in Redis with 24h TTL
                try:
                    cache_data = {
                        'embedding': embedding,
                        'timestamp': result.timestamp.isoformat(),
                        'metadata': result.metadata
                    }
                    await self._redis.setex(cache_key, 24*3600, json.dumps(cache_data))
                except Exception as e:
                    log.warning("Redis cache write failed", error=str(e))
            
            log.debug("Embedding generated successfully",
                     embedding_dimension=len(embedding),
                     cached=self.cache_embeddings)
            
            return result
            
        except Exception as e:
            log.error("Failed to generate embedding",
                     error=str(e),
                     text_length=len(text),
                     model=self.model)
            raise
    
    async def embed_texts(
        self,
        texts: List[str],
        input_type: str = "query",
        batch_size: int = 50,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts"""
        
        if not texts:
            return []
        
        if metadata and len(metadata) != len(texts):
            raise ValueError("Metadata list must match texts list length")
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size] if metadata else [{}] * len(batch_texts)
            
            log.info("Processing embedding batch",
                    batch_start=i,
                    batch_size=len(batch_texts),
                    total_texts=len(texts))
            
            # Check cache for each text
            batch_results = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                cache_key = self._cache_key(text, input_type)
                cached_result = None
                
                # Check L1 cache first
                if self.cache_embeddings and cache_key in self._local_cache:
                    cached_result = self._local_cache[cache_key]
                
                # Check L2 cache if not in L1
                elif self.cache_embeddings:
                    try:
                        cached_data = await self._redis.get(cache_key)
                        if cached_data:
                            embedding_data = json.loads(cached_data)
                            cached_result = EmbeddingResult(
                                text=text,
                                embedding=embedding_data['embedding'],
                                model=self.model,
                                timestamp=datetime.fromisoformat(embedding_data['timestamp']),
                                metadata=embedding_data.get('metadata', {})
                            )
                            # Store in L1 for next access
                            self._local_cache[cache_key] = cached_result
                    except Exception as e:
                        log.warning("Redis batch cache read failed", error=str(e))
                
                if cached_result:
                    batch_results.append(cached_result)
                else:
                    batch_results.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    response = self.client.embeddings.create(
                        input=uncached_texts,
                        model=self.model,
                        encoding_format="float",
                        extra_body={
                            "input_type": input_type,
                            "truncate": "NONE"
                        }
                    )
                    
                    # Process uncached results
                    for k, embedding_data in enumerate(response.data):
                        original_index = uncached_indices[k]
                        text = uncached_texts[k]
                        text_metadata = batch_metadata[original_index]
                        
                        result = EmbeddingResult(
                            text=text,
                            embedding=embedding_data.embedding,
                            model=self.model,
                            timestamp=datetime.now(timezone.utc),
                            metadata=text_metadata
                        )
                        
                        # Update batch results
                        batch_results[original_index] = result
                        
                        # Cache the result in both L1 and L2
                        if self.cache_embeddings:
                            cache_key = self._cache_key(text, input_type)
                            self._local_cache[cache_key] = result
                            
                            # Store in Redis with 24h TTL
                            try:
                                cache_data = {
                                    'embedding': result.embedding,
                                    'timestamp': result.timestamp.isoformat(),
                                    'metadata': result.metadata
                                }
                                await self._redis.setex(cache_key, 24*3600, json.dumps(cache_data))
                            except Exception as e:
                                log.warning("Redis batch cache write failed", error=str(e))
                
                except Exception as e:
                    log.error("Failed to generate batch embeddings",
                             error=str(e),
                             batch_size=len(uncached_texts))
                    raise
            
            results.extend(batch_results)
        
        log.info("Batch embedding generation completed",
                total_embeddings=len(results),
                cache_hits=len(texts) - len([r for r in results if r.timestamp.replace(microsecond=0) == datetime.now(timezone.utc).replace(microsecond=0)]))
        
        return results
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between two embeddings"""
        
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1 / (1 + distance)
        elif metric == "dot":
            # Dot product
            similarity = np.dot(emb1, emb2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return float(similarity)
    
    def find_similar_texts(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[Tuple[str, List[float]]],
        top_k: int = 10,
        threshold: float = 0.7,
        metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """Find most similar texts to a query embedding"""
        
        similarities = []
        
        for text, embedding in candidate_embeddings:
            similarity = self.compute_similarity(query_embedding, embedding, metric)
            
            if similarity >= threshold:
                similarities.append((text, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def semantic_search(
        self,
        query: str,
        knowledge_texts: List[str],
        top_k: int = 10,
        threshold: float = 0.7,
        input_type: str = "query"
    ) -> List[Tuple[str, float]]:
        """Perform semantic search over knowledge texts"""
        
        log.info("Starting semantic search",
                query_length=len(query),
                knowledge_texts_count=len(knowledge_texts),
                top_k=top_k,
                threshold=threshold)
        
        # Generate query embedding
        query_result = await self.embed_text(query, input_type="query")
        
        # Generate embeddings for knowledge texts
        knowledge_results = await self.embed_texts(
            knowledge_texts,
            input_type="passage"  # Use passage type for knowledge texts
        )
        
        # Prepare candidate embeddings
        candidates = [(result.text, result.embedding) for result in knowledge_results]
        
        # Find similar texts
        similar_texts = self.find_similar_texts(
            query_result.embedding,
            candidates,
            top_k=top_k,
            threshold=threshold
        )
        
        log.info("Semantic search completed",
                results_found=len(similar_texts),
                top_similarity=similar_texts[0][1] if similar_texts else 0.0)
        
        return similar_texts
    
    async def cluster_texts(
        self,
        texts: List[str],
        num_clusters: int = 5,
        input_type: str = "passage"
    ) -> Dict[int, List[str]]:
        """Cluster texts based on semantic similarity"""
        
        if len(texts) < num_clusters:
            # If fewer texts than clusters, each text is its own cluster
            return {i: [text] for i, text in enumerate(texts)}
        
        log.info("Starting text clustering",
                num_texts=len(texts),
                num_clusters=num_clusters)
        
        # Generate embeddings for all texts
        embedding_results = await self.embed_texts(texts, input_type=input_type)
        embeddings = np.array([result.embedding for result in embedding_results])
        
        # Simple K-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[i])
        
        log.info("Text clustering completed",
                clusters_created=len(clusters),
                avg_cluster_size=len(texts) / len(clusters))
        
        return clusters
    
    async def clear_cache(self):
        """Clear both L1 and L2 embedding caches"""
        self._local_cache.clear()
        try:
            # Clear Redis cache with pattern matching
            pattern = f"{self.model}:*"
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
            log.info("Embedding caches cleared", local_cleared=True, redis_keys_cleared=len(keys))
        except Exception as e:
            log.warning("Failed to clear Redis cache", error=str(e))
            log.info("Local embedding cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for both L1 and L2 caches"""
        local_size = len(self._local_cache)
        local_memory_mb = sum(
            len(result.embedding) * 4  # 4 bytes per float
            for result in self._local_cache.values()
        ) / (1024 * 1024)
        
        redis_keys = 0
        try:
            pattern = f"{self.model}:*"
            keys = await self._redis.keys(pattern)
            redis_keys = len(keys)
        except Exception as e:
            log.warning("Failed to get Redis cache stats", error=str(e))
        
        return {
            "cache_enabled": self.cache_embeddings,
            "model": self.model,
            "l1_cache_size": local_size,
            "l1_memory_mb": local_memory_mb,
            "l2_cache_keys": redis_keys,
            "total_cached_items": local_size + redis_keys
        }

# Global embedding service instance
_embedding_service = None

def get_embedding_service() -> KnowledgeEmbeddingService:
    """Get the global embedding service instance"""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = KnowledgeEmbeddingService()
    
    return _embedding_service