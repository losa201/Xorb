"""
Semantic Cache for LLM Responses
Uses vector similarity to cache and retrieve similar LLM responses
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    content: str
    embedding: np.ndarray
    context_hash: str
    timestamp: float
    usage_count: int
    confidence: float

class SemanticCache:
    """Vector-based semantic cache for LLM responses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        self.ttl_seconds = config.get('ttl_seconds', 3600)
        self.max_cache_size = config.get('max_cache_size', 10000)
        
        # Embedding model for semantic similarity
        model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        
        # ChromaDB client
        chroma_host = config.get('chroma_host', 'chroma.xorb-prod.svc.cluster.local')
        chroma_port = config.get('chroma_port', 8000)
        
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(allow_reset=True)
        )
        
        # Collections for different types of cached responses
        self.collections = {}
        self._initialize_collections()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'evictions': 0
        }
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections"""
        collection_names = [
            'llm_responses',
            'reasoning_results',
            'analysis_outputs'
        ]
        
        for name in collection_names:
            try:
                # Try to get existing collection
                collection = self.chroma_client.get_collection(name)
                logger.info(f"Connected to existing collection: {name}")
            except Exception:
                # Create new collection if it doesn't exist
                collection = self.chroma_client.create_collection(
                    name=name,
                    metadata={"description": f"Semantic cache for {name}"}
                )
                logger.info(f"Created new collection: {name}")
            
            self.collections[name] = collection
    
    def _generate_context_hash(self, context: Dict[str, Any]) -> str:
        """Generate hash for context to group similar requests"""
        # Sort context keys for consistent hashing
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _create_query_embedding(self, prompt: str) -> np.ndarray:
        """Create embedding for prompt"""
        return self.embedding_model.encode([prompt])[0]
    
    async def get_response(self, prompt: str, context: Dict[str, Any], 
                          collection_name: str = 'llm_responses') -> Optional[str]:
        """Retrieve cached response if similar prompt exists"""
        try:
            context_hash = self._generate_context_hash(context)
            query_embedding = self._create_query_embedding(prompt)
            
            collection = self.collections.get(collection_name)
            if not collection:
                logger.warning(f"Collection {collection_name} not found")
                return None
            
            # Query for similar embeddings
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5,
                where={"context_hash": context_hash},
                include=['metadatas', 'documents', 'distances']
            )
            
            if not results['documents'][0]:
                self.stats['misses'] += 1
                return None
            
            # Check if best match exceeds similarity threshold
            best_distance = results['distances'][0][0]
            similarity = 1 - best_distance
            
            if similarity >= self.similarity_threshold:
                # Check if cache entry is still valid (TTL)
                best_metadata = results['metadatas'][0][0]
                cache_time = float(best_metadata.get('timestamp', 0))
                
                if time.time() - cache_time < self.ttl_seconds:
                    cached_content = results['documents'][0][0]
                    
                    # Update usage count
                    await self._update_usage_count(
                        results['ids'][0][0], collection_name
                    )
                    
                    self.stats['hits'] += 1
                    logger.debug(f"Cache hit with similarity: {similarity:.3f}")
                    return cached_content
                else:
                    # Entry expired, remove it
                    await self._remove_expired_entry(
                        results['ids'][0][0], collection_name
                    )
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from semantic cache: {e}")
            self.stats['misses'] += 1
            return None
    
    async def store_response(self, prompt: str, context: Dict[str, Any], 
                           response: str, collection_name: str = 'llm_responses',
                           confidence: float = 1.0):
        """Store response in semantic cache"""
        try:
            context_hash = self._generate_context_hash(context)
            embedding = self._create_query_embedding(prompt)
            
            collection = self.collections.get(collection_name)
            if not collection:
                logger.warning(f"Collection {collection_name} not found")
                return
            
            # Generate unique ID
            entry_id = hashlib.sha256(
                f"{prompt}{context_hash}{time.time()}".encode()
            ).hexdigest()
            
            # Metadata for the cache entry
            metadata = {
                'context_hash': context_hash,
                'timestamp': str(time.time()),
                'usage_count': '1',
                'confidence': str(confidence),
                'prompt_length': str(len(prompt)),
                'response_length': str(len(response))
            }
            
            # Store in ChromaDB
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[response],
                metadatas=[metadata],
                ids=[entry_id]
            )
            
            self.stats['stores'] += 1
            logger.debug(f"Stored response in cache: {entry_id}")
            
            # Check if we need to evict old entries
            await self._check_cache_size(collection_name)
            
        except Exception as e:
            logger.error(f"Error storing in semantic cache: {e}")
    
    async def _update_usage_count(self, entry_id: str, collection_name: str):
        """Update usage count for cache entry"""
        try:
            collection = self.collections[collection_name]
            
            # Get current metadata
            result = collection.get(ids=[entry_id], include=['metadatas'])
            if result['metadatas']:
                metadata = result['metadatas'][0]
                current_count = int(metadata.get('usage_count', 0))
                metadata['usage_count'] = str(current_count + 1)
                
                # Update metadata
                collection.update(
                    ids=[entry_id],
                    metadatas=[metadata]
                )
                
        except Exception as e:
            logger.error(f"Error updating usage count: {e}")
    
    async def _remove_expired_entry(self, entry_id: str, collection_name: str):
        """Remove expired cache entry"""
        try:
            collection = self.collections[collection_name]
            collection.delete(ids=[entry_id])
            self.stats['evictions'] += 1
            logger.debug(f"Removed expired entry: {entry_id}")
        except Exception as e:
            logger.error(f"Error removing expired entry: {e}")
    
    async def _check_cache_size(self, collection_name: str):
        """Check cache size and evict LRU entries if needed"""
        try:
            collection = self.collections[collection_name]
            
            # Get collection count
            count_result = collection.count()
            if count_result <= self.max_cache_size:
                return
            
            # Get all entries sorted by last usage
            all_entries = collection.get(include=['metadatas'])
            
            if not all_entries['metadatas']:
                return
            
            # Sort by timestamp (oldest first)
            entries_with_time = [
                (entry_id, float(metadata.get('timestamp', 0)))
                for entry_id, metadata in zip(all_entries['ids'], all_entries['metadatas'])
            ]
            entries_with_time.sort(key=lambda x: x[1])
            
            # Calculate how many to evict (10% of max size)
            evict_count = max(1, self.max_cache_size // 10)
            entries_to_evict = entries_with_time[:evict_count]
            
            # Delete oldest entries
            ids_to_delete = [entry_id for entry_id, _ in entries_to_evict]
            collection.delete(ids=ids_to_delete)
            
            self.stats['evictions'] += len(ids_to_delete)
            logger.info(f"Evicted {len(ids_to_delete)} entries from {collection_name}")
            
        except Exception as e:
            logger.error(f"Error checking cache size: {e}")
    
    async def invalidate_context(self, context: Dict[str, Any], 
                                collection_name: str = 'llm_responses'):
        """Invalidate all cache entries for a specific context"""
        try:
            context_hash = self._generate_context_hash(context)
            collection = self.collections[collection_name]
            
            # Query for entries with this context
            results = collection.get(
                where={"context_hash": context_hash},
                include=['metadatas']
            )
            
            if results['ids']:
                collection.delete(ids=results['ids'])
                logger.info(f"Invalidated {len(results['ids'])} entries for context")
                
        except Exception as e:
            logger.error(f"Error invalidating context: {e}")
    
    async def clear_collection(self, collection_name: str):
        """Clear entire collection"""
        try:
            collection = self.collections[collection_name]
            
            # Delete all entries
            all_entries = collection.get()
            if all_entries['ids']:
                collection.delete(ids=all_entries['ids'])
                logger.info(f"Cleared collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    async def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests == 0:
            return 0.0
        return self.stats['hits'] / total_requests
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            collection_stats = {}
            
            for name, collection in self.collections.items():
                count = collection.count()
                collection_stats[name] = {'count': count}
            
            return {
                'hit_rate': await self.get_hit_rate(),
                'total_hits': self.stats['hits'],
                'total_misses': self.stats['misses'],
                'total_stores': self.stats['stores'],
                'total_evictions': self.stats['evictions'],
                'collections': collection_stats,
                'similarity_threshold': self.similarity_threshold,
                'ttl_seconds': self.ttl_seconds,
                'max_cache_size': self.max_cache_size
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {'error': str(e)}
    
    async def optimize_cache(self):
        """Optimize cache by removing low-usage entries"""
        try:
            for collection_name, collection in self.collections.items():
                # Get all entries with usage counts
                all_entries = collection.get(include=['metadatas'])
                
                if not all_entries['metadatas']:
                    continue
                
                # Find entries with low usage (used only once and old)
                current_time = time.time()
                ids_to_remove = []
                
                for entry_id, metadata in zip(all_entries['ids'], all_entries['metadatas']):
                    usage_count = int(metadata.get('usage_count', 0))
                    timestamp = float(metadata.get('timestamp', 0))
                    age_hours = (current_time - timestamp) / 3600
                    
                    # Remove entries used only once and older than 24 hours
                    if usage_count == 1 and age_hours > 24:
                        ids_to_remove.append(entry_id)
                
                if ids_to_remove:
                    collection.delete(ids=ids_to_remove)
                    self.stats['evictions'] += len(ids_to_remove)
                    logger.info(f"Optimized {collection_name}: removed {len(ids_to_remove)} low-usage entries")
                    
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")
    
    async def search_similar(self, prompt: str, context: Dict[str, Any],
                           n_results: int = 5, collection_name: str = 'llm_responses') -> List[Dict[str, Any]]:
        """Search for similar cached responses"""
        try:
            context_hash = self._generate_context_hash(context)
            query_embedding = self._create_query_embedding(prompt)
            
            collection = self.collections.get(collection_name)
            if not collection:
                return []
            
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where={"context_hash": context_hash},
                include=['metadatas', 'documents', 'distances']
            )
            
            similar_responses = []
            for i, (doc, distance, metadata) in enumerate(zip(
                results['documents'][0],
                results['distances'][0], 
                results['metadatas'][0]
            )):
                similarity = 1 - distance
                similar_responses.append({
                    'content': doc,
                    'similarity': similarity,
                    'timestamp': float(metadata.get('timestamp', 0)),
                    'usage_count': int(metadata.get('usage_count', 0)),
                    'confidence': float(metadata.get('confidence', 0))
                })
            
            return similar_responses
            
        except Exception as e:
            logger.error(f"Error searching similar responses: {e}")
            return []