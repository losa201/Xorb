"""
Production Embedding Service for XORB Platform
Handles text embedding generation with multiple backend support
"""

import asyncio
import logging
import hashlib
import json
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import aiohttp
import numpy as np
from uuid import UUID

from .base_service import XORBService

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    OPENAI = "openai"
    NVIDIA = "nvidia"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


@dataclass
class EmbeddingRequest:
    text: str
    model: str
    provider: EmbeddingProvider
    tenant_id: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    embedding: List[float]
    model: str
    provider: EmbeddingProvider
    dimensions: int
    usage_tokens: int
    processing_time_ms: float
    cached: bool = False


class ProductionEmbeddingService(XORBService):
    """Production-ready embedding service with multiple providers"""
    
    def __init__(self):
        super().__init__()
        self.providers = {}
        self.cache = {}
        self.model_cache = {}
        self.usage_stats = {}
        
        # Default configurations
        self.provider_configs = {
            EmbeddingProvider.OPENAI: {
                "models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                "dimensions": {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536},
                "max_tokens": 8191,
                "rate_limit": 3000,  # requests per minute
                "cost_per_1k_tokens": 0.0001
            },
            EmbeddingProvider.NVIDIA: {
                "models": ["nvidia/nv-embedqa-e5-v5", "nvidia/nv-embed-v1"],
                "dimensions": {"nvidia/nv-embedqa-e5-v5": 1024, "nvidia/nv-embed-v1": 1024},
                "max_tokens": 512,
                "rate_limit": 1000,
                "cost_per_1k_tokens": 0.0002
            },
            EmbeddingProvider.SENTENCE_TRANSFORMERS: {
                "models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
                "dimensions": {"all-MiniLM-L6-v2": 384, "all-mpnet-base-v2": 768, "multi-qa-MiniLM-L6-cos-v1": 384},
                "max_tokens": 512,
                "rate_limit": float('inf'),  # Local processing
                "cost_per_1k_tokens": 0.0  # Free local processing
            }
        }
        
        self.default_model = "all-MiniLM-L6-v2"
        self.default_provider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    
    async def initialize(self) -> bool:
        """Initialize embedding service with available providers"""
        try:
            logger.info("Initializing Embedding Service...")
            
            # Initialize providers in order of preference
            await self._initialize_sentence_transformers()
            await self._initialize_openai()
            await self._initialize_nvidia()
            await self._initialize_azure_openai()
            
            # Set default provider based on what's available
            if EmbeddingProvider.SENTENCE_TRANSFORMERS in self.providers:
                self.default_provider = EmbeddingProvider.SENTENCE_TRANSFORMERS
                self.default_model = "all-MiniLM-L6-v2"
            elif EmbeddingProvider.OPENAI in self.providers:
                self.default_provider = EmbeddingProvider.OPENAI
                self.default_model = "text-embedding-3-small"
            elif EmbeddingProvider.NVIDIA in self.providers:
                self.default_provider = EmbeddingProvider.NVIDIA
                self.default_model = "nvidia/nv-embedqa-e5-v5"
            
            logger.info(f"Embedding service initialized with {len(self.providers)} providers")
            logger.info(f"Default provider: {self.default_provider.value}, model: {self.default_model}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            return False
    
    async def _initialize_sentence_transformers(self):
        """Initialize local sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load default model
            model = SentenceTransformer(self.default_model)
            self.model_cache[self.default_model] = model
            
            self.providers[EmbeddingProvider.SENTENCE_TRANSFORMERS] = {
                "available": True,
                "models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                "client": model
            }
            
            logger.info("Sentence Transformers provider initialized")
            
        except ImportError:
            logger.warning("sentence-transformers not available")
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformers: {e}")
    
    async def _initialize_openai(self):
        """Initialize OpenAI embedding provider"""
        try:
            import openai
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found")
                return
            
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Test connection
            try:
                await client.embeddings.create(
                    model="text-embedding-3-small",
                    input="test",
                    dimensions=1536
                )
                
                self.providers[EmbeddingProvider.OPENAI] = {
                    "available": True,
                    "models": ["text-embedding-3-small", "text-embedding-3-large"],
                    "client": client
                }
                
                logger.info("OpenAI embedding provider initialized")
                
            except Exception as e:
                logger.warning(f"OpenAI API test failed: {e}")
                
        except ImportError:
            logger.warning("openai package not available")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    
    async def _initialize_nvidia(self):
        """Initialize NVIDIA embedding provider"""
        try:
            import os
            
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                logger.warning("NVIDIA API key not found")
                return
            
            # Test NVIDIA API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                test_payload = {
                    "input": ["test"],
                    "model": "nvidia/nv-embedqa-e5-v5"
                }
                
                async with session.post(
                    "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0149dedb-2268-4761-a2c6-75637bfe7e6b",
                    headers=headers,
                    json=test_payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        self.providers[EmbeddingProvider.NVIDIA] = {
                            "available": True,
                            "models": ["nvidia/nv-embedqa-e5-v5"],
                            "api_key": api_key,
                            "session": None  # Will create per request
                        }
                        
                        logger.info("NVIDIA embedding provider initialized")
                    else:
                        logger.warning(f"NVIDIA API test failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA: {e}")
    
    async def _initialize_azure_openai(self):
        """Initialize Azure OpenAI embedding provider"""
        try:
            import openai
            import os
            
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
            if not api_key or not endpoint:
                logger.warning("Azure OpenAI credentials not found")
                return
            
            client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version="2023-05-15"
            )
            
            self.providers[EmbeddingProvider.AZURE_OPENAI] = {
                "available": True,
                "models": ["text-embedding-ada-002"],
                "client": client
            }
            
            logger.info("Azure OpenAI embedding provider initialized")
            
        except ImportError:
            logger.warning("openai package not available for Azure")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
    
    async def generate_embedding(
        self, 
        text: str, 
        model: Optional[str] = None, 
        provider: Optional[EmbeddingProvider] = None,
        tenant_id: Optional[UUID] = None
    ) -> Optional[List[float]]:
        """Generate embedding for text using the best available provider"""
        
        if not text or not text.strip():
            return None
        
        # Use defaults if not specified
        provider = provider or self.default_provider
        model = model or self.default_model
        
        # Check cache first
        cache_key = self._get_cache_key(text, model, provider)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if self._is_cache_valid(cached_result):
                logger.debug(f"Cache hit for embedding: {cache_key[:16]}...")
                return cached_result["embedding"]
        
        # Generate new embedding
        start_time = datetime.utcnow()
        
        try:
            if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                embedding = await self._generate_sentence_transformer_embedding(text, model)
            elif provider == EmbeddingProvider.OPENAI:
                embedding = await self._generate_openai_embedding(text, model)
            elif provider == EmbeddingProvider.NVIDIA:
                embedding = await self._generate_nvidia_embedding(text, model)
            elif provider == EmbeddingProvider.AZURE_OPENAI:
                embedding = await self._generate_azure_embedding(text, model)
            else:
                logger.error(f"Unknown provider: {provider}")
                return None
            
            if embedding:
                # Cache the result
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.cache[cache_key] = {
                    "embedding": embedding,
                    "timestamp": datetime.utcnow(),
                    "processing_time": processing_time
                }
                
                # Update usage stats
                self._update_usage_stats(provider, model, len(text.split()))
                
                logger.debug(f"Generated {len(embedding)}-dim embedding in {processing_time:.1f}ms")
                return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            
            # Fallback to hash-based embedding
            return self._generate_fallback_embedding(text)
        
        return None
    
    async def _generate_sentence_transformer_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Generate embedding using local sentence transformers"""
        try:
            if EmbeddingProvider.SENTENCE_TRANSFORMERS not in self.providers:
                return None
            
            # Load model if not cached
            if model not in self.model_cache:
                from sentence_transformers import SentenceTransformer
                self.model_cache[model] = SentenceTransformer(model)
            
            model_instance = self.model_cache[model]
            
            # Generate embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: model_instance.encode(text, convert_to_tensor=False)
            )
            
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
        except Exception as e:
            logger.error(f"Sentence transformer embedding failed: {e}")
            return None
    
    async def _generate_openai_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API"""
        try:
            if EmbeddingProvider.OPENAI not in self.providers:
                return None
            
            client = self.providers[EmbeddingProvider.OPENAI]["client"]
            
            response = await client.embeddings.create(
                model=model,
                input=text,
                dimensions=1536 if model == "text-embedding-3-small" else None
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return None
    
    async def _generate_nvidia_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Generate embedding using NVIDIA API"""
        try:
            if EmbeddingProvider.NVIDIA not in self.providers:
                return None
            
            provider_config = self.providers[EmbeddingProvider.NVIDIA]
            api_key = provider_config["api_key"]
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": [text],
                "model": model
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0149dedb-2268-4761-a2c6-75637bfe7e6b",
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "data" in result and result["data"]:
                            return result["data"][0]["embedding"]
                    else:
                        logger.error(f"NVIDIA API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"NVIDIA embedding failed: {e}")
            return None
    
    async def _generate_azure_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Generate embedding using Azure OpenAI"""
        try:
            if EmbeddingProvider.AZURE_OPENAI not in self.providers:
                return None
            
            client = self.providers[EmbeddingProvider.AZURE_OPENAI]["client"]
            
            response = await client.embeddings.create(
                model=model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Azure OpenAI embedding failed: {e}")
            return None
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate deterministic hash-based embedding as fallback"""
        import hashlib
        import struct
        
        # Create deterministic hash
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert to 384-dimensional vector
        embedding = []
        for i in range(0, 96):  # 96 * 4 = 384 floats
            start_idx = (i * 4) % len(hash_bytes)
            end_idx = min(start_idx + 4, len(hash_bytes))
            
            # Pad with zeros if needed
            chunk = hash_bytes[start_idx:end_idx] + b'\x00' * (4 - (end_idx - start_idx))
            
            # Convert to float
            val = struct.unpack('f', chunk)[0]
            
            # Normalize to [-1, 1] range
            normalized_val = (val % 256 - 128) / 128.0
            embedding.append(normalized_val)
        
        logger.warning("Using fallback hash-based embedding")
        return embedding
    
    def _get_cache_key(self, text: str, model: str, provider: EmbeddingProvider) -> str:
        """Generate cache key for embedding"""
        content = f"{provider.value}:{model}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        cache_ttl = timedelta(hours=24)  # Cache for 24 hours
        return datetime.utcnow() - cache_entry["timestamp"] < cache_ttl
    
    def _update_usage_stats(self, provider: EmbeddingProvider, model: str, token_count: int):
        """Update usage statistics"""
        key = f"{provider.value}:{model}"
        
        if key not in self.usage_stats:
            self.usage_stats[key] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
                "last_used": datetime.utcnow()
            }
        
        stats = self.usage_stats[key]
        stats["requests"] += 1
        stats["tokens"] += token_count
        stats["last_used"] = datetime.utcnow()
        
        # Calculate cost
        config = self.provider_configs.get(provider, {})
        cost_per_1k = config.get("cost_per_1k_tokens", 0.0)
        stats["cost"] += (token_count / 1000.0) * cost_per_1k
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models by provider"""
        available = {}
        
        for provider, config in self.providers.items():
            if config.get("available", False):
                available[provider.value] = config.get("models", [])
        
        return available
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "providers": list(self.providers.keys()),
            "default_provider": self.default_provider.value,
            "default_model": self.default_model,
            "cache_size": len(self.cache),
            "usage_by_model": self.usage_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on embedding service"""
        health = {
            "status": "healthy",
            "providers": {},
            "cache_size": len(self.cache),
            "model_cache_size": len(self.model_cache)
        }
        
        # Test each provider
        for provider, config in self.providers.items():
            try:
                # Quick test embedding
                test_embedding = await self.generate_embedding(
                    "test", 
                    provider=provider
                )
                
                health["providers"][provider.value] = {
                    "available": test_embedding is not None,
                    "models": config.get("models", []),
                    "last_test": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                health["providers"][provider.value] = {
                    "available": False,
                    "error": str(e),
                    "last_test": datetime.utcnow().isoformat()
                }
        
        # Overall health
        available_providers = sum(1 for p in health["providers"].values() if p.get("available", False))
        if available_providers == 0:
            health["status"] = "unhealthy"
        elif available_providers < len(self.providers):
            health["status"] = "degraded"
        
        return health
    
    async def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    async def shutdown(self) -> bool:
        """Shutdown embedding service"""
        try:
            # Clear caches
            self.cache.clear()
            self.model_cache.clear()
            
            # Close provider connections
            for provider, config in self.providers.items():
                if "client" in config and hasattr(config["client"], "close"):
                    try:
                        await config["client"].close()
                    except:
                        pass
            
            logger.info("Embedding service shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown embedding service: {e}")
            return False


# Global service instance
_embedding_service: Optional[ProductionEmbeddingService] = None


async def get_embedding_service() -> Optional[ProductionEmbeddingService]:
    """Get global embedding service instance"""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = ProductionEmbeddingService()
        await _embedding_service.initialize()
    
    return _embedding_service


async def generate_embedding(text: str, model: Optional[str] = None) -> Optional[List[float]]:
    """Convenience function for generating embeddings"""
    service = await get_embedding_service()
    if service:
        return await service.generate_embedding(text, model=model)
    return None