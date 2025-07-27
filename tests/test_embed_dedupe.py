"""
Test SHA-1 deduplication functionality for embedding service
"""

import pytest
import asyncio
import json
import hashlib
from unittest.mock import AsyncMock, patch, MagicMock
from xorb_core.knowledge_fabric.embedding_service import KnowledgeEmbeddingService, EmbeddingResult


@pytest.mark.asyncio
async def test_sha1_cache_deduplication():
    """Test that identical texts use SHA-1 deduplication and cache properly"""
    
    fake_embedding = [0.1, 0.2, 0.3, 0.4]
    
    # Mock OpenAI client response
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = fake_embedding
    
    # Mock Redis to always return None (cache miss)
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.setex = AsyncMock()
    
    with patch('redis.asyncio.from_url', return_value=mock_redis):
        service = KnowledgeEmbeddingService(cache_embeddings=True)
        
        # Mock the OpenAI client
        service.client.embeddings.create = AsyncMock(return_value=mock_response)
        
        # First call - should hit API
        result1 = await service.embed_text("hello world")
        
        # Second call with identical text - should hit local cache
        result2 = await service.embed_text("hello world")
        
        # Verify results are identical
        assert result1.embedding == result2.embedding == fake_embedding
        assert result1.text == result2.text == "hello world"
        
        # API should only be called once due to caching
        service.client.embeddings.create.assert_called_once()
        
        # Redis should be called for both set operations
        assert mock_redis.setex.call_count == 1


@pytest.mark.asyncio
async def test_sha1_redis_cache_hit():
    """Test that Redis cache hits work correctly"""
    
    cached_embedding = [0.5, 0.6, 0.7, 0.8]
    text = "cached text"
    
    # Create cache data as it would be stored in Redis
    cache_data = {
        'embedding': cached_embedding,
        'timestamp': '2025-07-25T10:00:00+00:00',
        'metadata': {}
    }
    
    # Mock Redis to return cached data
    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps(cache_data)
    
    with patch('redis.asyncio.from_url', return_value=mock_redis):
        service = KnowledgeEmbeddingService(cache_embeddings=True)
        
        # Mock the OpenAI client (should not be called)
        service.client.embeddings.create = AsyncMock()
        
        # Call embed_text
        result = await service.embed_text(text)
        
        # Verify result comes from cache
        assert result.embedding == cached_embedding
        assert result.text == text
        
        # API should not be called
        service.client.embeddings.create.assert_not_called()
        
        # Redis get should be called with SHA-1 hash key
        expected_key = f"nvidia/embed-qa-4:query:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
        mock_redis.get.assert_called_once_with(expected_key)


@pytest.mark.asyncio 
async def test_cache_key_generation():
    """Test that cache keys are generated correctly with SHA-1"""
    
    service = KnowledgeEmbeddingService()
    
    text = "test text for caching"
    expected_hash = hashlib.sha1(text.encode('utf-8')).hexdigest()
    expected_key = f"nvidia/embed-qa-4:query:{expected_hash}"
    
    cache_key = service._cache_key(text, "query")
    
    assert cache_key == expected_key


@pytest.mark.asyncio
async def test_batch_embedding_deduplication():
    """Test that batch embedding properly deduplicates identical texts"""
    
    fake_embedding1 = [0.1, 0.2, 0.3]
    fake_embedding2 = [0.4, 0.5, 0.6]
    
    # Mock OpenAI client response for batch
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=fake_embedding1),
        MagicMock(embedding=fake_embedding2)
    ]
    
    # Mock Redis to always return None (cache miss)
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.setex = AsyncMock()
    
    with patch('redis.asyncio.from_url', return_value=mock_redis):
        service = KnowledgeEmbeddingService(cache_embeddings=True)
        service.client.embeddings.create = AsyncMock(return_value=mock_response)
        
        # Test batch with duplicate texts
        texts = ["hello", "world", "hello"]  # "hello" appears twice
        results = await service.embed_texts(texts)
        
        # Should get 3 results
        assert len(results) == 3
        
        # First and third results should be identical (deduplication)
        assert results[0].embedding == results[2].embedding
        assert results[0].text == results[2].text == "hello"
        
        # Second result should be different
        assert results[1].text == "world"
        assert results[1].embedding != results[0].embedding
        
        # API should only be called once for unique texts
        service.client.embeddings.create.assert_called_once()
        
        # Should be called with only unique texts
        call_args = service.client.embeddings.create.call_args
        assert set(call_args[1]['input']) == {"hello", "world"}


@pytest.mark.asyncio
async def test_cache_stats():
    """Test cache statistics functionality"""
    
    mock_redis = AsyncMock()
    mock_redis.keys.return_value = ["key1", "key2", "key3"]
    
    with patch('redis.asyncio.from_url', return_value=mock_redis):
        service = KnowledgeEmbeddingService(cache_embeddings=True)
        
        # Add some items to local cache
        service._local_cache["test1"] = EmbeddingResult(
            text="test1", 
            embedding=[0.1, 0.2], 
            model="test", 
            timestamp=None,
            metadata={}
        )
        service._local_cache["test2"] = EmbeddingResult(
            text="test2", 
            embedding=[0.3, 0.4], 
            model="test", 
            timestamp=None,
            metadata={}
        )
        
        stats = await service.get_cache_stats()
        
        assert stats["cache_enabled"] is True
        assert stats["l1_cache_size"] == 2
        assert stats["l2_cache_keys"] == 3
        assert stats["total_cached_items"] == 5
        assert "l1_memory_mb" in stats