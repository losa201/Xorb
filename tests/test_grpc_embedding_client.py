"""
Test suite for gRPC embedding client
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from xorb_common.clients.embedding_grpc_client import EmbeddingGRPCClient, EmbeddingResult, EmbeddingMetrics


@pytest.mark.asyncio
async def test_embedding_client_connection():
    """Test gRPC client connection handling"""
    
    client = EmbeddingGRPCClient(server_url="localhost:50051")
    
    # Test connection establishment
    with patch('grpc.aio.insecure_channel') as mock_channel:
        mock_channel_instance = AsyncMock()
        mock_channel_instance.channel_ready = AsyncMock()
        mock_channel.return_value = mock_channel_instance
        
        await client.connect()
        
        mock_channel.assert_called_once()
        assert client._channel is not None
        
        # Test close
        await client.close()
        mock_channel_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_embed_texts_functionality():
    """Test the core embedding functionality"""
    
    client = EmbeddingGRPCClient()
    
    # Mock the connection
    client._channel = AsyncMock()
    client._stub = AsyncMock()
    
    # Test with sample texts
    texts = ["hello world", "test embedding", "grpc service"]
    
    results, metrics = await client.embed_texts(
        texts=texts,
        model="nvidia/embed-qa-4",
        input_type="query",
        use_cache=True
    )
    
    # Verify results structure
    assert len(results) == len(texts)
    assert isinstance(metrics, EmbeddingMetrics)
    
    for i, result in enumerate(results):
        assert isinstance(result, EmbeddingResult)
        assert result.text == texts[i]
        assert len(result.embedding) == 1024  # Mock embedding dimension
        assert result.model == "nvidia/embed-qa-4"
        assert result.cache_key is not None
        assert isinstance(result.from_cache, bool)


@pytest.mark.asyncio
async def test_embed_single_text():
    """Test single text embedding"""
    
    client = EmbeddingGRPCClient()
    client._channel = AsyncMock()
    
    result = await client.embed_single_text(
        text="single test text",
        model="nvidia/embed-qa-4",
        input_type="query"
    )
    
    assert isinstance(result, EmbeddingResult)
    assert result.text == "single test text"
    assert len(result.embedding) == 1024


@pytest.mark.asyncio
async def test_compute_similarity():
    """Test similarity computation"""
    
    client = EmbeddingGRPCClient()
    client._channel = AsyncMock()
    
    # Test vectors
    embedding1 = [1.0, 0.0, 0.0]
    embedding2 = [0.0, 1.0, 0.0]
    
    # Test cosine similarity
    similarity = await client.compute_similarity(
        embedding1=embedding1,
        embedding2=embedding2,
        metric="cosine"
    )
    
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0
    
    # Test with identical vectors
    similarity_identical = await client.compute_similarity(
        embedding1=embedding1,
        embedding2=embedding1,
        metric="cosine"
    )
    
    assert abs(similarity_identical - 1.0) < 1e-10  # Should be very close to 1.0


@pytest.mark.asyncio
async def test_health_check():
    """Test service health check"""
    
    client = EmbeddingGRPCClient()
    client._channel = AsyncMock()
    
    health_info = await client.get_health()
    
    assert isinstance(health_info, dict)
    assert "status" in health_info
    assert "uptime_seconds" in health_info
    assert "cache_stats" in health_info
    
    cache_stats = health_info["cache_stats"]
    assert "l1_cache_size" in cache_stats
    assert "l2_cache_keys" in cache_stats
    assert "cache_hit_rate_1h" in cache_stats


@pytest.mark.asyncio
async def test_cache_clear():
    """Test cache clearing functionality"""
    
    client = EmbeddingGRPCClient()
    client._channel = AsyncMock()
    
    # Clear all caches
    result = await client.clear_cache()
    
    assert isinstance(result, dict)
    assert "l1_keys_cleared" in result
    assert "l2_keys_cleared" in result
    assert "status" in result
    assert result["status"] == "success"
    
    # Clear specific model
    result_model = await client.clear_cache(model="nvidia/embed-qa-4")
    assert result_model["status"] == "success"
    
    # Clear L1 only
    result_l1 = await client.clear_cache(l1_only=True)
    assert result_l1["l2_keys_cleared"] == 0


@pytest.mark.asyncio
async def test_retry_mechanism():
    """Test retry mechanism with backoff"""
    
    client = EmbeddingGRPCClient(max_retry_attempts=3)
    
    # Mock operation that fails twice then succeeds
    mock_operation = AsyncMock(side_effect=[
        Exception("Connection failed"),
        Exception("Timeout"),
        "Success"
    ])
    
    result = await client._retry_with_backoff(mock_operation)
    
    assert result == "Success"
    assert mock_operation.call_count == 3


@pytest.mark.asyncio
async def test_context_manager():
    """Test async context manager functionality"""
    
    with patch('grpc.aio.insecure_channel') as mock_channel:
        mock_channel_instance = AsyncMock()
        mock_channel_instance.channel_ready = AsyncMock()
        mock_channel.return_value = mock_channel_instance
        
        async with EmbeddingGRPCClient() as client:
            assert client._channel is not None
            
            # Use the client
            results, metrics = await client.embed_texts(["test"])
            assert len(results) == 1
        
        # Verify connection was closed
        mock_channel_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_empty_texts_handling():
    """Test handling of empty text lists"""
    
    client = EmbeddingGRPCClient()
    
    results, metrics = await client.embed_texts(texts=[])
    
    assert results == []
    assert metrics.total_tokens == 0
    assert metrics.cache_hits == 0
    assert metrics.cache_misses == 0


@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing with large text lists"""
    
    client = EmbeddingGRPCClient()
    client._channel = AsyncMock()
    
    # Create a large list of texts
    texts = [f"test text {i}" for i in range(150)]  # More than default batch size
    
    results, metrics = await client.embed_texts(
        texts=texts,
        batch_size=50  # Will require 3 batches
    )
    
    assert len(results) == 150
    assert metrics.total_tokens > 0
    
    # Verify all texts are processed
    result_texts = {result.text for result in results}
    expected_texts = set(texts)
    assert result_texts == expected_texts


@pytest.mark.asyncio
async def test_global_client_management():
    """Test global client instance management"""
    
    from xorb_common.clients.embedding_grpc_client import get_embedding_client, close_embedding_client
    
    with patch('grpc.aio.insecure_channel') as mock_channel:
        mock_channel_instance = AsyncMock()
        mock_channel_instance.channel_ready = AsyncMock()
        mock_channel.return_value = mock_channel_instance
        
        # Get global client
        client1 = await get_embedding_client()
        client2 = await get_embedding_client()
        
        # Should be the same instance
        assert client1 is client2
        
        # Close global client
        await close_embedding_client()
        
        mock_channel_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_performance_metrics():
    """Test that performance metrics are properly calculated"""
    
    client = EmbeddingGRPCClient()
    client._channel = AsyncMock()
    
    texts = ["performance", "test", "metrics"]
    
    results, metrics = await client.embed_texts(texts)
    
    # Verify metrics structure
    assert metrics.request_duration_ms > 0
    assert metrics.total_tokens == sum(len(text.split()) for text in texts)
    assert metrics.cache_hits + metrics.cache_misses == len(texts)
    assert 0.0 <= metrics.cache_hit_rate <= 1.0
    assert metrics.cached_tokens + metrics.api_tokens == metrics.total_tokens
    assert metrics.cost_usd >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])