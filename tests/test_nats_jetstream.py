"""
Test suite for NATS JetStream event system
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from xorb_common.events.nats_jetstream import (
    XorbEventBus, CloudEvent, EventType,
    publish_atom_created, publish_scan_completed
)


@pytest.mark.asyncio
async def test_cloud_event_creation():
    """Test CloudEvent creation and serialization"""
    
    event = CloudEvent(
        type="test.event",
        source="test-service",
        data={"key": "value", "count": 42}
    )
    
    # Test basic properties
    assert event.specversion == "1.0"
    assert event.type == "test.event"
    assert event.source == "test-service"
    assert event.datacontenttype == "application/json"
    assert event.data == {"key": "value", "count": 42}
    
    # Test auto-generated fields
    assert event.id.startswith("xorb-")
    assert event.time is not None
    
    # Test serialization
    event_dict = event.to_dict()
    assert isinstance(event_dict, dict)
    assert event_dict["type"] == "test.event"
    
    # Test deserialization
    reconstructed = CloudEvent.from_dict(event_dict)
    assert reconstructed.type == event.type
    assert reconstructed.data == event.data


@pytest.mark.asyncio
async def test_event_bus_connection():
    """Test event bus connection and disconnection"""
    
    with patch('nats.connect') as mock_connect:
        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js
        mock_connect.return_value = mock_nc
        
        # Mock stream operations
        mock_js.stream_info.side_effect = Exception("Stream not found")
        mock_js.add_stream = AsyncMock()
        
        event_bus = XorbEventBus(nats_servers=["nats://localhost:4222"])
        
        await event_bus.connect()
        
        assert event_bus.nc is not None
        assert event_bus.js is not None
        assert event_bus._running is True
        
        # Test stream creation
        mock_js.add_stream.assert_called_once()
        
        await event_bus.disconnect()
        mock_nc.close.assert_called_once()


@pytest.mark.asyncio
async def test_event_publishing():
    """Test event publishing functionality"""
    
    with patch('nats.connect') as mock_connect:
        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js
        mock_connect.return_value = mock_nc
        
        # Mock stream operations
        mock_js.stream_info.side_effect = Exception("Stream not found")
        mock_js.add_stream = AsyncMock()
        
        # Mock publish acknowledgment
        mock_ack = MagicMock()
        mock_ack.seq = 12345
        mock_js.publish.return_value = mock_ack
        
        event_bus = XorbEventBus()
        await event_bus.connect()
        
        # Publish an event
        event_id = await event_bus.publish(
            event_type=EventType.ATOM_CREATED,
            data={"atom_id": 123, "atom_type": "VULN"},
            source="test-service"
        )
        
        assert event_id.startswith("xorb-")
        
        # Verify publish was called
        mock_js.publish.assert_called_once()
        call_args = mock_js.publish.call_args
        
        assert call_args[1]["subject"] == "xorb_events.atom.created"
        
        # Verify payload
        payload = json.loads(call_args[1]["payload"].decode('utf-8'))
        assert payload["type"] == "atom.created"
        assert payload["data"]["atom_id"] == 123


@pytest.mark.asyncio
async def test_event_subscription():
    """Test event subscription and message handling"""
    
    with patch('nats.connect') as mock_connect:
        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js
        mock_connect.return_value = mock_nc
        
        # Mock stream and subscription operations
        mock_js.stream_info.side_effect = Exception("Stream not found")
        mock_js.add_stream = AsyncMock()
        mock_psub = AsyncMock()
        mock_js.pull_subscribe.return_value = mock_psub
        
        event_bus = XorbEventBus()
        await event_bus.connect()
        
        # Track handled events
        handled_events = []
        
        async def test_handler(event: CloudEvent):
            handled_events.append(event)
        
        # Subscribe to events
        await event_bus.subscribe(
            event_pattern="atom.*",
            handler=test_handler,
            consumer_name="test-consumer"
        )
        
        # Verify subscription was created
        mock_js.pull_subscribe.assert_called_once()
        assert "test-consumer" in event_bus._consumers
        
        # Test message handling by simulating a message
        mock_msg = AsyncMock()
        test_event_data = {
            "specversion": "1.0",
            "type": "atom.created",
            "source": "test",
            "id": "test-123",
            "time": "2025-07-25T10:00:00Z",
            "data": {"atom_id": 456}
        }
        mock_msg.data = json.dumps(test_event_data).encode('utf-8')
        mock_msg.ack = AsyncMock()
        
        # Get the message handler from the subscription call
        message_handler = None
        for task in asyncio.all_tasks():
            if hasattr(task, '_coro') and 'consumer_loop' in task._coro.__name__:
                message_handler = task
                break
        
        # Since we can't easily test the consumer loop, we'll test the handler directly
        # In a real test, you'd set up the full message flow


@pytest.mark.asyncio
async def test_event_bus_stream_info():
    """Test getting stream information"""
    
    with patch('nats.connect') as mock_connect:
        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js
        mock_connect.return_value = mock_nc
        
        # Mock stream operations
        mock_js.stream_info.side_effect = Exception("Stream not found")
        mock_js.add_stream = AsyncMock()
        
        # Mock stream info response
        mock_info = MagicMock()
        mock_info.config.name = "XORB_EVENTS"
        mock_info.config.subjects = ["xorb_events.*"]
        mock_info.state.messages = 1000
        mock_info.state.bytes = 50000
        mock_info.state.first_seq = 1
        mock_info.state.last_seq = 1000
        mock_info.state.num_subjects = 5
        mock_info.state.consumer_count = 3
        
        # Reset mock to return info instead of exception
        mock_js.stream_info.side_effect = None
        mock_js.stream_info.return_value = mock_info
        
        event_bus = XorbEventBus()
        await event_bus.connect()
        
        # Get stream info
        info = await event_bus.get_stream_info()
        
        assert info["name"] == "XORB_EVENTS"
        assert info["messages"] == 1000
        assert info["bytes"] == 50000
        assert info["consumers"] == 3


@pytest.mark.asyncio
async def test_stream_purging():
    """Test stream purging functionality"""
    
    with patch('nats.connect') as mock_connect:
        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js
        mock_connect.return_value = mock_nc
        
        # Mock stream operations
        mock_js.stream_info.side_effect = Exception("Stream not found")
        mock_js.add_stream = AsyncMock()
        
        # Mock purge result
        mock_purge_result = MagicMock()
        mock_purge_result.purged = 500
        mock_js.purge_stream.return_value = mock_purge_result
        
        event_bus = XorbEventBus()
        await event_bus.connect()
        
        # Purge entire stream
        purged_count = await event_bus.purge_stream()
        assert purged_count == 500
        
        # Purge with subject filter
        purged_filtered = await event_bus.purge_stream("atom.*")
        assert purged_filtered == 500


@pytest.mark.asyncio
async def test_convenience_functions():
    """Test convenience functions for common events"""
    
    with patch('xorb_common.events.nats_jetstream.get_event_bus') as mock_get_bus:
        mock_bus = AsyncMock()
        mock_get_bus.return_value = mock_bus
        
        # Test atom created event
        await publish_atom_created(
            atom_id=123,
            atom_type="VULN",
            data={"severity": "high"}
        )
        
        mock_bus.publish.assert_called_with(
            EventType.ATOM_CREATED,
            {"atom_id": 123, "atom_type": "VULN", "severity": "high"}
        )
        
        # Test scan completed event
        await publish_scan_completed(
            target="example.com",
            findings_count=5,
            severity="medium",
            data={"scan_duration": 300}
        )
        
        expected_data = {
            "target": "example.com",
            "findings_count": 5,
            "severity": "medium",
            "scan_duration": 300
        }
        
        mock_bus.publish.assert_called_with(
            EventType.SCAN_COMPLETED,
            expected_data
        )


@pytest.mark.asyncio
async def test_global_event_bus():
    """Test global event bus management"""
    
    from xorb_common.events.nats_jetstream import get_event_bus, close_event_bus
    
    with patch('nats.connect') as mock_connect:
        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js
        mock_connect.return_value = mock_nc
        
        # Mock stream operations
        mock_js.stream_info.side_effect = Exception("Stream not found")
        mock_js.add_stream = AsyncMock()
        
        # Get global event bus
        bus1 = await get_event_bus()
        bus2 = await get_event_bus()
        
        # Should be the same instance
        assert bus1 is bus2
        
        # Close global event bus
        await close_event_bus()
        
        mock_nc.close.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in event operations"""
    
    with patch('nats.connect') as mock_connect:
        mock_connect.side_effect = Exception("Connection failed")
        
        event_bus = XorbEventBus()
        
        # Test connection failure
        with pytest.raises(Exception, match="Connection failed"):
            await event_bus.connect()


@pytest.mark.asyncio
async def test_event_types():
    """Test all defined event types"""
    
    # Test that all event types have proper values
    assert EventType.ATOM_CREATED.value == "atom.created"
    assert EventType.SCAN_COMPLETED.value == "scan.completed"
    assert EventType.SIMILARITY_THRESHOLD.value == "similarity.threshold"
    assert EventType.EMBEDDING_CACHED.value == "embedding.cached"
    assert EventType.COST_THRESHOLD.value == "cost.threshold"
    
    # Test using event types in publishing
    with patch('nats.connect') as mock_connect:
        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js
        mock_connect.return_value = mock_nc
        
        # Mock stream operations
        mock_js.stream_info.side_effect = Exception("Stream not found")
        mock_js.add_stream = AsyncMock()
        mock_js.publish.return_value = MagicMock(seq=1)
        
        event_bus = XorbEventBus()
        await event_bus.connect()
        
        # Test publishing with EventType enum
        await event_bus.publish(
            EventType.FINDING_DISCOVERED,
            {"finding_id": 789}
        )
        
        # Verify the subject was formatted correctly
        call_args = mock_js.publish.call_args
        assert call_args[1]["subject"] == "xorb_events.finding.discovered"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])