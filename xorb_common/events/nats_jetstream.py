"""
NATS JetStream integration for event-driven architecture
Provides high-performance messaging with persistence and exactly-once delivery
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

import nats
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig, ConsumerConfig, DeliverPolicy, AckPolicy
from nats.js.client import JetStreamContext
from prometheus_client import Counter, Histogram, Gauge
import structlog

from ..logging import get_logger

log = get_logger(__name__)

# Prometheus metrics for event system
events_published_total = Counter(
    'xorb_events_published_total',
    'Total events published',
    ['subject', 'event_type', 'status']
)

events_consumed_total = Counter(
    'xorb_events_consumed_total',
    'Total events consumed',
    ['subject', 'event_type', 'status']
)

event_processing_duration = Histogram(
    'xorb_event_processing_duration_seconds',
    'Time spent processing events',
    ['subject', 'event_type']
)

active_consumers = Gauge(
    'xorb_jetstream_active_consumers',
    'Number of active JetStream consumers'
)


class EventType(Enum):
    """Standard event types for the Xorb system"""
    ATOM_CREATED = "atom.created"
    ATOM_UPDATED = "atom.updated"
    ATOM_DELETED = "atom.deleted"
    SCAN_STARTED = "scan.started"
    SCAN_COMPLETED = "scan.completed"
    SCAN_FAILED = "scan.failed"
    FINDING_DISCOVERED = "finding.discovered"
    FINDING_TRIAGED = "finding.triaged"
    SIMILARITY_THRESHOLD = "similarity.threshold"
    EMBEDDING_CACHED = "embedding.cached"
    CAMPAIGN_CREATED = "campaign.created"
    CAMPAIGN_COMPLETED = "campaign.completed"
    AGENT_SELECTED = "agent.selected"
    AGENT_EXECUTED = "agent.executed"
    COST_THRESHOLD = "cost.threshold"
    PERFORMANCE_ALERT = "performance.alert"


@dataclass
class CloudEvent:
    """CloudEvents specification compliant event"""
    specversion: str = "1.0"
    type: str = ""
    source: str = "xorb-system"
    id: str = ""
    time: str = ""
    datacontenttype: str = "application/json"
    data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"xorb-{int(time.time() * 1000)}"
        if not self.time:
            self.time = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CloudEvent':
        """Create CloudEvent from dictionary"""
        return cls(**data)


class XorbEventBus:
    """High-performance event bus using NATS JetStream"""
    
    def __init__(
        self,
        nats_servers: List[str] = None,
        stream_name: str = "XORB_EVENTS",
        max_age_seconds: int = 7 * 24 * 3600,  # 7 days
        max_bytes: int = 10 * 1024 * 1024 * 1024,  # 10GB
        replicas: int = 1
    ):
        self.nats_servers = nats_servers or ["nats://nats:4222"]
        self.stream_name = stream_name
        self.max_age_seconds = max_age_seconds
        self.max_bytes = max_bytes
        self.replicas = replicas
        
        self.nc: Optional[NATS] = None
        self.js: Optional[JetStreamContext] = None
        self._consumers: Dict[str, Any] = {}
        self._running = False
        
        log.info("Event bus initialized",
                stream_name=stream_name,
                nats_servers=self.nats_servers)
    
    async def connect(self) -> None:
        """Connect to NATS and setup JetStream"""
        try:
            # Connect to NATS
            self.nc = await nats.connect(
                servers=self.nats_servers,
                max_reconnect_attempts=10,
                reconnect_time_wait=2,
                max_pending=10000,
                max_outstanding=1000
            )
            
            # Get JetStream context
            self.js = self.nc.jetstream()
            
            # Create or update stream
            await self._setup_stream()
            
            self._running = True
            
            log.info("Connected to NATS JetStream",
                    servers=self.nc.connected_url,
                    stream=self.stream_name)
            
        except Exception as e:
            log.error("Failed to connect to NATS", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from NATS"""
        self._running = False
        
        # Stop all consumers
        for consumer_name, consumer in self._consumers.items():
            try:
                await consumer.stop()
                log.info("Stopped consumer", consumer=consumer_name)
            except Exception as e:
                log.warning("Error stopping consumer", consumer=consumer_name, error=str(e))
        
        self._consumers.clear()
        
        # Close NATS connection
        if self.nc:
            await self.nc.close()
            log.info("Disconnected from NATS")
    
    async def _setup_stream(self) -> None:
        """Setup JetStream stream for events"""
        
        stream_config = StreamConfig(
            name=self.stream_name,
            subjects=[f"{self.stream_name.lower()}.*"],
            max_age=self.max_age_seconds,
            max_bytes=self.max_bytes,
            replicas=self.replicas,
            storage="file",  # Persistent storage
            retention="limits",  # Keep based on limits
            discard="old"  # Discard old messages when limits reached
        )
        
        try:
            # Try to get existing stream
            await self.js.stream_info(self.stream_name)
            log.info("JetStream stream exists", stream=self.stream_name)
            
            # Update stream if needed
            await self.js.update_stream(stream_config)
            log.info("Updated JetStream stream", stream=self.stream_name)
            
        except Exception:
            # Stream doesn't exist, create it
            await self.js.add_stream(stream_config)
            log.info("Created JetStream stream", stream=self.stream_name)
    
    async def publish(
        self,
        event_type: Union[EventType, str],
        data: Dict[str, Any],
        source: str = "xorb-system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Publish an event to the stream"""
        
        if not self._running:
            await self.connect()
        
        # Create CloudEvent
        event = CloudEvent(
            type=event_type.value if isinstance(event_type, EventType) else event_type,
            source=source,
            data=data
        )
        
        # Add metadata if provided
        if metadata:
            event.data = event.data or {}
            event.data["_metadata"] = metadata
        
        # Determine subject
        subject = f"{self.stream_name.lower()}.{event.type}"
        
        try:
            # Publish to JetStream
            ack = await self.js.publish(
                subject=subject,
                payload=json.dumps(event.to_dict()).encode('utf-8')
            )
            
            events_published_total.labels(
                subject=subject,
                event_type=event.type,
                status="success"
            ).inc()
            
            log.debug("Event published",
                     event_id=event.id,
                     event_type=event.type,
                     subject=subject,
                     sequence=ack.seq)
            
            return event.id
            
        except Exception as e:
            events_published_total.labels(
                subject=subject,
                event_type=event.type,
                status="error"
            ).inc()
            
            log.error("Failed to publish event",
                     event_type=event.type,
                     error=str(e))
            raise
    
    async def subscribe(
        self,
        event_pattern: str,
        handler: Callable[[CloudEvent], None],
        consumer_name: str,
        durable: bool = True,
        ack_policy: AckPolicy = AckPolicy.EXPLICIT,
        max_deliver: int = 3,
        ack_wait_seconds: int = 30
    ) -> None:
        """Subscribe to events matching pattern"""
        
        if not self._running:
            await self.connect()
        
        subject = f"{self.stream_name.lower()}.{event_pattern}"
        
        # Create consumer configuration
        consumer_config = ConsumerConfig(
            name=consumer_name,
            durable_name=consumer_name if durable else None,
            deliver_policy=DeliverPolicy.NEW,
            ack_policy=ack_policy,
            max_deliver=max_deliver,
            ack_wait=ack_wait_seconds,
            filter_subject=subject
        )
        
        async def message_handler(msg):
            """Handle incoming messages"""
            start_time = time.time()
            
            try:
                # Parse CloudEvent
                event_data = json.loads(msg.data.decode('utf-8'))
                event = CloudEvent.from_dict(event_data)
                
                log.debug("Processing event",
                         event_id=event.id,
                         event_type=event.type,
                         consumer=consumer_name)
                
                # Call user handler
                await handler(event)
                
                # Acknowledge message
                await msg.ack()
                
                duration = time.time() - start_time
                event_processing_duration.labels(
                    subject=subject,
                    event_type=event.type
                ).observe(duration)
                
                events_consumed_total.labels(
                    subject=subject,
                    event_type=event.type,
                    status="success"
                ).inc()
                
                log.debug("Event processed successfully",
                         event_id=event.id,
                         duration=duration)
                
            except Exception as e:
                duration = time.time() - start_time
                
                events_consumed_total.labels(
                    subject=subject,
                    event_type="unknown",
                    status="error"
                ).inc()
                
                log.error("Error processing event",
                         consumer=consumer_name,
                         error=str(e),
                         duration=duration)
                
                # Negative acknowledge to retry
                await msg.nak()
        
        try:
            # Create pull subscription
            psub = await self.js.pull_subscribe(
                subject=subject,
                consumer=consumer_config.name,
                config=consumer_config
            )
            
            # Store consumer reference
            self._consumers[consumer_name] = psub
            active_consumers.inc()
            
            log.info("Created event consumer",
                    consumer=consumer_name,
                    subject=subject,
                    durable=durable)
            
            # Start message processing loop
            asyncio.create_task(self._consumer_loop(psub, message_handler, consumer_name))
            
        except Exception as e:
            log.error("Failed to create consumer",
                     consumer=consumer_name,
                     error=str(e))
            raise
    
    async def _consumer_loop(
        self,
        subscription,
        handler: Callable,
        consumer_name: str
    ) -> None:
        """Message processing loop for a consumer"""
        
        while self._running:
            try:
                # Fetch messages in batches
                msgs = await subscription.fetch(batch=10, timeout=1.0)
                
                # Process messages concurrently
                if msgs:
                    await asyncio.gather(
                        *[handler(msg) for msg in msgs],
                        return_exceptions=True
                    )
                
            except nats.errors.TimeoutError:
                # No messages available, continue
                continue
                
            except Exception as e:
                log.error("Error in consumer loop",
                         consumer=consumer_name,
                         error=str(e))
                await asyncio.sleep(1)  # Brief pause before retry
        
        active_consumers.dec()
        log.info("Consumer loop stopped", consumer=consumer_name)
    
    async def get_stream_info(self) -> Dict[str, Any]:
        """Get stream information and statistics"""
        
        if not self.js:
            raise RuntimeError("Not connected to JetStream")
        
        try:
            info = await self.js.stream_info(self.stream_name)
            
            return {
                "name": info.config.name,
                "subjects": info.config.subjects,
                "messages": info.state.messages,
                "bytes": info.state.bytes,
                "first_seq": info.state.first_seq,
                "last_seq": info.state.last_seq,
                "num_subjects": info.state.num_subjects,
                "consumers": info.state.consumer_count
            }
            
        except Exception as e:
            log.error("Failed to get stream info", error=str(e))
            raise
    
    async def purge_stream(self, subject_filter: Optional[str] = None) -> int:
        """Purge messages from the stream"""
        
        if not self.js:
            raise RuntimeError("Not connected to JetStream")
        
        try:
            if subject_filter:
                result = await self.js.purge_stream(
                    self.stream_name,
                    subject=f"{self.stream_name.lower()}.{subject_filter}"
                )
            else:
                result = await self.js.purge_stream(self.stream_name)
            
            log.info("Stream purged",
                    stream=self.stream_name,
                    messages_purged=result.purged,
                    subject_filter=subject_filter)
            
            return result.purged
            
        except Exception as e:
            log.error("Failed to purge stream", error=str(e))
            raise


# Global event bus instance
_global_event_bus: Optional[XorbEventBus] = None


async def get_event_bus() -> XorbEventBus:
    """Get or create global event bus instance"""
    global _global_event_bus
    
    if _global_event_bus is None:
        _global_event_bus = XorbEventBus()
        await _global_event_bus.connect()
    
    return _global_event_bus


async def close_event_bus():
    """Close global event bus"""
    global _global_event_bus
    
    if _global_event_bus:
        await _global_event_bus.disconnect()
        _global_event_bus = None


# Convenience functions for common event types
async def publish_atom_created(atom_id: int, atom_type: str, data: Dict[str, Any]):
    """Publish atom created event"""
    bus = await get_event_bus()
    await bus.publish(
        EventType.ATOM_CREATED,
        {"atom_id": atom_id, "atom_type": atom_type, **data}
    )


async def publish_scan_completed(target: str, findings_count: int, severity: str, data: Dict[str, Any] = None):
    """Publish scan completed event"""
    bus = await get_event_bus()
    await bus.publish(
        EventType.SCAN_COMPLETED,
        {
            "target": target,
            "findings_count": findings_count,
            "severity": severity,
            **(data or {})
        }
    )


async def publish_similarity_threshold(source_id: int, nearest_id: int, score: float):
    """Publish similarity threshold exceeded event"""
    bus = await get_event_bus()
    await bus.publish(
        EventType.SIMILARITY_THRESHOLD,
        {
            "source_id": source_id,
            "nearest_id": nearest_id,
            "similarity_score": score
        }
    )


async def publish_embedding_cached(text_hash: str, model: str, cache_type: str):
    """Publish embedding cached event"""
    bus = await get_event_bus()
    await bus.publish(
        EventType.EMBEDDING_CACHED,
        {
            "text_hash": text_hash,
            "model": model,
            "cache_type": cache_type
        }
    )


async def publish_cost_threshold(service: str, cost_usd: float, threshold_usd: float):
    """Publish cost threshold exceeded event"""
    bus = await get_event_bus()
    await bus.publish(
        EventType.COST_THRESHOLD,
        {
            "service": service,
            "cost_usd": cost_usd,
            "threshold_usd": threshold_usd
        }
    )