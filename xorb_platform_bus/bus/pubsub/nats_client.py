#!/usr/bin/env python3
"""
NATS Client with Replay-Safe Streaming - Phase G4

Provides a production-ready NATS JetStream client with:
- Subject schema validation (v1 immutable)
- Per-tenant isolation and quotas
- Dedicated replay lanes with .replay suffix
- Time-bounded replay with DeliverPolicy=ByStartTime
- Rate-limited replay workers with global limits
- Storage I/O isolation and priority handling
- Comprehensive SLO monitoring and chaos testing support
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Callable
from enum import Enum
from datetime import datetime, timedelta
import uuid

import nats
from nats.errors import TimeoutError, NoRespondersError
from nats.js import JetStreamContext
from nats.js.api import StreamConfig, ConsumerConfig, RetentionPolicy, DiscardPolicy, ReplayPolicy
from nats.js.errors import NotFoundError, BadRequestError

logger = logging.getLogger(__name__)


class Domain(str, Enum):
    """Valid domains for NATS subjects (v1 immutable)."""
    EVIDENCE = "evidence"
    SCAN = "scan"
    COMPLIANCE = "compliance"
    CONTROL = "control"


class Event(str, Enum):
    """Valid events for NATS subjects (v1 immutable)."""
    CREATED = "created"
    UPDATED = "updated"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLAY = "replay"


class StreamClass(str, Enum):
    """Stream classes with different characteristics."""
    LIVE = "live"      # High-priority operational streams
    REPLAY = "replay"  # Lower-priority replay lanes with bounded windows


@dataclass
class SubjectComponents:
    """Components of a NATS subject following v1 schema."""
    tenant: str
    domain: Domain
    service: str
    event: Event

    def validate(self) -> bool:
        """Validate subject components against v1 schema."""
        # Tenant validation
        if not (3 <= len(self.tenant) <= 63):
            return False
        if not self.tenant.replace('-', '').replace('_', '').isalnum():
            return False
        if self.tenant.startswith('-') or self.tenant.endswith('-'):
            return False

        # Service validation
        if not (1 <= len(self.service) <= 32):
            return False
        if not self.service.replace('-', '').replace('_', '').isalnum():
            return False
        if self.service.startswith('-') or self.service.endswith('-'):
            return False

        return True

    def to_subject(self) -> str:
        """Convert components to NATS subject string."""
        if not self.validate():
            raise ValueError(f"Invalid subject components: {self}")
        return f"xorb.{self.tenant}.{self.domain.value}.{self.service}.{self.event.value}"


@dataclass
class ConsumerSettings:
    """Consumer configuration with tuned flow control settings."""
    ack_wait: str = "30s"
    max_ack_pending: int = 1024
    flow_control: bool = True
    idle_heartbeat: str = "5s"
    deliver_policy: str = "last"
    rate_limit_bps: Optional[int] = 1048576  # 1MB/s
    max_deliver: int = 3
    replay_policy: ReplayPolicy = ReplayPolicy.INSTANT
    priority: int = 1  # I/O priority (1=high, 10=low)


@dataclass
class ReplaySettings:
    """Replay-specific configuration with bounded windows."""
    time_window_hours: int = 168  # 7 days bounded window
    global_rate_limit_bps: int = 5242880  # 5MB/s global rate limit
    max_replay_workers: int = 5
    concurrency_cap: int = 10
    storage_isolation: bool = True
    start_time_policy: str = "ByStartTime"
    
    @classmethod
    def from_env(cls) -> "ReplaySettings":
        """Create replay settings from environment variables."""
        return cls(
            time_window_hours=int(os.getenv("NATS_REPLAY_WINDOW_HOURS", "168")),
            global_rate_limit_bps=int(os.getenv("NATS_REPLAY_RATE_LIMIT", "5242880")),
            max_replay_workers=int(os.getenv("NATS_REPLAY_MAX_WORKERS", "5")),
            concurrency_cap=int(os.getenv("NATS_REPLAY_CONCURRENCY_CAP", "10")),
            storage_isolation=os.getenv("NATS_REPLAY_STORAGE_ISOLATION", "true").lower() == "true"
        )

    @classmethod
    def from_env(cls) -> "ConsumerSettings":
        """Create consumer settings from environment variables."""
        return cls(
            ack_wait=os.getenv("NATS_ACK_WAIT", "30s"),
            max_ack_pending=int(os.getenv("NATS_MAX_ACK_PENDING", "1024")),
            flow_control=os.getenv("NATS_FLOW_CONTROL", "true").lower() == "true",
            idle_heartbeat=os.getenv("NATS_IDLE_HEARTBEAT", "5s"),
            deliver_policy=os.getenv("NATS_DELIVER_POLICY", "last"),
            rate_limit_bps=int(os.getenv("NATS_RATE_LIMIT_BPS", "1048576")),
            max_deliver=int(os.getenv("NATS_MAX_DELIVER", "3"))
        )


class SubjectBuilder:
    """Builder for constructing and validating NATS subjects."""

    def __init__(self, tenant: str):
        self.tenant = tenant

    def build(self, domain: Domain, service: str, event: Event, replay: bool = False) -> str:
        """Build a validated NATS subject with optional replay suffix."""
        if replay and event != Event.REPLAY:
            # Force replay event for replay subjects
            event = Event.REPLAY
            
        components = SubjectComponents(
            tenant=self.tenant,
            domain=domain,
            service=service,
            event=event
        )
        return components.to_subject()
    
    def build_replay_subject(self, domain: Domain, service: str) -> str:
        """Build a replay-specific subject with .replay suffix."""
        return self.build(domain, service, Event.REPLAY, replay=True)

    def parse(self, subject: str) -> SubjectComponents:
        """Parse a NATS subject into components."""
        parts = subject.split('.')
        if len(parts) != 5 or parts[0] != 'xorb':
            raise ValueError(f"Invalid subject format: {subject}")

        try:
            return SubjectComponents(
                tenant=parts[1],
                domain=Domain(parts[2]),
                service=parts[3],
                event=Event(parts[4])
            )
        except ValueError as e:
            raise ValueError(f"Invalid subject components in {subject}: {e}")


class NATSClient:
    """Production NATS JetStream client with tenant isolation and tuning."""

    def __init__(
        self,
        servers: List[str],
        tenant_id: str,
        credentials_file: Optional[str] = None,
        consumer_settings: Optional[ConsumerSettings] = None,
        replay_settings: Optional[ReplaySettings] = None
    ):
        self.servers = servers
        self.tenant_id = tenant_id
        self.credentials_file = credentials_file
        self.consumer_settings = consumer_settings or ConsumerSettings.from_env()
        self.replay_settings = replay_settings or ReplaySettings.from_env()

        self._nc: Optional[nats.NATS] = None
        self._js: Optional[JetStreamContext] = None
        self.subject_builder = SubjectBuilder(tenant_id)

        # Connection state
        self._connected = False
        self._connecting = False

        # Metrics
        self._messages_published = 0
        self._messages_consumed = 0
        self._connection_retries = 0
        self._replay_messages_consumed = 0
        self._live_messages_consumed = 0
        
        # Rate limiting for replay workers
        self._replay_workers_count = 0
        self._replay_rate_limiter_tokens = self.replay_settings.global_rate_limit_bps
        self._last_rate_limit_reset = time.time()

    async def connect(self, **kwargs) -> None:
        """Connect to NATS with retry logic."""
        if self._connected or self._connecting:
            return

        self._connecting = True
        max_retries = kwargs.get('max_retries', 5)
        retry_delay = kwargs.get('retry_delay', 1.0)

        for attempt in range(max_retries):
            try:
                connect_kwargs = {
                    'servers': self.servers,
                    'name': f"xorb-{self.tenant_id}",
                    'max_reconnect_attempts': 10,
                    'reconnect_time_wait': 2,
                    'error_cb': self._error_callback,
                    'closed_cb': self._closed_callback,
                    'reconnected_cb': self._reconnected_callback,
                    **kwargs
                }

                if self.credentials_file:
                    connect_kwargs['user_credentials'] = self.credentials_file

                self._nc = await nats.connect(**connect_kwargs)
                self._js = self._nc.jetstream()
                self._connected = True
                self._connecting = False

                logger.info(f"Connected to NATS for tenant {self.tenant_id}")
                return

            except Exception as e:
                self._connection_retries += 1
                logger.warning(f"NATS connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self._connecting = False
                    raise ConnectionError(f"Failed to connect to NATS after {max_retries} attempts")

    async def disconnect(self) -> None:
        """Disconnect from NATS."""
        if self._nc and self._connected:
            await self._nc.close()
            self._connected = False
            logger.info(f"Disconnected from NATS for tenant {self.tenant_id}")

    @asynccontextmanager
    async def connection(self, **kwargs) -> AsyncGenerator["NATSClient", None]:
        """Context manager for NATS connection."""
        await self.connect(**kwargs)
        try:
            yield self
        finally:
            await self.disconnect()

    async def publish(
        self,
        domain: Domain,
        service: str,
        event: Event,
        data: Union[Dict[Any, Any], str, bytes],
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        replay: bool = False
    ) -> None:
        """Publish a message with subject validation and replay support."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")

        # Apply rate limiting for replay messages
        if replay:
            await self._apply_replay_rate_limit()
            
        subject = self.subject_builder.build(domain, service, event, replay=replay)

        # Prepare message data
        if isinstance(data, dict):
            payload = json.dumps(data).encode()
            content_type = "application/json"
        elif isinstance(data, str):
            payload = data.encode()
            content_type = "text/plain"
        else:
            payload = data
            content_type = "application/octet-stream"

        # Prepare headers with replay metadata
        msg_headers = {
            "Content-Type": content_type,
            "X-Tenant-ID": self.tenant_id,
            "X-Message-ID": str(uuid.uuid4()),
            "X-Timestamp": str(int(time.time() * 1000)),
            "X-Stream-Class": "replay" if replay else "live",
            "X-Priority": "5" if replay else "1"  # Lower priority for replay
        }
        if headers:
            msg_headers.update(headers)

        try:
            ack = await self._js.publish(
                subject,
                payload,
                headers=msg_headers,
                timeout=timeout
            )
            self._messages_published += 1
            if replay:
                logger.debug(f"Published REPLAY message to {subject}, ack: {ack}")
            else:
                logger.debug(f"Published LIVE message to {subject}, ack: {ack}")

        except Exception as e:
            logger.error(f"Failed to publish to {subject}: {e}")
            raise

    async def create_consumer(
        self,
        stream_name: str,
        consumer_name: str,
        subjects: List[str],
        settings: Optional[ConsumerSettings] = None,
        replay_mode: bool = False
    ) -> str:
        """Create a consumer with tuned flow control settings."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")

        settings = settings or self.consumer_settings
        
        # Apply replay-specific settings
        if replay_mode:
            settings = self._get_replay_consumer_settings(settings)

        # Build consumer configuration with replay-specific settings
        deliver_policy = "ByStartTime" if replay_mode else settings.deliver_policy
        
        config = ConsumerConfig(
            name=consumer_name,
            durable_name=consumer_name,
            filter_subjects=subjects,
            deliver_policy=deliver_policy,
            ack_policy="explicit",
            ack_wait=settings.ack_wait,
            max_deliver=settings.max_deliver,
            max_ack_pending=settings.max_ack_pending,
            flow_control=settings.flow_control,
            idle_heartbeat=settings.idle_heartbeat,
            replay_policy=settings.replay_policy
        )
        
        # Set start time for time-bounded replay
        if replay_mode:
            config.opt_start_time = datetime.utcnow() - timedelta(hours=self.replay_settings.time_window_hours)

        if settings.rate_limit_bps:
            config.rate_limit_bps = settings.rate_limit_bps

        try:
            consumer = await self._js.add_consumer(stream_name, config)
            logger.info(f"Created consumer {consumer_name} for stream {stream_name}")
            return consumer_name

        except Exception as e:
            logger.error(f"Failed to create consumer {consumer_name}: {e}")
            raise

    async def subscribe(
        self,
        stream_name: str,
        consumer_name: str,
        callback: Callable[[nats.Msg], Any],
        error_callback: Optional[Callable[[Exception], None]] = None,
        replay_mode: bool = False
    ) -> None:
        """Subscribe to messages with pull-based consumption and replay support."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")
        
        # Check replay worker limits
        if replay_mode and not await self._can_start_replay_worker():
            raise RuntimeError(f"Maximum replay workers ({self.replay_settings.max_replay_workers}) exceeded")

        try:
            # Get consumer info to validate it exists
            consumer_info = await self._js.consumer_info(stream_name, consumer_name)
            logger.debug(f"Consumer {consumer_name} info: {consumer_info}")

            # Create pull subscription
            psub = await self._js.pull_subscribe("", consumer_name, stream=stream_name)

            # Process messages in batches (smaller batches for replay)
            if replay_mode:
                batch_size = min(self.replay_settings.concurrency_cap // 2, 64)
                self._replay_workers_count += 1
            else:
                batch_size = min(self.consumer_settings.max_ack_pending // 4, 256)

            while True:
                try:
                    # Fetch messages in batches
                    messages = await psub.fetch(batch_size, timeout=1.0)

                    for msg in messages:
                        try:
                            # Process message
                            await callback(msg) if asyncio.iscoroutinefunction(callback) else callback(msg)

                            # Acknowledge message with metrics tracking
                            await msg.ack()
                            self._messages_consumed += 1
                            
                            if replay_mode:
                                self._replay_messages_consumed += 1
                            else:
                                self._live_messages_consumed += 1

                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            if error_callback:
                                error_callback(e)
                            # NACK message for retry
                            await msg.nak(delay=5)  # 5 second delay before retry

                except TimeoutError:
                    # No messages available, continue polling
                    continue
                except Exception as e:
                    logger.error(f"Error in message fetch: {e}")
                    if error_callback:
                        error_callback(e)
                    await asyncio.sleep(1)  # Brief pause before retry

        except Exception as e:
            logger.error(f"Failed to subscribe to {stream_name}/{consumer_name}: {e}")
            raise
        finally:
            # Cleanup replay worker count
            if replay_mode and hasattr(self, '_replay_workers_count'):
                self._replay_workers_count = max(0, self._replay_workers_count - 1)

    async def create_stream(
        self,
        domain: Domain,
        stream_class: StreamClass = StreamClass.LIVE,
        max_age: str = "30d",
        max_bytes: int = 536870912,  # 512MB
        max_messages: int = 1000000,
        replicas: int = 1
    ) -> str:
        """Create a stream for the tenant and domain."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")

        stream_name = f"xorb-{self.tenant_id}-{domain.value}-{stream_class.value}"
        subjects = [f"xorb.{self.tenant_id}.{domain.value}.>"]

        # Configure stream based on class
        if stream_class == StreamClass.REPLAY:
            # WORM-like characteristics for audit streams
            retention = RetentionPolicy.LIMITS
            discard = DiscardPolicy.OLD
            max_bytes = max_bytes * 2  # Larger storage for audit
            max_messages = max_messages * 2
        else:
            # Live operational streams
            retention = RetentionPolicy.LIMITS
            discard = DiscardPolicy.OLD

        config = StreamConfig(
            name=stream_name,
            description=f"{stream_class.value.title()} stream for tenant {self.tenant_id} domain {domain.value}",
            subjects=subjects,
            max_age=max_age,
            max_bytes=max_bytes,
            max_messages=max_messages,
            storage="file",  # Persistent storage
            replicas=replicas,
            retention=retention,
            discard=discard,
            duplicate_window=120,  # 2 minutes deduplication
            compression="s2"  # Enable compression
        )

        try:
            stream = await self._js.add_stream(config)
            logger.info(f"Created stream {stream_name}")
            return stream_name

        except Exception as e:
            logger.error(f"Failed to create stream {stream_name}: {e}")
            raise

    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about a stream."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")

        try:
            info = await self._js.stream_info(stream_name)
            return {
                "name": info.config.name,
                "subjects": info.config.subjects,
                "messages": info.state.messages,
                "bytes": info.state.bytes,
                "first_seq": info.state.first_seq,
                "last_seq": info.state.last_seq,
                "consumers": info.state.consumer_count
            }
        except NotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Failed to get stream info for {stream_name}: {e}")
            raise

    async def get_consumer_info(self, stream_name: str, consumer_name: str) -> Dict[str, Any]:
        """Get information about a consumer."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")

        try:
            info = await self._js.consumer_info(stream_name, consumer_name)
            return {
                "name": info.name,
                "stream_name": info.stream_name,
                "delivered": info.delivered.stream_seq,
                "ack_pending": info.ack_floor.stream_seq,
                "waiting": info.num_waiting,
                "ack_pending_count": info.num_ack_pending
            }
        except NotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Failed to get consumer info for {stream_name}/{consumer_name}: {e}")
            raise

    # Event callbacks
    async def _error_callback(self, e: Exception) -> None:
        """Handle NATS connection errors."""
        logger.error(f"NATS error for tenant {self.tenant_id}: {e}")

    async def _closed_callback(self) -> None:
        """Handle NATS connection closed."""
        logger.warning(f"NATS connection closed for tenant {self.tenant_id}")
        self._connected = False

    async def _reconnected_callback(self) -> None:
        """Handle NATS reconnection."""
        logger.info(f"NATS reconnected for tenant {self.tenant_id}")
        self._connected = True

    # Properties
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._nc and not self._nc.is_closed

    @property
    def metrics(self) -> Dict[str, int]:
        """Get client metrics with replay tracking."""
        return {
            "messages_published": self._messages_published,
            "messages_consumed": self._messages_consumed,
            "live_messages_consumed": self._live_messages_consumed,
            "replay_messages_consumed": self._replay_messages_consumed,
            "connection_retries": self._connection_retries,
            "replay_workers_active": self._replay_workers_count,
            "rate_limiter_tokens": self._replay_rate_limiter_tokens
        }
    
    # Replay-specific helper methods
    def _get_replay_consumer_settings(self, base_settings: ConsumerSettings) -> ConsumerSettings:
        """Get replay-specific consumer settings with lower priority."""
        return ConsumerSettings(
            ack_wait="60s",  # Longer timeout for replay
            max_ack_pending=min(base_settings.max_ack_pending, 256),  # Lower pending
            flow_control=True,
            idle_heartbeat="10s",  # Less frequent heartbeat
            deliver_policy="ByStartTime",
            rate_limit_bps=self.replay_settings.global_rate_limit_bps // self.replay_settings.max_replay_workers,
            max_deliver=2,  # Fewer retries for replay
            replay_policy=ReplayPolicy.INSTANT,
            priority=5  # Lower I/O priority
        )
    
    async def _can_start_replay_worker(self) -> bool:
        """Check if we can start a new replay worker."""
        return self._replay_workers_count < self.replay_settings.max_replay_workers
    
    async def _apply_replay_rate_limit(self) -> None:
        """Apply rate limiting for replay operations."""
        current_time = time.time()
        
        # Reset token bucket every second
        if current_time - self._last_rate_limit_reset >= 1.0:
            self._replay_rate_limiter_tokens = self.replay_settings.global_rate_limit_bps
            self._last_rate_limit_reset = current_time
        
        # Simple token bucket implementation
        if self._replay_rate_limiter_tokens <= 0:
            sleep_time = 1.0 - (current_time - self._last_rate_limit_reset)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self._replay_rate_limiter_tokens = self.replay_settings.global_rate_limit_bps
                self._last_rate_limit_reset = time.time()
        
        # Consume tokens (estimate 1KB per message)
        self._replay_rate_limiter_tokens = max(0, self._replay_rate_limiter_tokens - 1024)
    
    async def create_replay_consumer(
        self,
        domain: Domain,
        service: str,
        consumer_name: str,
        start_time: Optional[datetime] = None
    ) -> str:
        """Create a time-bounded replay consumer."""
        stream_name = f"xorb-{self.tenant_id}-{domain.value}-replay"
        replay_subjects = [self.subject_builder.build_replay_subject(domain, service)]
        
        return await self.create_consumer(
            stream_name,
            f"{consumer_name}-replay",
            replay_subjects,
            replay_mode=True
        )
    
    async def start_bounded_replay(
        self,
        domain: Domain,
        service: str,
        callback: Callable[[nats.Msg], Any],
        hours_back: int = 24,
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> None:
        """Start a time-bounded replay from specific hours back."""
        consumer_name = f"replay-{domain.value}-{service}-{int(time.time())}"
        
        # Create bounded replay consumer
        await self.create_replay_consumer(domain, service, consumer_name)
        
        # Start subscription with replay mode
        stream_name = f"xorb-{self.tenant_id}-{domain.value}-replay"
        await self.subscribe(
            stream_name,
            f"{consumer_name}-replay",
            callback,
            error_callback,
            replay_mode=True
        )


# Factory function for creating clients
def create_nats_client(
    tenant_id: str,
    servers: Optional[List[str]] = None,
    credentials_file: Optional[str] = None,
    consumer_settings: Optional[ConsumerSettings] = None,
    replay_settings: Optional[ReplaySettings] = None
) -> NATSClient:
    """Factory function to create a NATS client."""
    servers = servers or [os.getenv("NATS_URL", "nats://localhost:4222")]
    credentials_file = credentials_file or os.getenv("NATS_CREDENTIALS")

    return NATSClient(
        servers=servers,
        tenant_id=tenant_id,
        credentials_file=credentials_file,
        consumer_settings=consumer_settings,
        replay_settings=replay_settings
    )


# Example usage and testing functions
async def example_usage():
    """Example usage of the NATS client with replay support."""
    client = create_nats_client("t-qa")

    async with client.connection():
        # Create streams
        await client.create_stream(Domain.SCAN, StreamClass.LIVE)
        await client.create_stream(Domain.SCAN, StreamClass.REPLAY)  # Dedicated replay stream
        
        # Publish live messages
        await client.publish(
            Domain.SCAN,
            "nmap",
            Event.CREATED,
            {"target": "192.168.1.1", "ports": [80, 443]}
        )
        
        # Publish replay message
        await client.publish(
            Domain.SCAN,
            "nmap",
            Event.REPLAY,
            {"target": "192.168.1.1", "replay_data": "historical_scan"},
            replay=True
        )

        # Create live consumer
        live_consumer = await client.create_consumer(
            "xorb-t-qa-scan-live",
            "scan-processor-live",
            ["xorb.t-qa.scan.nmap.created"]
        )
        
        # Create replay consumer with time bounds
        replay_consumer = await client.create_replay_consumer(
            Domain.SCAN,
            "nmap",
            "scan-replay-processor"
        )

        # Live message handler
        async def live_message_handler(msg):
            print(f"LIVE: {msg.data.decode()}")
            
        # Replay message handler
        async def replay_message_handler(msg):
            print(f"REPLAY: {msg.data.decode()}")

        # Subscribe to live messages (high priority)
        await client.subscribe(
            "xorb-t-qa-scan-live",
            live_consumer,
            live_message_handler
        )
        
        # Start bounded replay (lower priority)
        await client.start_bounded_replay(
            Domain.SCAN,
            "nmap",
            replay_message_handler,
            hours_back=24  # Replay last 24 hours
        )


if __name__ == "__main__":
    # Run example with replay support
    asyncio.run(example_usage())
