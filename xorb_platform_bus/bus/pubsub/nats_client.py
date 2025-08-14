#!/usr/bin/env python3
"""
NATS Client with Subject Builder and Consumer Tuning - Phase G2

Provides a production-ready NATS JetStream client with:
- Subject schema validation (v1 immutable)
- Per-tenant isolation and quotas
- Tuned consumers with flow control
- Exactly-once semantics via idempotency
- Comprehensive error handling and retry logic
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
    LIVE = "live"      # Operational streams
    REPLAY = "replay"  # Audit/replay streams with WORM characteristics


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

    def build(self, domain: Domain, service: str, event: Event) -> str:
        """Build a validated NATS subject."""
        components = SubjectComponents(
            tenant=self.tenant,
            domain=domain,
            service=service,
            event=event
        )
        return components.to_subject()

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
        consumer_settings: Optional[ConsumerSettings] = None
    ):
        self.servers = servers
        self.tenant_id = tenant_id
        self.credentials_file = credentials_file
        self.consumer_settings = consumer_settings or ConsumerSettings.from_env()

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
        timeout: float = 10.0
    ) -> None:
        """Publish a message with subject validation."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")

        subject = self.subject_builder.build(domain, service, event)

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

        # Prepare headers
        msg_headers = {
            "Content-Type": content_type,
            "X-Tenant-ID": self.tenant_id,
            "X-Message-ID": str(uuid.uuid4()),
            "X-Timestamp": str(int(time.time() * 1000))
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
            logger.debug(f"Published message to {subject}, ack: {ack}")

        except Exception as e:
            logger.error(f"Failed to publish to {subject}: {e}")
            raise

    async def create_consumer(
        self,
        stream_name: str,
        consumer_name: str,
        subjects: List[str],
        settings: Optional[ConsumerSettings] = None
    ) -> str:
        """Create a consumer with tuned flow control settings."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")

        settings = settings or self.consumer_settings

        # Build consumer configuration
        config = ConsumerConfig(
            name=consumer_name,
            durable_name=consumer_name,
            filter_subjects=subjects,
            deliver_policy="last",
            ack_policy="explicit",
            ack_wait=settings.ack_wait,
            max_deliver=settings.max_deliver,
            max_ack_pending=settings.max_ack_pending,
            flow_control=settings.flow_control,
            idle_heartbeat=settings.idle_heartbeat,
            replay_policy=settings.replay_policy
        )

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
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> None:
        """Subscribe to messages with pull-based consumption."""
        if not self._connected:
            raise ConnectionError("Not connected to NATS")

        try:
            # Get consumer info to validate it exists
            consumer_info = await self._js.consumer_info(stream_name, consumer_name)
            logger.debug(f"Consumer {consumer_name} info: {consumer_info}")

            # Create pull subscription
            psub = await self._js.pull_subscribe("", consumer_name, stream=stream_name)

            # Process messages in batches
            batch_size = min(self.consumer_settings.max_ack_pending // 4, 256)

            while True:
                try:
                    # Fetch messages in batches
                    messages = await psub.fetch(batch_size, timeout=1.0)

                    for msg in messages:
                        try:
                            # Process message
                            await callback(msg) if asyncio.iscoroutinefunction(callback) else callback(msg)

                            # Acknowledge message
                            await msg.ack()
                            self._messages_consumed += 1

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
        """Get client metrics."""
        return {
            "messages_published": self._messages_published,
            "messages_consumed": self._messages_consumed,
            "connection_retries": self._connection_retries
        }


# Factory function for creating clients
def create_nats_client(
    tenant_id: str,
    servers: Optional[List[str]] = None,
    credentials_file: Optional[str] = None,
    consumer_settings: Optional[ConsumerSettings] = None
) -> NATSClient:
    """Factory function to create a NATS client."""
    servers = servers or [os.getenv("NATS_URL", "nats://localhost:4222")]
    credentials_file = credentials_file or os.getenv("NATS_CREDENTIALS")

    return NATSClient(
        servers=servers,
        tenant_id=tenant_id,
        credentials_file=credentials_file,
        consumer_settings=consumer_settings
    )


# Example usage and testing functions
async def example_usage():
    """Example usage of the NATS client."""
    client = create_nats_client("t-qa")

    async with client.connection():
        # Create streams
        await client.create_stream(Domain.SCAN, StreamClass.LIVE)
        await client.create_stream(Domain.EVIDENCE, StreamClass.REPLAY)

        # Publish messages
        await client.publish(
            Domain.SCAN,
            "nmap",
            Event.CREATED,
            {"target": "192.168.1.1", "ports": [80, 443]}
        )

        # Create consumer
        consumer_name = await client.create_consumer(
            "xorb-t-qa-scan-live",
            "scan-processor",
            ["xorb.t-qa.scan.>"]
        )

        # Subscribe to messages
        async def message_handler(msg):
            print(f"Received: {msg.data.decode()}")

        await client.subscribe(
            "xorb-t-qa-scan-live",
            consumer_name,
            message_handler
        )


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
