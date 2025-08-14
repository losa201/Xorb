import asyncio
from typing import Optional, Callable, Any
from nats.aio.client import Client as NATSClient
from nats.js.client import JetStreamContext

class NATSBus:
    """NATS JetStream helper for XORB event bus operations."""
    
    def __init__(self, nc: NATSClient, js: JetStreamContext):
        self.nc = nc
        self.js = js
        
    @classmethod
    async def connect(cls, servers: list[str] = ["nats://localhost:4222"]):
        """Connect to NATS cluster and create JetStream context."""
        nc = NATSClient()
        await nc.connect(servers=servers)
        js = nc.jetstream()
        return cls(nc, js)
        
    async def publish(
        self,
        subject: str,
        payload: bytes,
        timeout: float = 5.0
    ) -> None:
        """Publish message to NATS JetStream with timeout."""
        try:
            await asyncio.wait_for(
                self.js.publish(subject, payload),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError("NATS publish operation timed out")
        
    async def subscribe(
        self,
        subject: str,
        durable_name: str,
        callback: Callable[[Any], Any],
        ack_wait: float = 30.0,
        max_ack_pending: int = 1024,
        flow_control: bool = True,
        idle_heartbeat: float = 5.0
    ) -> None:
        """Create durable pull consumer with configured options."""
        consumer_config = {
            "durable_name": durable_name,
            "ack_wait": ack_wait,
            "max_ack_pending": max_ack_pending,
            "flow_control": flow_control,
            "idle_heartbeat": idle_heartbeat
        }
        
        # Create ephemeral consumer if no durable name provided
        if not durable_name:
            consumer_config.pop("durable_name", None)
            
        sub = await self.js.pull_subscribe(
            subject,
            durable=durable_name or None,
            config=consumer_config
        )
        
        async def _message_handler(msg):
            try:
                await callback(msg)
            except Exception as e:
                print(f"Error processing message: {e}")
                
        await sub.fetch_handler(10, _message_handler)
        
    async def close(self) -> None:
        """Close NATS connection gracefully."""
        if self.nc and not self.nc.is_closed:
            await self.nc.close()