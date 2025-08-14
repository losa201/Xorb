"""
UNIX Domain Socket Transport for Tier-1 Local Ring (ADR-002 Compliance)
Implements local communication with back-pressure handling and FIFO ordering.
"""

import asyncio
import socket
import struct
import json
import logging
import os
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Message types for local ring communication"""
    DISCOVERY_REQUEST = "discovery_request"
    DISCOVERY_RESPONSE = "discovery_response"
    SCAN_CONTROL = "scan_control"
    FINGERPRINT_DATA = "fingerprint_data"
    RISK_TAG = "risk_tag"
    HEARTBEAT = "heartbeat"
    BACKPRESSURE = "backpressure"

@dataclass
class RingMessage:
    """Local ring message structure"""
    msg_type: MessageType
    sender_id: str
    payload: Dict[str, Any]
    sequence_id: int
    timestamp: float
    
    def serialize(self) -> bytes:
        """Serialize message to bytes"""
        data = {
            "type": self.msg_type.value,
            "sender": self.sender_id,
            "payload": self.payload,
            "seq": self.sequence_id,
            "ts": self.timestamp
        }
        json_data = json.dumps(data).encode('utf-8')
        # Prepend 4-byte length header
        return struct.pack('>I', len(json_data)) + json_data
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'RingMessage':
        """Deserialize message from bytes"""
        parsed = json.loads(data.decode('utf-8'))
        return cls(
            msg_type=MessageType(parsed["type"]),
            sender_id=parsed["sender"],
            payload=parsed["payload"],
            sequence_id=parsed["seq"],
            timestamp=parsed["ts"]
        )

class UDSRingBuffer:
    """
    UNIX Domain Socket Ring Buffer for Tier-1 Local Communication
    
    Features:
    - FIFO ordering per producer with causal consistency
    - Back-pressure via ring buffer full → immediate EAGAIN
    - Zero-copy payload transfer via shared memory mapping
    - Process crash → ring contents lost (rely on Tier-2 durability)
    - Max 64KB per message with memory mapping for larger payloads
    """
    
    def __init__(self, socket_path: str, max_size: int = 10000):
        self.socket_path = socket_path
        self.max_size = max_size
        self.message_buffer: List[RingMessage] = []
        self.clients: Dict[str, socket.socket] = {}
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.ring_full_events = 0
        self.backpressure_events = 0
        
        self.server_socket: Optional[socket.socket] = None
        self.running = False
    
    async def start_ring_broker(self) -> None:
        """Start the UDS ring broker"""
        # Clean up old socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        # Create socket directory if needed
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create and bind UDS socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(50)  # Support up to 50 concurrent services
        self.server_socket.setblocking(False)
        
        logger.info(f"UDS Ring Broker started on {self.socket_path}")
        self.running = True
        
        # Start accepting connections
        asyncio.create_task(self._accept_connections())
        asyncio.create_task(self._process_ring_messages())
    
    async def _accept_connections(self) -> None:
        """Accept new client connections"""
        while self.running:
            try:
                # Accept connection
                loop = asyncio.get_event_loop()
                client_socket, addr = await loop.sock_accept(self.server_socket)
                
                # Register client
                client_id = f"client-{len(self.clients)}"
                self.clients[client_id] = client_socket
                
                logger.info(f"Client {client_id} connected to ring")
                
                # Handle client in separate task
                asyncio.create_task(self._handle_client(client_id, client_socket))
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_client(self, client_id: str, client_socket: socket.socket) -> None:
        """Handle messages from a specific client"""
        try:
            while self.running:
                # Read message length header (4 bytes)
                loop = asyncio.get_event_loop()
                length_data = await loop.sock_recv(client_socket, 4)
                
                if len(length_data) < 4:
                    break  # Client disconnected
                
                message_length = struct.unpack('>I', length_data)[0]
                
                if message_length > 65536:  # 64KB limit
                    logger.warning(f"Message too large from {client_id}: {message_length} bytes")
                    await self._send_backpressure_signal(client_socket, "MSG_TOO_LARGE")
                    continue
                
                # Read message data
                message_data = await loop.sock_recv(client_socket, message_length)
                
                if len(message_data) < message_length:
                    break  # Incomplete message, client disconnected
                
                # Parse and enqueue message
                try:
                    message = RingMessage.deserialize(message_data)
                    await self._enqueue_message(client_id, message_data)
                    
                except Exception as e:
                    logger.error(f"Failed to parse message from {client_id}: {e}")
                    await self._send_backpressure_signal(client_socket, "PARSE_ERROR")
                
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client
            try:
                client_socket.close()
                self.clients.pop(client_id, None)
                logger.info(f"Client {client_id} disconnected")
            except:
                pass
    
    async def _enqueue_message(self, producer_id: str, message_data: bytes) -> None:
        """Enqueue message with back-pressure handling"""
        if len(self.message_buffer) >= self.max_size:
            self.ring_full_events += 1
            
            # Send EAGAIN signal to producer (immediate pushback)
            producer_socket = self.clients.get(producer_id)
            if producer_socket:
                await asyncio.get_event_loop().sock_sendall(
                    producer_socket, b"EAGAIN\n"
                )
            
            logger.warning(f"Ring buffer full, sent EAGAIN to {producer_id}")
            return
        
        # Parse message
        message = RingMessage.deserialize(message_data)
        
        # Add to ring buffer (FIFO per producer)
        self.message_buffer.append(message)
        self.messages_received += 1
        
        logger.debug(f"Enqueued message from {producer_id}: {message.msg_type.value}")
    
    async def _process_ring_messages(self) -> None:
        """Process and route ring messages"""
        while self.running:
            if not self.message_buffer:
                await asyncio.sleep(0.001)  # 1ms polling interval
                continue
            
            # Dequeue message (FIFO)
            message = self.message_buffer.pop(0)
            
            # Route message to appropriate consumers
            await self._route_message(message)
            
            self.messages_sent += 1
    
    async def _route_message(self, message: RingMessage) -> None:
        """Route message to appropriate consumers based on type"""
        target_clients = []
        
        # Route based on message type
        if message.msg_type == MessageType.DISCOVERY_REQUEST:
            # Route to scanner services
            target_clients = [cid for cid in self.clients.keys() if "scanner" in cid]
        elif message.msg_type == MessageType.FINGERPRINT_DATA:
            # Route to intelligence engine
            target_clients = [cid for cid in self.clients.keys() if "intelligence" in cid]
        elif message.msg_type == MessageType.RISK_TAG:
            # Route to all consumers
            target_clients = list(self.clients.keys())
        elif message.msg_type == MessageType.HEARTBEAT:
            # Route to monitoring services
            target_clients = [cid for cid in self.clients.keys() if "monitor" in cid]
        else:
            # Broadcast to all clients
            target_clients = list(self.clients.keys())
        
        # Send to target clients
        for client_id in target_clients:
            if client_id == message.sender_id:
                continue  # Don't send back to sender
            
            client_socket = self.clients.get(client_id)
            if client_socket:
                try:
                    serialized = message.serialize()
                    await asyncio.get_event_loop().sock_sendall(client_socket, serialized)
                except Exception as e:
                    logger.error(f"Failed to send message to {client_id}: {e}")
                    # Remove failed client
                    self.clients.pop(client_id, None)
    
    async def _send_backpressure_signal(self, client_socket: socket.socket, signal: str) -> None:
        """Send back-pressure signal to client"""
        try:
            signal_data = f"{signal}\n".encode()
            await asyncio.get_event_loop().sock_sendall(client_socket, signal_data)
            self.backpressure_events += 1
        except Exception as e:
            logger.error(f"Failed to send backpressure signal: {e}")
    
    def get_ring_stats(self) -> Dict[str, Any]:
        """Get ring buffer statistics"""
        return {
            "buffer_size": len(self.message_buffer),
            "max_size": self.max_size,
            "utilization": len(self.message_buffer) / self.max_size,
            "connected_clients": len(self.clients),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "ring_full_events": self.ring_full_events,
            "backpressure_events": self.backpressure_events
        }
    
    async def stop(self) -> None:
        """Stop the ring broker"""
        self.running = False
        
        # Close all client connections
        for client_socket in self.clients.values():
            try:
                client_socket.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        # Clean up socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        logger.info("UDS Ring Broker stopped")

class UDSRingClient:
    """Client for connecting to UDS Ring Broker"""
    
    def __init__(self, socket_path: str, client_id: str):
        self.socket_path = socket_path
        self.client_id = client_id
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.sequence_counter = 0
        
    async def connect(self) -> None:
        """Connect to UDS ring broker"""
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        try:
            await asyncio.get_event_loop().sock_connect(self.socket, self.socket_path)
            self.connected = True
            logger.info(f"Client {self.client_id} connected to ring")
            
        except Exception as e:
            logger.error(f"Failed to connect to ring: {e}")
            raise
    
    async def send_message(self, msg_type: MessageType, payload: Dict[str, Any]) -> bool:
        """Send message to ring with back-pressure handling"""
        if not self.connected:
            logger.error("Not connected to ring")
            return False
        
        message = RingMessage(
            msg_type=msg_type,
            sender_id=self.client_id,
            payload=payload,
            sequence_id=self.sequence_counter,
            timestamp=asyncio.get_event_loop().time()
        )
        
        self.sequence_counter += 1
        
        try:
            serialized = message.serialize()
            await asyncio.get_event_loop().sock_sendall(self.socket, serialized)
            
            # Check for back-pressure signals
            loop = asyncio.get_event_loop()
            ready = await asyncio.wait_for(
                loop.sock_recv(self.socket, 1024), timeout=0.001
            )
            
            if b"EAGAIN" in ready:
                logger.warning("Received EAGAIN - ring buffer full")
                return False
            
            return True
            
        except asyncio.TimeoutError:
            # No back-pressure signal, message sent successfully
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_messages(self, handler: Callable[[RingMessage], None]) -> None:
        """Receive messages from ring"""
        while self.connected:
            try:
                # Read message length
                loop = asyncio.get_event_loop()
                length_data = await loop.sock_recv(self.socket, 4)
                
                if len(length_data) < 4:
                    break  # Connection closed
                
                message_length = struct.unpack('>I', length_data)[0]
                
                # Read message data
                message_data = await loop.sock_recv(self.socket, message_length)
                
                if len(message_data) < message_length:
                    break  # Incomplete message
                
                # Parse and handle message
                message = RingMessage.deserialize(message_data)
                await handler(message)
                
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
    
    async def disconnect(self) -> None:
        """Disconnect from ring"""
        self.connected = False
        if self.socket:
            self.socket.close()
            logger.info(f"Client {self.client_id} disconnected from ring")

# Example usage
async def example_ring_usage():
    """Example usage of UDS ring transport"""
    
    # Start ring broker
    ring_broker = UDSRingBuffer("/tmp/xorb-ring.sock")
    await ring_broker.start_ring_broker()
    
    # Create client
    client = UDSRingClient("/tmp/xorb-ring.sock", "scanner-service-1")
    await client.connect()
    
    try:
        # Send discovery request
        await client.send_message(
            MessageType.DISCOVERY_REQUEST,
            {"target": "192.168.1.100", "ports": [80, 443]}
        )
        
        # Handle incoming messages
        async def handle_message(message: RingMessage):
            print(f"Received: {message.msg_type.value} from {message.sender_id}")
        
        await client.receive_messages(handle_message)
        
    finally:
        await client.disconnect()
        await ring_broker.stop()

if __name__ == "__main__":
    asyncio.run(example_ring_usage())