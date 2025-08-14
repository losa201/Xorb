#!/usr/bin/env python3
"""
Basic Python Template for XORB Agent with Logging + Metrics
Demonstrates the core structure for autonomous agents in the XORB ecosystem.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any

import aioredis
import asyncpg
import structlog
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("xorb_agent_template")

# Prometheus Metrics
agent_operations_total = Counter(
    'xorb_agent_operations_total',
    'Total operations performed by agent',
    ['agent_id', 'operation_type', 'status']
)

operation_duration_seconds = Histogram(
    'xorb_agent_operation_duration_seconds',
    'Duration of agent operations',
    ['agent_id', 'operation_type']
)

agent_health_status = Gauge(
    'xorb_agent_health_status',
    'Agent health status (1=healthy, 0=unhealthy)',
    ['agent_id']
)

class XORBAgentTemplate:
    """
    Template for XORB autonomous agents

    Features:
    - Structured logging with structlog
    - Prometheus metrics emission
    - Redis pub/sub communication
    - PostgreSQL persistence
    - Graceful shutdown handling
    - Health check endpoints
    """

    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or f"template-agent-{uuid.uuid4().hex[:8]}"
        self.redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
        self.postgres_url = os.environ.get('POSTGRES_URL', 'postgresql://localhost:5432/xorb')
        self.prometheus_port = int(os.environ.get('PROMETHEUS_PORT', '8005'))

        # Runtime state
        self.is_running = False
        self.redis_pool = None
        self.db_pool = None
        self.operations_count = 0

        logger.info("XORBAgentTemplate initialized", agent_id=self.agent_id)

    async def initialize(self):
        """Initialize agent connections and components"""
        try:
            # Initialize Redis connection
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url, max_connections=5
            )

            # Initialize PostgreSQL connection
            self.db_pool = await asyncpg.create_pool(
                self.postgres_url, min_size=1, max_size=5
            )

            # Start Prometheus metrics server
            start_http_server(self.prometheus_port)

            # Test connections
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            await redis.ping()

            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            # Update health status
            agent_health_status.labels(agent_id=self.agent_id).set(1)

            # Log initialization
            await self._log_audit_event("agent_initialization", "success", {
                "prometheus_port": self.prometheus_port
            })

            logger.info("Agent initialized successfully",
                       prometheus_port=self.prometheus_port)

        except Exception as e:
            agent_health_status.labels(agent_id=self.agent_id).set(0)
            logger.error("Failed to initialize agent", error=str(e))
            raise

    async def start(self):
        """Start the agent"""
        self.is_running = True
        logger.info("Starting agent")

        try:
            # Start main processing loop
            await self._main_processing_loop()
        except Exception as e:
            logger.error("Agent failed", error=str(e))
            raise
        finally:
            self.is_running = False

    async def stop(self):
        """Stop the agent gracefully"""
        logger.info("Stopping agent")
        self.is_running = False

        # Update health status
        agent_health_status.labels(agent_id=self.agent_id).set(0)

        # Log shutdown
        await self._log_audit_event("agent_shutdown", "success", {
            "operations_performed": self.operations_count
        })

        # Close connections
        if self.redis_pool:
            await self.redis_pool.disconnect()
        if self.db_pool:
            await self.db_pool.close()

    async def _main_processing_loop(self):
        """Main agent processing loop"""
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        pubsub = redis.pubsub()

        try:
            # Subscribe to agent communication channels
            await pubsub.subscribe('xorb:coordination', 'xorb:test_channel')
            logger.info("Subscribed to coordination channels")

            while self.is_running:
                try:
                    # Get message with timeout
                    message = await pubsub.get_message(timeout=5.0)

                    if message and message['type'] == 'message':
                        await self._process_message(message)

                    # Perform periodic operations
                    await self._perform_periodic_operations()

                    # Small delay to prevent busy waiting
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error("Error in main processing loop", error=str(e))
                    await asyncio.sleep(5)

        finally:
            await pubsub.unsubscribe('xorb:coordination', 'xorb:test_channel')
            await pubsub.close()

    async def _process_message(self, message):
        """Process incoming Redis pub/sub message"""
        try:
            channel = message['channel'].decode('utf-8')
            data = json.loads(message['data'].decode('utf-8'))

            logger.info("Received message",
                       channel=channel,
                       message_type=data.get('type', 'unknown'))

            operation_start = time.time()

            # Process different message types
            if data.get('type') == 'test_message':
                await self._handle_test_message(data)
            elif data.get('type') == 'coordination_request':
                await self._handle_coordination_request(data)
            else:
                logger.debug("Unknown message type", message_type=data.get('type'))

            # Record metrics
            operation_duration = time.time() - operation_start
            operation_duration_seconds.labels(
                agent_id=self.agent_id,
                operation_type='message_processing'
            ).observe(operation_duration)

            agent_operations_total.labels(
                agent_id=self.agent_id,
                operation_type='message_processing',
                status='success'
            ).inc()

            self.operations_count += 1

        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            agent_operations_total.labels(
                agent_id=self.agent_id,
                operation_type='message_processing',
                status='failure'
            ).inc()

    async def _handle_test_message(self, data: dict[str, Any]):
        """Handle test message"""
        logger.info("Processing test message", data=data)

        # Echo back response
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        response = {
            "type": "test_response",
            "from_agent": self.agent_id,
            "original_message": data,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "acknowledged"
        }

        await redis.publish('xorb:coordination', json.dumps(response))

        await self._log_audit_event("test_message_processed", "success", data)

    async def _handle_coordination_request(self, data: dict[str, Any]):
        """Handle coordination request from another agent"""
        logger.info("Processing coordination request",
                   from_agent=data.get('from_agent'),
                   request_type=data.get('action_type'))

        # Simulate processing
        await asyncio.sleep(0.1)

        # Send response
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        response = {
            "type": "coordination_response",
            "from_agent": self.agent_id,
            "to_agent": data.get('from_agent'),
            "request_id": data.get('request_id'),
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }

        await redis.publish('xorb:coordination', json.dumps(response))

        await self._log_audit_event("coordination_request_handled", "success", {
            "from_agent": data.get('from_agent'),
            "action_type": data.get('action_type')
        })

    async def _perform_periodic_operations(self):
        """Perform periodic agent operations"""
        # Simulate periodic work (every 30 seconds)
        if self.operations_count % 30 == 0:
            operation_start = time.time()

            try:
                # Simulate some work
                await asyncio.sleep(0.1)

                # Publish status update
                redis = aioredis.Redis(connection_pool=self.redis_pool)
                status_update = {
                    "type": "agent_status",
                    "agent_id": self.agent_id,
                    "status": "active",
                    "operations_count": self.operations_count,
                    "uptime_seconds": int(time.time() - operation_start),
                    "timestamp": datetime.utcnow().isoformat()
                }

                await redis.publish('xorb:agent_status', json.dumps(status_update))

                # Record metrics
                operation_duration = time.time() - operation_start
                operation_duration_seconds.labels(
                    agent_id=self.agent_id,
                    operation_type='periodic_operation'
                ).observe(operation_duration)

                agent_operations_total.labels(
                    agent_id=self.agent_id,
                    operation_type='periodic_operation',
                    status='success'
                ).inc()

                logger.debug("Periodic operation completed",
                           operations_count=self.operations_count)

            except Exception as e:
                logger.error("Periodic operation failed", error=str(e))
                agent_operations_total.labels(
                    agent_id=self.agent_id,
                    operation_type='periodic_operation',
                    status='failure'
                ).inc()

    async def _log_audit_event(self, action: str, outcome: str, details: dict[str, Any]):
        """Log event to audit trail"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_logs (agent_id, action, target, outcome, details, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, self.agent_id, action, "system", outcome,
                    json.dumps(details), datetime.utcnow())

        except Exception as e:
            logger.error("Failed to log audit event", error=str(e))

    async def get_health_status(self) -> dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "running" if self.is_running else "stopped",
            "operations_count": self.operations_count,
            "prometheus_port": self.prometheus_port,
            "redis_connected": self.redis_pool is not None,
            "database_connected": self.db_pool is not None,
            "timestamp": datetime.utcnow().isoformat()
        }

async def send_test_threat_signal():
    """Send a test threat signal to trigger autonomous response"""
    try:
        redis = aioredis.Redis.from_url('redis://localhost:6379')

        test_signal = {
            "signal_id": f"test_{int(time.time())}",
            "threat_type": "malware",
            "priority": "critical",
            "confidence": 0.85,
            "source_ip": "192.168.1.100",
            "target_assets": ["web-server-01", "db-server-02"],
            "indicators": ["malicious.com", "abc123def456"],
            "timestamp": datetime.utcnow().isoformat(),
            "context": {
                "detection_method": "signature",
                "attack_vector": "network"
            }
        }

        # Send to high priority threats channel
        await redis.lpush('high_priority_threats', json.dumps(test_signal))

        # Publish coordination message
        coordination_msg = {
            "type": "test_message",
            "from_agent": "test_client",
            "message": "Test coordination message",
            "timestamp": datetime.utcnow().isoformat()
        }

        await redis.publish('xorb:coordination', json.dumps(coordination_msg))

        print("Test threat signal and coordination message sent successfully!")

        await redis.close()

    except Exception as e:
        print(f"Failed to send test signal: {e}")

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="XORB Agent Template")
    parser.add_argument("--mode", choices=["agent", "test"], default="agent",
                       help="Run mode: agent or test")
    parser.add_argument("--agent-id", help="Agent ID")

    args = parser.parse_args()

    if args.mode == "test":
        print("Sending test threat signal...")
        await send_test_threat_signal()
        return

    # Setup logging
    log_level = os.environ.get('LOG_LEVEL', 'INFO')
    logging.basicConfig(level=getattr(logging, log_level))

    # Initialize and start agent
    agent = XORBAgentTemplate(agent_id=args.agent_id)

    try:
        await agent.initialize()
        await agent.start()

    except KeyboardInterrupt:
        logger.info("Agent interrupted by user")
    except Exception as e:
        logger.error("Agent failed", error=str(e))
        return 1
    finally:
        await agent.stop()

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
