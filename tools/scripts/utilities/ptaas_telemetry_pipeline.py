#!/usr/bin/env python3
"""
XORB PTaaS Real-time Telemetry Pipeline
High-performance data ingestion and streaming for PTaaS-Learning integration
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
from collections import defaultdict, deque
import hashlib

import aiohttp
import aiokafka
import redis.asyncio as redis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import structlog

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(
    title="XORB PTaaS Telemetry Pipeline",
    description="Real-time data ingestion and streaming for PTaaS learning integration",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TelemetryEventType(Enum):
    AGENT_HEARTBEAT = "agent_heartbeat"
    SCAN_STARTED = "scan_started"
    SCAN_PROGRESS = "scan_progress"
    SCAN_COMPLETED = "scan_completed"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    FALSE_POSITIVE_FLAGGED = "false_positive_flagged"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_OCCURRED = "error_occurred"
    ANOMALY_DETECTED = "anomaly_detected"
    RESOURCE_UTILIZATION = "resource_utilization"

class DataQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class TelemetryEvent:
    event_id: str
    event_type: TelemetryEventType
    timestamp: datetime
    agent_id: str
    test_id: Optional[str]
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    quality_score: float
    data_quality: DataQuality
    processing_latency: Optional[float] = None

@dataclass
class StreamingMetrics:
    events_per_second: float
    processing_latency_p95: float
    data_quality_distribution: Dict[str, int]
    pipeline_health_score: float
    backlog_size: int
    error_rate: float
    last_updated: datetime

class TelemetryValidator:
    """Advanced telemetry data validation and quality assessment"""

    def __init__(self):
        self.schema_registry = {
            TelemetryEventType.AGENT_HEARTBEAT: {
                'required_fields': ['agent_id', 'status', 'timestamp'],
                'field_types': {'status': str, 'load': float, 'memory_usage': float}
            },
            TelemetryEventType.VULNERABILITY_DETECTED: {
                'required_fields': ['vulnerability_type', 'severity', 'confidence'],
                'field_types': {'confidence': float, 'severity': str, 'cvss_score': float}
            },
            TelemetryEventType.PERFORMANCE_METRIC: {
                'required_fields': ['metric_name', 'metric_value', 'unit'],
                'field_types': {'metric_value': (int, float), 'timestamp': str}
            }
        }

        self.anomaly_thresholds = {
            'processing_time': 30.0,  # seconds
            'memory_usage': 0.9,      # 90% threshold
            'cpu_usage': 0.85,        # 85% threshold
            'error_rate': 0.1         # 10% threshold
        }

    def validate_event(self, event_data: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """Validate telemetry event and return quality score"""
        errors = []
        quality_score = 1.0

        # Check required fields
        if 'event_type' not in event_data:
            errors.append("Missing event_type")
            quality_score -= 0.3

        if 'timestamp' not in event_data:
            errors.append("Missing timestamp")
            quality_score -= 0.2
        else:
            # Validate timestamp format
            try:
                datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                errors.append("Invalid timestamp format")
                quality_score -= 0.2

        if 'agent_id' not in event_data:
            errors.append("Missing agent_id")
            quality_score -= 0.2

        # Schema-specific validation
        event_type = event_data.get('event_type')
        if event_type:
            try:
                event_enum = TelemetryEventType(event_type)
                schema = self.schema_registry.get(event_enum)

                if schema:
                    # Check required fields
                    for field in schema['required_fields']:
                        if field not in event_data.get('payload', {}):
                            errors.append(f"Missing required field: {field}")
                            quality_score -= 0.1

                    # Check field types
                    payload = event_data.get('payload', {})
                    for field, expected_type in schema['field_types'].items():
                        if field in payload and not isinstance(payload[field], expected_type):
                            errors.append(f"Invalid type for field {field}")
                            quality_score -= 0.05

            except ValueError:
                errors.append(f"Unknown event type: {event_type}")
                quality_score -= 0.3

        # Data freshness check
        if 'timestamp' in event_data:
            try:
                event_time = datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00'))
                age = (datetime.now() - event_time.replace(tzinfo=None)).total_seconds()

                if age > 300:  # 5 minutes old
                    quality_score -= 0.1
                elif age > 60:  # 1 minute old
                    quality_score -= 0.05

            except Exception:
                pass

        # Anomaly detection
        payload = event_data.get('payload', {})
        for metric, threshold in self.anomaly_thresholds.items():
            if metric in payload:
                value = payload[metric]
                if isinstance(value, (int, float)) and value > threshold:
                    quality_score -= 0.05

        quality_score = max(0.0, quality_score)
        is_valid = len(errors) == 0 and quality_score > 0.5

        return is_valid, quality_score, errors

class HighThroughputBuffer:
    """High-performance circular buffer for telemetry events"""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.index = 0
        self.total_events = 0
        self.last_flush = time.time()

    def add_event(self, event: TelemetryEvent):
        """Add event to buffer with automatic overflow handling"""
        self.buffer.append(event)
        self.index = (self.index + 1) % self.max_size
        self.total_events += 1

    def get_batch(self, batch_size: int = 1000) -> List[TelemetryEvent]:
        """Get batch of events for processing"""
        batch = []

        for _ in range(min(batch_size, len(self.buffer))):
            if self.buffer:
                batch.append(self.buffer.popleft())

        return batch

    def size(self) -> int:
        return len(self.buffer)

    def is_ready_for_flush(self, flush_interval: float = 5.0) -> bool:
        """Check if buffer should be flushed based on time or size"""
        current_time = time.time()
        time_based = (current_time - self.last_flush) >= flush_interval
        size_based = len(self.buffer) >= (self.max_size * 0.8)

        return time_based or size_based

class TelemetryPipeline:
    """High-performance telemetry data pipeline"""

    def __init__(self):
        # Connections
        self.redis_client = None
        self.postgres_pool = None
        self.kafka_producer = None

        # Processing components
        self.validator = TelemetryValidator()
        self.buffer = HighThroughputBuffer()
        self.processing_queue = asyncio.Queue(maxsize=10000)

        # Metrics and monitoring
        self.metrics_registry = CollectorRegistry()
        self._setup_metrics()

        # Pipeline state
        self.is_running = False
        self.worker_tasks = []
        self.streaming_metrics = StreamingMetrics(
            events_per_second=0.0,
            processing_latency_p95=0.0,
            data_quality_distribution={},
            pipeline_health_score=1.0,
            backlog_size=0,
            error_rate=0.0,
            last_updated=datetime.now()
        )

        # Rate limiting and flow control
        self.rate_limiter = defaultdict(lambda: deque(maxlen=100))
        self.backpressure_threshold = 50000

    def _setup_metrics(self):
        """Setup Prometheus metrics for pipeline monitoring"""
        self.events_ingested = Counter(
            'ptaas_telemetry_events_total',
            'Total telemetry events ingested',
            ['event_type', 'agent_id', 'quality'],
            registry=self.metrics_registry
        )

        self.processing_latency = Histogram(
            'ptaas_telemetry_processing_seconds',
            'Time taken to process telemetry events',
            ['event_type'],
            registry=self.metrics_registry
        )

        self.pipeline_throughput = Gauge(
            'ptaas_telemetry_throughput_eps',
            'Events processed per second',
            registry=self.metrics_registry
        )

        self.data_quality_score = Gauge(
            'ptaas_telemetry_quality_score',
            'Average data quality score',
            ['event_type'],
            registry=self.metrics_registry
        )

        self.pipeline_health = Gauge(
            'ptaas_telemetry_pipeline_health',
            'Overall pipeline health score',
            registry=self.metrics_registry
        )

        self.buffer_size = Gauge(
            'ptaas_telemetry_buffer_size',
            'Current buffer size',
            registry=self.metrics_registry
        )

    async def initialize(self):
        """Initialize all pipeline connections and components"""
        try:
            # Redis connection for caching
            self.redis_client = redis.from_url(
                "redis://localhost:6379",
                decode_responses=True,
                max_connections=20
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for telemetry pipeline")

            # PostgreSQL connection pool for persistence
            self.postgres_pool = await asyncpg.create_pool(
                "postgresql://xorb_user:xorb_secure_2024@localhost:5432/xorb_ptaas",
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool established")

            # Kafka producer for streaming
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers='localhost:9092',
                value_serializer=lambda x: json.dumps(x, default=str).encode(),
                compression_type="lz4",
                batch_size=16384,
                linger_ms=10,
                max_request_size=1048576
            )
            await self.kafka_producer.start()
            logger.info("Kafka producer initialized for telemetry streaming")

            # Initialize database schema
            await self._initialize_database_schema()

        except Exception as e:
            logger.error("Failed to initialize telemetry pipeline", error=str(e))
            raise

    async def _initialize_database_schema(self):
        """Initialize PostgreSQL schema for telemetry storage"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS ptaas_telemetry (
            event_id UUID PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            agent_id VARCHAR(50) NOT NULL,
            test_id VARCHAR(50),
            payload JSONB NOT NULL,
            metadata JSONB,
            quality_score FLOAT NOT NULL,
            data_quality VARCHAR(20) NOT NULL,
            processing_latency FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON ptaas_telemetry(timestamp);
        CREATE INDEX IF NOT EXISTS idx_telemetry_agent ON ptaas_telemetry(agent_id);
        CREATE INDEX IF NOT EXISTS idx_telemetry_type ON ptaas_telemetry(event_type);
        CREATE INDEX IF NOT EXISTS idx_telemetry_quality ON ptaas_telemetry(quality_score);

        CREATE TABLE IF NOT EXISTS ptaas_telemetry_aggregates (
            id SERIAL PRIMARY KEY,
            window_start TIMESTAMPTZ NOT NULL,
            window_end TIMESTAMPTZ NOT NULL,
            agent_id VARCHAR(50) NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            event_count INTEGER NOT NULL,
            avg_quality_score FLOAT NOT NULL,
            avg_processing_latency FLOAT,
            min_timestamp TIMESTAMPTZ,
            max_timestamp TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_aggregates_window ON ptaas_telemetry_aggregates(window_start, window_end);
        """

        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)

        logger.info("Database schema initialized for telemetry storage")

    async def ingest_event(self, event_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Ingest telemetry event with validation and quality assessment"""
        start_time = time.time()

        try:
            # Rate limiting check
            agent_id = event_data.get('agent_id', 'unknown')
            current_time = time.time()
            self.rate_limiter[agent_id].append(current_time)

            # Check rate limit (max 1000 events per minute per agent)
            recent_events = [t for t in self.rate_limiter[agent_id] if current_time - t < 60]
            if len(recent_events) > 1000:
                return False, "Rate limit exceeded"

            # Validate event
            is_valid, quality_score, errors = self.validator.validate_event(event_data)

            if not is_valid and quality_score < 0.3:
                logger.warning("Event validation failed",
                             event_data=event_data, errors=errors)
                return False, f"Validation failed: {', '.join(errors)}"

            # Determine data quality
            if quality_score >= 0.9:
                data_quality = DataQuality.HIGH
            elif quality_score >= 0.7:
                data_quality = DataQuality.MEDIUM
            elif quality_score >= 0.5:
                data_quality = DataQuality.LOW
            else:
                data_quality = DataQuality.INVALID

            # Create telemetry event
            telemetry_event = TelemetryEvent(
                event_id=str(uuid.uuid4()),
                event_type=TelemetryEventType(event_data.get('event_type')),
                timestamp=datetime.fromisoformat(
                    event_data.get('timestamp', datetime.now().isoformat()).replace('Z', '+00:00')
                ),
                agent_id=agent_id,
                test_id=event_data.get('test_id'),
                payload=event_data.get('payload', {}),
                metadata=event_data.get('metadata', {}),
                quality_score=quality_score,
                data_quality=data_quality,
                processing_latency=time.time() - start_time
            )

            # Add to buffer
            self.buffer.add_event(telemetry_event)

            # Update metrics
            self.events_ingested.labels(
                event_type=telemetry_event.event_type.value,
                agent_id=agent_id,
                quality=data_quality.value
            ).inc()

            self.processing_latency.labels(
                event_type=telemetry_event.event_type.value
            ).observe(telemetry_event.processing_latency)

            # Add to processing queue if high quality
            if data_quality in [DataQuality.HIGH, DataQuality.MEDIUM]:
                try:
                    self.processing_queue.put_nowait(telemetry_event)
                except asyncio.QueueFull:
                    logger.warning("Processing queue full, dropping event")

            return True, "Event ingested successfully"

        except Exception as e:
            logger.error("Error ingesting telemetry event", error=str(e))
            return False, f"Ingestion error: {str(e)}"

    async def start_pipeline(self):
        """Start the telemetry pipeline with multiple worker tasks"""
        if self.is_running:
            return

        self.is_running = True

        # Start worker tasks
        self.worker_tasks = [
            asyncio.create_task(self._processing_worker()),
            asyncio.create_task(self._persistence_worker()),
            asyncio.create_task(self._streaming_worker()),
            asyncio.create_task(self._metrics_worker()),
            asyncio.create_task(self._buffer_flush_worker())
        ]

        logger.info("Telemetry pipeline started with 5 worker tasks")

    async def stop_pipeline(self):
        """Stop the telemetry pipeline gracefully"""
        self.is_running = False

        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        # Flush remaining data
        await self._flush_buffer()

        logger.info("Telemetry pipeline stopped gracefully")

    async def _processing_worker(self):
        """Worker task for processing telemetry events"""
        logger.info("Starting telemetry processing worker")

        while self.is_running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )

                # Process event for learning insights
                await self._process_for_learning(event)

                # Update quality metrics
                self.data_quality_score.labels(
                    event_type=event.event_type.value
                ).set(event.quality_score)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error in processing worker", error=str(e))
                await asyncio.sleep(1)

    async def _persistence_worker(self):
        """Worker task for persisting telemetry data"""
        logger.info("Starting telemetry persistence worker")

        while self.is_running:
            try:
                # Get batch from buffer
                batch = self.buffer.get_batch(batch_size=1000)

                if batch:
                    await self._persist_batch(batch)
                    logger.debug(f"Persisted batch of {len(batch)} events")

                await asyncio.sleep(2)  # Batch every 2 seconds

            except Exception as e:
                logger.error("Error in persistence worker", error=str(e))
                await asyncio.sleep(5)

    async def _streaming_worker(self):
        """Worker task for streaming data to Kafka"""
        logger.info("Starting telemetry streaming worker")

        while self.is_running:
            try:
                # Check if buffer is ready for streaming
                if self.buffer.size() > 100:  # Stream in batches
                    batch = self.buffer.get_batch(batch_size=500)

                    for event in batch:
                        await self._stream_event(event)

                await asyncio.sleep(1)  # Stream every second

            except Exception as e:
                logger.error("Error in streaming worker", error=str(e))
                await asyncio.sleep(3)

    async def _metrics_worker(self):
        """Worker task for updating pipeline metrics"""
        logger.info("Starting telemetry metrics worker")

        event_counts = deque(maxlen=60)  # Last 60 seconds

        while self.is_running:
            try:
                current_time = time.time()
                current_events = self.buffer.total_events
                event_counts.append((current_time, current_events))

                # Calculate throughput (events per second)
                if len(event_counts) >= 2:
                    time_diff = event_counts[-1][0] - event_counts[0][0]
                    event_diff = event_counts[-1][1] - event_counts[0][1]

                    if time_diff > 0:
                        throughput = event_diff / time_diff
                        self.pipeline_throughput.set(throughput)
                        self.streaming_metrics.events_per_second = throughput

                # Update buffer size metric
                buffer_size = self.buffer.size()
                self.buffer_size.set(buffer_size)
                self.streaming_metrics.backlog_size = buffer_size

                # Calculate pipeline health
                health_score = self._calculate_pipeline_health()
                self.pipeline_health.set(health_score)
                self.streaming_metrics.pipeline_health_score = health_score

                self.streaming_metrics.last_updated = datetime.now()

                await asyncio.sleep(1)  # Update metrics every second

            except Exception as e:
                logger.error("Error in metrics worker", error=str(e))
                await asyncio.sleep(5)

    async def _buffer_flush_worker(self):
        """Worker task for managing buffer flushes"""
        logger.info("Starting buffer flush worker")

        while self.is_running:
            try:
                if self.buffer.is_ready_for_flush():
                    await self._flush_buffer()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error("Error in buffer flush worker", error=str(e))
                await asyncio.sleep(10)

    async def _process_for_learning(self, event: TelemetryEvent):
        """Process event for learning engine integration"""
        # Generate learning-focused data
        learning_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'agent_id': event.agent_id,
            'test_id': event.test_id,
            'quality_score': event.quality_score,
            'processing_latency': event.processing_latency,
            'payload': event.payload,
            'context': {
                'data_quality': event.data_quality.value,
                'pipeline_health': self.streaming_metrics.pipeline_health_score,
                'throughput': self.streaming_metrics.events_per_second
            }
        }

        # Calculate reward based on event characteristics
        reward = self._calculate_learning_reward(event)
        learning_data['reward'] = reward
        learning_data['confidence'] = event.quality_score

        # Store in Redis for learning engine consumption
        if self.redis_client:
            await self.redis_client.lpush(
                'ptaas:learning:events',
                json.dumps(learning_data, default=str)
            )
            await self.redis_client.expire('ptaas:learning:events', 3600)  # 1 hour TTL

    def _calculate_learning_reward(self, event: TelemetryEvent) -> float:
        """Calculate learning reward for telemetry event"""
        base_reward = 0.5

        # Quality-based reward
        quality_reward = event.quality_score * 0.3

        # Event type importance
        type_rewards = {
            TelemetryEventType.VULNERABILITY_DETECTED: 1.0,
            TelemetryEventType.ANOMALY_DETECTED: 0.8,
            TelemetryEventType.PERFORMANCE_METRIC: 0.6,
            TelemetryEventType.FALSE_POSITIVE_FLAGGED: -0.3,
            TelemetryEventType.ERROR_OCCURRED: -0.2
        }
        type_reward = type_rewards.get(event.event_type, 0.0)

        # Timeliness reward (fresher data is better)
        age_seconds = (datetime.now() - event.timestamp.replace(tzinfo=None)).total_seconds()
        timeliness_reward = max(0, (300 - age_seconds) / 300) * 0.2  # 5-minute window

        total_reward = base_reward + quality_reward + type_reward + timeliness_reward
        return max(0.0, min(1.0, total_reward))  # Clamp to [0, 1]

    async def _persist_batch(self, batch: List[TelemetryEvent]):
        """Persist batch of telemetry events to PostgreSQL"""
        if not batch or not self.postgres_pool:
            return

        insert_sql = """
        INSERT INTO ptaas_telemetry
        (event_id, event_type, timestamp, agent_id, test_id, payload, metadata,
         quality_score, data_quality, processing_latency)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        async with self.postgres_pool.acquire() as conn:
            await conn.executemany(
                insert_sql,
                [
                    (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp,
                        event.agent_id,
                        event.test_id,
                        json.dumps(event.payload),
                        json.dumps(event.metadata),
                        event.quality_score,
                        event.data_quality.value,
                        event.processing_latency
                    )
                    for event in batch
                ]
            )

    async def _stream_event(self, event: TelemetryEvent):
        """Stream telemetry event to Kafka"""
        if not self.kafka_producer:
            return

        stream_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'agent_id': event.agent_id,
            'test_id': event.test_id,
            'payload': event.payload,
            'metadata': event.metadata,
            'quality_score': event.quality_score,
            'data_quality': event.data_quality.value
        }

        # Route to appropriate Kafka topic based on event type
        topic = f"ptaas-telemetry-{event.event_type.value.replace('_', '-')}"

        await self.kafka_producer.send(topic, value=stream_data)

    def _calculate_pipeline_health(self) -> float:
        """Calculate overall pipeline health score"""
        health_factors = []

        # Buffer health (not too full)
        buffer_ratio = self.buffer.size() / self.buffer.max_size
        buffer_health = 1.0 - min(1.0, buffer_ratio * 2)  # Penalize when >50% full
        health_factors.append(buffer_health)

        # Queue health
        queue_ratio = self.processing_queue.qsize() / self.processing_queue.maxsize
        queue_health = 1.0 - min(1.0, queue_ratio * 2)
        health_factors.append(queue_health)

        # Connection health
        connection_health = 1.0
        if not self.redis_client or not self.postgres_pool or not self.kafka_producer:
            connection_health = 0.5
        health_factors.append(connection_health)

        # Overall health is the minimum of all factors
        return min(health_factors) if health_factors else 0.0

    async def _flush_buffer(self):
        """Flush buffer contents to persistence and streaming"""
        logger.info("Flushing telemetry buffer")

        # Get all remaining events
        remaining_events = self.buffer.get_batch(batch_size=self.buffer.size())

        if remaining_events:
            # Persist to database
            await self._persist_batch(remaining_events)

            # Stream to Kafka
            for event in remaining_events:
                await self._stream_event(event)

        self.buffer.last_flush = time.time()
        logger.info(f"Flushed {len(remaining_events)} events from buffer")

    async def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics"""
        return {
            'streaming_metrics': asdict(self.streaming_metrics),
            'buffer_stats': {
                'current_size': self.buffer.size(),
                'max_size': self.buffer.max_size,
                'total_events': self.buffer.total_events,
                'utilization': self.buffer.size() / self.buffer.max_size
            },
            'queue_stats': {
                'current_size': self.processing_queue.qsize(),
                'max_size': self.processing_queue.maxsize,
                'utilization': self.processing_queue.qsize() / self.processing_queue.maxsize
            },
            'connection_status': {
                'redis': self.redis_client is not None,
                'postgres': self.postgres_pool is not None,
                'kafka': self.kafka_producer is not None
            },
            'worker_status': {
                'total_workers': len(self.worker_tasks),
                'running_workers': sum(1 for task in self.worker_tasks if not task.done()),
                'is_pipeline_running': self.is_running
            }
        }

# Global pipeline instance
telemetry_pipeline = TelemetryPipeline()

@app.on_event("startup")
async def startup_event():
    """Initialize telemetry pipeline on startup"""
    await telemetry_pipeline.initialize()
    await telemetry_pipeline.start_pipeline()

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown telemetry pipeline"""
    await telemetry_pipeline.stop_pipeline()

@app.get("/health")
async def health_check():
    """Comprehensive health check for telemetry pipeline"""
    pipeline_metrics = await telemetry_pipeline.get_pipeline_metrics()

    return {
        "status": "healthy" if pipeline_metrics['streaming_metrics']['pipeline_health_score'] > 0.7 else "degraded",
        "service": "ptaas_telemetry_pipeline",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "metrics": pipeline_metrics
    }

@app.post("/api/v1/telemetry/ingest")
async def ingest_telemetry(event_data: dict):
    """Ingest telemetry event into pipeline"""
    success, message = await telemetry_pipeline.ingest_event(event_data)

    if success:
        return {"status": "success", "message": message}
    else:
        raise HTTPException(status_code=400, detail=message)

@app.post("/api/v1/telemetry/ingest/batch")
async def ingest_telemetry_batch(events: List[dict]):
    """Ingest batch of telemetry events"""
    results = []

    for event_data in events:
        success, message = await telemetry_pipeline.ingest_event(event_data)
        results.append({
            "event_id": event_data.get("event_id", "unknown"),
            "success": success,
            "message": message
        })

    successful = sum(1 for r in results if r["success"])

    return {
        "total_events": len(events),
        "successful_ingests": successful,
        "failed_ingests": len(events) - successful,
        "results": results
    }

@app.get("/api/v1/telemetry/metrics")
async def get_telemetry_metrics():
    """Get comprehensive telemetry pipeline metrics"""
    return await telemetry_pipeline.get_pipeline_metrics()

@app.get("/api/v1/telemetry/stream")
async def telemetry_stream():
    """Stream telemetry metrics in real-time"""
    async def generate_stream():
        while True:
            metrics = await telemetry_pipeline.get_pipeline_metrics()
            yield f"data: {json.dumps(metrics)}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.websocket("/ws/telemetry/realtime")
async def telemetry_websocket(websocket: WebSocket):
    """Real-time WebSocket endpoint for telemetry data"""
    await websocket.accept()

    try:
        while True:
            metrics = await telemetry_pipeline.get_pipeline_metrics()
            await websocket.send_json({
                "type": "telemetry_metrics",
                "data": metrics,
                "timestamp": datetime.now().isoformat()
            })

            await asyncio.sleep(2)  # Send updates every 2 seconds

    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "ptaas_telemetry_pipeline:app",
        host="0.0.0.0",
        port=8087,
        reload=False,
        log_level="info"
    )
