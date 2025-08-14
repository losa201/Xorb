"""
Telemetry Collector for Red/Blue Agent Framework

Collects, processes, and stores telemetry data from agents, missions, and system components.
Provides real-time data streaming and batch processing capabilities.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

import redis.asyncio as redis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

logger = logging.getLogger(__name__)


class TelemetryEventType(Enum):
    """Types of telemetry events"""
    MISSION_STARTED = "mission_started"
    MISSION_COMPLETED = "mission_completed"
    MISSION_FAILED = "mission_failed"
    AGENT_ASSIGNED = "agent_assigned"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TECHNIQUE_EXECUTED = "technique_executed"
    SANDBOX_CREATED = "sandbox_created"
    SANDBOX_DESTROYED = "sandbox_destroyed"
    DETECTION_EVENT = "detection_event"
    RESPONSE_ACTION = "response_action"
    SYSTEM_METRIC = "system_metric"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"


@dataclass
class TelemetryEvent:
    """Base telemetry event structure"""
    event_id: str
    event_type: TelemetryEventType
    timestamp: datetime
    source: str  # agent_id, mission_id, system, etc.
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []


@dataclass
class AgentTelemetry:
    """Agent-specific telemetry data"""
    agent_id: str
    agent_type: str
    mission_id: str
    event_type: str
    technique_id: Optional[str] = None
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    network_activity: Optional[Dict[str, Any]] = None
    security_events: Optional[List[Dict[str, Any]]] = None


@dataclass
class MissionTelemetry:
    """Mission-specific telemetry data"""
    mission_id: str
    mission_name: str
    event_type: str
    agent_count: int = 0
    completion_percentage: float = 0.0
    vulnerabilities_found: int = 0
    detections_triggered: int = 0
    response_actions: int = 0
    total_execution_time: float = 0.0
    success_rate: float = 0.0


@dataclass
class SystemTelemetry:
    """System-level telemetry data"""
    component: str
    event_type: str
    cpu_usage: float = 0.0
    memory_usage_mb: int = 0
    disk_usage_mb: int = 0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    active_connections: int = 0
    queue_depth: int = 0
    error_rate: float = 0.0


class TelemetryCollector:
    """
    Central telemetry collection service for the Red/Blue Agent Framework.
    
    Features:
    - Real-time event collection from multiple sources
    - Hot and cold data storage (Redis + PostgreSQL)
    - Batch processing for high-volume data
    - Prometheus metrics integration
    - Data retention and archival policies
    - Privacy and compliance controls
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None,
                 postgres_pool: Optional[asyncpg.Pool] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.redis_client = redis_client
        self.postgres_pool = postgres_pool
        self.config = config or {}
        
        # Configuration
        self.batch_size = self.config.get("batch_size", 1000)
        self.flush_interval = self.config.get("flush_interval", 60)  # seconds
        self.hot_data_ttl = self.config.get("hot_data_ttl", 3600)  # 1 hour
        self.sampling_rate = self.config.get("sampling_rate", 1.0)  # 0.0-1.0
        
        # Internal state
        self.event_buffer: List[TelemetryEvent] = []
        self.buffer_lock = asyncio.Lock()
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.registry = CollectorRegistry()
        self.events_total = Counter(
            'telemetry_events_total',
            'Total number of telemetry events collected',
            ['event_type', 'source'],
            registry=self.registry
        )
        self.processing_duration = Histogram(
            'telemetry_processing_duration_seconds',
            'Time spent processing telemetry events',
            ['event_type'],
            registry=self.registry
        )
        self.buffer_size = Gauge(
            'telemetry_buffer_size',
            'Current size of telemetry event buffer',
            registry=self.registry
        )
        self.storage_errors = Counter(
            'telemetry_storage_errors_total',
            'Total number of storage errors',
            ['storage_type'],
            registry=self.registry
        )
        
    async def initialize(self):
        """Initialize the telemetry collector"""
        logger.info("Initializing Telemetry Collector...")
        
        # Initialize database schema if needed
        if self.postgres_pool:
            await self._initialize_database_schema()
            
        # Start background processing tasks
        self.is_running = True
        
        # Buffer flush task
        flush_task = asyncio.create_task(self._flush_buffer_periodically())
        self.background_tasks.append(flush_task)
        
        # Metrics update task
        metrics_task = asyncio.create_task(self._update_metrics_periodically())
        self.background_tasks.append(metrics_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_old_data_periodically())
        self.background_tasks.append(cleanup_task)
        
        logger.info("Telemetry Collector initialized successfully")
        
    async def _initialize_database_schema(self):
        """Initialize PostgreSQL schema for telemetry data"""
        schema_sql = """
        -- Telemetry events table (partitioned by date)
        CREATE TABLE IF NOT EXISTS telemetry_events (
            event_id UUID PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            source VARCHAR(100) NOT NULL,
            data JSONB NOT NULL,
            metadata JSONB DEFAULT '{}',
            tags TEXT[] DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        ) PARTITION BY RANGE (timestamp);
        
        -- Agent telemetry table
        CREATE TABLE IF NOT EXISTS agent_telemetry (
            id SERIAL PRIMARY KEY,
            agent_id VARCHAR(100) NOT NULL,
            agent_type VARCHAR(50) NOT NULL,
            mission_id UUID NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            technique_id VARCHAR(100),
            execution_time FLOAT DEFAULT 0.0,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            resource_usage JSONB,
            network_activity JSONB,
            security_events JSONB DEFAULT '[]',
            timestamp TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Mission telemetry table
        CREATE TABLE IF NOT EXISTS mission_telemetry (
            id SERIAL PRIMARY KEY,
            mission_id UUID NOT NULL,
            mission_name VARCHAR(200) NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            agent_count INTEGER DEFAULT 0,
            completion_percentage FLOAT DEFAULT 0.0,
            vulnerabilities_found INTEGER DEFAULT 0,
            detections_triggered INTEGER DEFAULT 0,
            response_actions INTEGER DEFAULT 0,
            total_execution_time FLOAT DEFAULT 0.0,
            success_rate FLOAT DEFAULT 0.0,
            timestamp TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- System telemetry table
        CREATE TABLE IF NOT EXISTS system_telemetry (
            id SERIAL PRIMARY KEY,
            component VARCHAR(100) NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            cpu_usage FLOAT DEFAULT 0.0,
            memory_usage_mb INTEGER DEFAULT 0,
            disk_usage_mb INTEGER DEFAULT 0,
            network_rx_bytes BIGINT DEFAULT 0,
            network_tx_bytes BIGINT DEFAULT 0,
            active_connections INTEGER DEFAULT 0,
            queue_depth INTEGER DEFAULT 0,
            error_rate FLOAT DEFAULT 0.0,
            timestamp TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_telemetry_events_timestamp ON telemetry_events (timestamp);
        CREATE INDEX IF NOT EXISTS idx_telemetry_events_type ON telemetry_events (event_type);
        CREATE INDEX IF NOT EXISTS idx_telemetry_events_source ON telemetry_events (source);
        CREATE INDEX IF NOT EXISTS idx_agent_telemetry_mission ON agent_telemetry (mission_id);
        CREATE INDEX IF NOT EXISTS idx_agent_telemetry_timestamp ON agent_telemetry (timestamp);
        CREATE INDEX IF NOT EXISTS idx_mission_telemetry_timestamp ON mission_telemetry (timestamp);
        CREATE INDEX IF NOT EXISTS idx_system_telemetry_timestamp ON system_telemetry (timestamp);
        
        -- Create current month partition if it doesn't exist
        DO $$
        DECLARE
            partition_date DATE := DATE_TRUNC('month', CURRENT_DATE);
            partition_name TEXT := 'telemetry_events_' || TO_CHAR(partition_date, 'YYYY_MM');
            next_month DATE := partition_date + INTERVAL '1 month';
        BEGIN
            EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF telemetry_events 
                           FOR VALUES FROM (%L) TO (%L)', 
                          partition_name, partition_date, next_month);
        END $$;
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)
            
        logger.info("Database schema initialized")
        
    async def collect_event(self, event: TelemetryEvent):
        """Collect a single telemetry event"""
        # Apply sampling if configured
        if self.sampling_rate < 1.0:
            import random
            if random.random() > self.sampling_rate:
                return
                
        # Update metrics
        self.events_total.labels(
            event_type=event.event_type.value,
            source=event.source
        ).inc()
        
        # Add to buffer
        async with self.buffer_lock:
            self.event_buffer.append(event)
            self.buffer_size.set(len(self.event_buffer))
            
        # Flush buffer if it's full
        if len(self.event_buffer) >= self.batch_size:
            await self._flush_buffer()
            
    async def collect_agent_event(self, agent_id: str, event_type: str, data: Dict[str, Any]):
        """Collect agent-specific telemetry event"""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=TelemetryEventType(event_type),
            timestamp=datetime.utcnow(),
            source=agent_id,
            data=data,
            tags=["agent", data.get("agent_type", "unknown")]
        )
        
        await self.collect_event(event)
        
        # Store in hot cache for real-time access
        if self.redis_client:
            try:
                cache_key = f"agent_telemetry:{agent_id}:latest"
                await self.redis_client.setex(
                    cache_key,
                    self.hot_data_ttl,
                    json.dumps(asdict(event), default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to cache agent telemetry: {e}")
                
    async def collect_mission_event(self, mission_id: str, event_type: str, data: Dict[str, Any]):
        """Collect mission-specific telemetry event"""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=TelemetryEventType(event_type),
            timestamp=datetime.utcnow(),
            source=mission_id,
            data=data,
            tags=["mission"]
        )
        
        await self.collect_event(event)
        
        # Store in hot cache
        if self.redis_client:
            try:
                cache_key = f"mission_telemetry:{mission_id}:latest"
                await self.redis_client.setex(
                    cache_key,
                    self.hot_data_ttl,
                    json.dumps(asdict(event), default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to cache mission telemetry: {e}")
                
    async def collect_system_event(self, component: str, event_type: str, data: Dict[str, Any]):
        """Collect system-level telemetry event"""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=TelemetryEventType(event_type),
            timestamp=datetime.utcnow(),
            source=component,
            data=data,
            tags=["system"]
        )
        
        await self.collect_event(event)
        
    async def collect_technique_execution(self, agent_id: str, technique_id: str, 
                                        parameters: Dict[str, Any], result: Dict[str, Any]):
        """Collect technique execution telemetry"""
        data = {
            "agent_id": agent_id,
            "technique_id": technique_id,
            "parameters": parameters,
            "result": result,
            "execution_time": result.get("execution_time", 0.0),
            "success": result.get("success", False),
            "error": result.get("error")
        }
        
        await self.collect_agent_event(agent_id, "technique_executed", data)
        
    async def get_agent_metrics(self, agent_id: str, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get aggregated metrics for an agent"""
        if not self.postgres_pool:
            return {}
            
        time_range = time_range or timedelta(hours=24)
        start_time = datetime.utcnow() - time_range
        
        query = """
        SELECT 
            agent_type,
            COUNT(*) as total_events,
            COUNT(*) FILTER (WHERE success = true) as successful_events,
            COUNT(*) FILTER (WHERE success = false) as failed_events,
            AVG(execution_time) as avg_execution_time,
            MAX(execution_time) as max_execution_time,
            COUNT(DISTINCT technique_id) as unique_techniques
        FROM agent_telemetry 
        WHERE agent_id = $1 AND timestamp >= $2
        GROUP BY agent_type
        """
        
        try:
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(query, agent_id, start_time)
                
                if row:
                    return {
                        "agent_type": row["agent_type"],
                        "total_events": row["total_events"],
                        "successful_events": row["successful_events"],
                        "failed_events": row["failed_events"],
                        "success_rate": (row["successful_events"] / row["total_events"]) * 100 if row["total_events"] > 0 else 0,
                        "avg_execution_time": float(row["avg_execution_time"] or 0),
                        "max_execution_time": float(row["max_execution_time"] or 0),
                        "unique_techniques": row["unique_techniques"]
                    }
        except Exception as e:
            logger.error(f"Failed to get agent metrics: {e}")
            
        return {}
        
    async def get_mission_metrics(self, mission_id: str) -> Dict[str, Any]:
        """Get aggregated metrics for a mission"""
        if not self.postgres_pool:
            return {}
            
        query = """
        SELECT 
            mission_name,
            MAX(agent_count) as max_agents,
            MAX(completion_percentage) as completion_percentage,
            MAX(vulnerabilities_found) as vulnerabilities_found,
            MAX(detections_triggered) as detections_triggered,
            MAX(response_actions) as response_actions,
            MAX(total_execution_time) as total_execution_time,
            MAX(success_rate) as success_rate
        FROM mission_telemetry 
        WHERE mission_id = $1
        GROUP BY mission_name
        """
        
        try:
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(query, mission_id)
                
                if row:
                    return {
                        "mission_name": row["mission_name"],
                        "max_agents": row["max_agents"],
                        "completion_percentage": float(row["completion_percentage"] or 0),
                        "vulnerabilities_found": row["vulnerabilities_found"],
                        "detections_triggered": row["detections_triggered"],
                        "response_actions": row["response_actions"],
                        "total_execution_time": float(row["total_execution_time"] or 0),
                        "success_rate": float(row["success_rate"] or 0)
                    }
        except Exception as e:
            logger.error(f"Failed to get mission metrics: {e}")
            
        return {}
        
    async def get_system_metrics(self, component: str, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get system metrics for a component"""
        if not self.postgres_pool:
            return {}
            
        time_range = time_range or timedelta(hours=1)
        start_time = datetime.utcnow() - time_range
        
        query = """
        SELECT 
            AVG(cpu_usage) as avg_cpu_usage,
            MAX(cpu_usage) as max_cpu_usage,
            AVG(memory_usage_mb) as avg_memory_usage,
            MAX(memory_usage_mb) as max_memory_usage,
            AVG(active_connections) as avg_connections,
            MAX(active_connections) as max_connections,
            AVG(error_rate) as avg_error_rate,
            MAX(error_rate) as max_error_rate
        FROM system_telemetry 
        WHERE component = $1 AND timestamp >= $2
        """
        
        try:
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(query, component, start_time)
                
                if row:
                    return {
                        "avg_cpu_usage": float(row["avg_cpu_usage"] or 0),
                        "max_cpu_usage": float(row["max_cpu_usage"] or 0),
                        "avg_memory_usage": int(row["avg_memory_usage"] or 0),
                        "max_memory_usage": int(row["max_memory_usage"] or 0),
                        "avg_connections": int(row["avg_connections"] or 0),
                        "max_connections": int(row["max_connections"] or 0),
                        "avg_error_rate": float(row["avg_error_rate"] or 0),
                        "max_error_rate": float(row["max_error_rate"] or 0)
                    }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            
        return {}
        
    async def _flush_buffer(self):
        """Flush the event buffer to storage"""
        if not self.event_buffer:
            return
            
        start_time = time.time()
        
        async with self.buffer_lock:
            events_to_flush = self.event_buffer.copy()
            self.event_buffer.clear()
            self.buffer_size.set(0)
            
        logger.debug(f"Flushing {len(events_to_flush)} telemetry events")
        
        # Store in PostgreSQL (cold storage)
        if self.postgres_pool:
            await self._store_events_postgres(events_to_flush)
            
        # Store in Redis (hot storage)
        if self.redis_client:
            await self._store_events_redis(events_to_flush)
            
        processing_time = time.time() - start_time
        logger.debug(f"Flushed {len(events_to_flush)} events in {processing_time:.2f}s")
        
    async def _store_events_postgres(self, events: List[TelemetryEvent]):
        """Store events in PostgreSQL"""
        if not events:
            return
            
        try:
            async with self.postgres_pool.acquire() as conn:
                # Prepare batch insert data
                insert_data = []
                for event in events:
                    insert_data.append((
                        event.event_id,
                        event.event_type.value,
                        event.timestamp,
                        event.source,
                        json.dumps(event.data),
                        json.dumps(event.metadata),
                        event.tags
                    ))
                    
                # Batch insert
                await conn.executemany(
                    """INSERT INTO telemetry_events 
                       (event_id, event_type, timestamp, source, data, metadata, tags)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    insert_data
                )
                
                # Store specialized telemetry data
                await self._store_specialized_telemetry(conn, events)
                
        except Exception as e:
            self.storage_errors.labels(storage_type="postgres").inc()
            logger.error(f"Failed to store events in PostgreSQL: {e}")
            
    async def _store_specialized_telemetry(self, conn: asyncpg.Connection, events: List[TelemetryEvent]):
        """Store specialized telemetry data in dedicated tables"""
        agent_events = []
        mission_events = []
        system_events = []
        
        for event in events:
            if event.source.startswith("agent_"):
                agent_events.append(event)
            elif event.source.startswith("mission_"):
                mission_events.append(event)
            elif "system" in event.tags:
                system_events.append(event)
                
        # Store agent telemetry
        if agent_events:
            agent_data = []
            for event in agent_events:
                data = event.data
                agent_data.append((
                    data.get("agent_id", event.source),
                    data.get("agent_type", "unknown"),
                    data.get("mission_id"),
                    event.event_type.value,
                    data.get("technique_id"),
                    data.get("execution_time", 0.0),
                    data.get("success", True),
                    data.get("error"),
                    json.dumps(data.get("resource_usage", {})),
                    json.dumps(data.get("network_activity", {})),
                    json.dumps(data.get("security_events", [])),
                    event.timestamp
                ))
                
            await conn.executemany(
                """INSERT INTO agent_telemetry 
                   (agent_id, agent_type, mission_id, event_type, technique_id, 
                    execution_time, success, error_message, resource_usage, 
                    network_activity, security_events, timestamp)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
                agent_data
            )
            
        # Store mission telemetry
        if mission_events:
            mission_data = []
            for event in mission_events:
                data = event.data
                mission_data.append((
                    data.get("mission_id", event.source),
                    data.get("mission_name", "unknown"),
                    event.event_type.value,
                    data.get("agent_count", 0),
                    data.get("completion_percentage", 0.0),
                    data.get("vulnerabilities_found", 0),
                    data.get("detections_triggered", 0),
                    data.get("response_actions", 0),
                    data.get("total_execution_time", 0.0),
                    data.get("success_rate", 0.0),
                    event.timestamp
                ))
                
            await conn.executemany(
                """INSERT INTO mission_telemetry 
                   (mission_id, mission_name, event_type, agent_count, completion_percentage,
                    vulnerabilities_found, detections_triggered, response_actions,
                    total_execution_time, success_rate, timestamp)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                mission_data
            )
            
        # Store system telemetry
        if system_events:
            system_data = []
            for event in system_events:
                data = event.data
                system_data.append((
                    event.source,
                    event.event_type.value,
                    data.get("cpu_usage", 0.0),
                    data.get("memory_usage_mb", 0),
                    data.get("disk_usage_mb", 0),
                    data.get("network_rx_bytes", 0),
                    data.get("network_tx_bytes", 0),
                    data.get("active_connections", 0),
                    data.get("queue_depth", 0),
                    data.get("error_rate", 0.0),
                    event.timestamp
                ))
                
            await conn.executemany(
                """INSERT INTO system_telemetry 
                   (component, event_type, cpu_usage, memory_usage_mb, disk_usage_mb,
                    network_rx_bytes, network_tx_bytes, active_connections, queue_depth,
                    error_rate, timestamp)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                system_data
            )
            
    async def _store_events_redis(self, events: List[TelemetryEvent]):
        """Store events in Redis for hot data access"""
        if not events:
            return
            
        try:
            pipe = self.redis_client.pipeline()
            
            for event in events:
                # Store individual event
                event_key = f"telemetry:event:{event.event_id}"
                event_data = json.dumps(asdict(event), default=str)
                pipe.setex(event_key, self.hot_data_ttl, event_data)
                
                # Add to time-series list for the event type
                series_key = f"telemetry:series:{event.event_type.value}"
                pipe.lpush(series_key, event_data)
                pipe.ltrim(series_key, 0, 999)  # Keep last 1000 events
                pipe.expire(series_key, self.hot_data_ttl)
                
                # Add to source-specific list
                source_key = f"telemetry:source:{event.source}"
                pipe.lpush(source_key, event_data)
                pipe.ltrim(source_key, 0, 99)  # Keep last 100 events per source
                pipe.expire(source_key, self.hot_data_ttl)
                
            await pipe.execute()
            
        except Exception as e:
            self.storage_errors.labels(storage_type="redis").inc()
            logger.error(f"Failed to store events in Redis: {e}")
            
    async def _flush_buffer_periodically(self):
        """Periodically flush the buffer"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in buffer flush task: {e}")
                await asyncio.sleep(5)
                
    async def _update_metrics_periodically(self):
        """Periodically update metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Update buffer size metric
                async with self.buffer_lock:
                    self.buffer_size.set(len(self.event_buffer))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics update task: {e}")
                await asyncio.sleep(5)
                
    async def _cleanup_old_data_periodically(self):
        """Periodically cleanup old data"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_old_data(self):
        """Cleanup old telemetry data based on retention policies"""
        if not self.postgres_pool:
            return
            
        # Default retention: 30 days for detailed data, 1 year for aggregated data
        retention_days = self.config.get("retention_days", 30)
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        try:
            async with self.postgres_pool.acquire() as conn:
                # Delete old detailed telemetry
                deleted = await conn.execute(
                    "DELETE FROM telemetry_events WHERE timestamp < $1",
                    cutoff_date
                )
                
                if deleted:
                    logger.info(f"Cleaned up old telemetry data: {deleted}")
                    
                # Also cleanup specialized tables
                await conn.execute(
                    "DELETE FROM agent_telemetry WHERE timestamp < $1",
                    cutoff_date
                )
                await conn.execute(
                    "DELETE FROM mission_telemetry WHERE timestamp < $1", 
                    cutoff_date
                )
                await conn.execute(
                    "DELETE FROM system_telemetry WHERE timestamp < $1",
                    cutoff_date
                )
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            
    async def shutdown(self):
        """Shutdown the telemetry collector"""
        logger.info("Shutting down Telemetry Collector...")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        try:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error during background task shutdown: {e}")
            
        # Final buffer flush
        await self._flush_buffer()
        
        logger.info("Telemetry Collector shutdown complete")