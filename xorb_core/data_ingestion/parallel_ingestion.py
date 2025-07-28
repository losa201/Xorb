#!/usr/bin/env python3
"""
Parallel Data Ingestion Engine for XORB
High-performance streaming data ingestion with queue prioritization
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Union
import logging
from collections import defaultdict, deque
import hashlib

import nats
import redis.asyncio as redis
from nats.js import JetStreamContext
import aiohttp

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """Types of data sources for ingestion"""
    VULNERABILITY_FEED = "vulnerability_feed"
    THREAT_INTELLIGENCE = "threat_intelligence"
    AGENT_TELEMETRY = "agent_telemetry"
    CAMPAIGN_RESULTS = "campaign_results"
    SECURITY_EVENTS = "security_events"
    NETWORK_TRAFFIC = "network_traffic"
    LOG_STREAMS = "log_streams"
    API_RESPONSES = "api_responses"


class IngestionPriority(str, Enum):
    """Data ingestion priority levels"""
    CRITICAL = "critical"      # Immediate processing required
    HIGH = "high"             # Process within 1 second
    MEDIUM = "medium"         # Process within 5 seconds
    LOW = "low"              # Process within 30 seconds
    BACKGROUND = "background" # Process when resources available


class ProcessingStage(str, Enum):
    """Data processing pipeline stages"""
    RAW_INGESTION = "raw_ingestion"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    ENRICHMENT = "enrichment"
    ANALYSIS = "analysis"
    STORAGE = "storage"
    NOTIFICATION = "notification"


@dataclass
class DataPacket:
    """Individual data packet for ingestion"""
    packet_id: str
    source_type: DataSourceType
    priority: IngestionPriority
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_stage: ProcessingStage = ProcessingStage.RAW_INGESTION
    retry_count: int = 0
    error_history: List[str] = field(default_factory=list)
    size_bytes: int = 0
    checksum: Optional[str] = None


@dataclass
class IngestionMetrics:
    """Metrics tracking for data ingestion"""
    total_packets: int = 0
    packets_per_priority: Dict[IngestionPriority, int] = field(default_factory=lambda: defaultdict(int))
    packets_per_source: Dict[DataSourceType, int] = field(default_factory=lambda: defaultdict(int))
    processing_times: Dict[ProcessingStage, List[float]] = field(default_factory=lambda: defaultdict(list))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=100))
    backlog_size: int = 0
    
    
class PriorityQueue:
    """High-performance priority queue with multiple priority levels"""
    
    def __init__(self, max_size: int = 100000):
        self.queues = {priority: deque() for priority in IngestionPriority}
        self.max_size = max_size
        self.current_size = 0
        self.priority_weights = {
            IngestionPriority.CRITICAL: 1.0,
            IngestionPriority.HIGH: 0.8,
            IngestionPriority.MEDIUM: 0.6,
            IngestionPriority.LOW: 0.4,
            IngestionPriority.BACKGROUND: 0.2
        }
        
    async def put(self, packet: DataPacket) -> bool:
        """Add packet to priority queue"""
        if self.current_size >= self.max_size:
            # Apply backpressure - drop lowest priority packets if needed
            if packet.priority == IngestionPriority.CRITICAL:
                await self._drop_lowest_priority()
            else:
                return False
                
        self.queues[packet.priority].append(packet)
        self.current_size += 1
        return True
        
    async def get(self) -> Optional[DataPacket]:
        """Get next packet based on priority"""
        # Process in priority order with some fairness
        for priority in IngestionPriority:
            if self.queues[priority]:
                packet = self.queues[priority].popleft()
                self.current_size -= 1
                return packet
        return None
        
    async def get_batch(self, max_batch_size: int = 10) -> List[DataPacket]:
        """Get batch of packets for efficient processing"""
        batch = []
        
        # Weighted round-robin selection
        for priority in IngestionPriority:
            weight = self.priority_weights[priority]
            batch_size = int(max_batch_size * weight)
            
            for _ in range(min(batch_size, len(self.queues[priority]))):
                if len(batch) >= max_batch_size:
                    break
                packet = self.queues[priority].popleft()
                self.current_size -= 1
                batch.append(packet)
                
        return batch
        
    def size(self) -> int:
        """Get current queue size"""
        return self.current_size
        
    def size_by_priority(self) -> Dict[IngestionPriority, int]:
        """Get queue sizes by priority"""
        return {priority: len(queue) for priority, queue in self.queues.items()}
        
    async def _drop_lowest_priority(self):
        """Drop lowest priority packet to make room"""
        for priority in reversed(list(IngestionPriority)):
            if self.queues[priority]:
                self.queues[priority].popleft()
                self.current_size -= 1
                logger.warning(f"Dropped {priority} packet due to queue full")
                return


class DataProcessor:
    """Parallel data processor with stage-based pipeline"""
    
    def __init__(self, stage: ProcessingStage, worker_count: int = 4):
        self.stage = stage
        self.worker_count = worker_count
        self.workers: List[asyncio.Task] = []
        self.input_queue = asyncio.Queue(maxsize=1000)
        self.output_queue = asyncio.Queue(maxsize=1000)
        self.metrics = {
            "processed": 0,
            "errors": 0,
            "avg_processing_time": 0.0
        }
        self.running = False
        
    async def start(self):
        """Start processor workers"""
        self.running = True
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            
    async def stop(self):
        """Stop processor workers"""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        
    async def process(self, packet: DataPacket) -> Optional[DataPacket]:
        """Add packet for processing"""
        try:
            await self.input_queue.put(packet)
            return await self.output_queue.get()
        except asyncio.QueueFull:
            logger.warning(f"Processing queue full for stage {self.stage}")
            return None
            
    async def _worker(self, worker_id: str):
        """Individual worker for processing packets"""
        logger.info(f"Started {self.stage} worker: {worker_id}")
        
        while self.running:
            try:
                # Get packet with timeout
                packet = await asyncio.wait_for(
                    self.input_queue.get(), timeout=1.0
                )
                
                # Process packet
                start_time = time.time()
                processed_packet = await self._process_packet(packet)
                processing_time = time.time() - start_time
                
                # Update metrics
                self.metrics["processed"] += 1
                self.metrics["avg_processing_time"] = (
                    (self.metrics["avg_processing_time"] * (self.metrics["processed"] - 1) + processing_time) /
                    self.metrics["processed"]
                )
                
                # Output processed packet
                if processed_packet:
                    await self.output_queue.put(processed_packet)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics["errors"] += 1
                logger.error(f"Worker {worker_id} error: {e}")
                
    async def _process_packet(self, packet: DataPacket) -> Optional[DataPacket]:
        """Process individual packet based on stage"""
        
        try:
            if self.stage == ProcessingStage.VALIDATION:
                return await self._validate_packet(packet)
            elif self.stage == ProcessingStage.NORMALIZATION:
                return await self._normalize_packet(packet)
            elif self.stage == ProcessingStage.ENRICHMENT:
                return await self._enrich_packet(packet)
            elif self.stage == ProcessingStage.ANALYSIS:
                return await self._analyze_packet(packet)
            elif self.stage == ProcessingStage.STORAGE:
                return await self._store_packet(packet)
            else:
                # Pass through unknown stages
                return packet
                
        except Exception as e:
            packet.error_history.append(f"{self.stage}: {str(e)}")
            packet.retry_count += 1
            logger.error(f"Processing error in {self.stage}: {e}")
            return packet
            
    async def _validate_packet(self, packet: DataPacket) -> DataPacket:
        """Validate packet data integrity and format"""
        
        # Calculate checksum if not present
        if not packet.checksum:
            data_str = json.dumps(packet.data, sort_keys=True) if isinstance(packet.data, dict) else str(packet.data)
            packet.checksum = hashlib.sha256(data_str.encode()).hexdigest()
            
        # Validate required fields
        if not packet.packet_id or not packet.source_type:
            raise ValueError("Missing required packet fields")
            
        # Calculate size
        if not packet.size_bytes:
            packet.size_bytes = len(json.dumps(packet.data).encode())
            
        packet.processing_stage = ProcessingStage.NORMALIZATION
        return packet
        
    async def _normalize_packet(self, packet: DataPacket) -> DataPacket:
        """Normalize packet data to standard format"""
        
        # Source-specific normalization
        if packet.source_type == DataSourceType.VULNERABILITY_FEED:
            packet.data = await self._normalize_vulnerability_data(packet.data)
        elif packet.source_type == DataSourceType.THREAT_INTELLIGENCE:
            packet.data = await self._normalize_threat_data(packet.data)
        elif packet.source_type == DataSourceType.AGENT_TELEMETRY:
            packet.data = await self._normalize_telemetry_data(packet.data)
            
        packet.processing_stage = ProcessingStage.ENRICHMENT
        return packet
        
    async def _enrich_packet(self, packet: DataPacket) -> DataPacket:
        """Enrich packet with additional context and metadata"""
        
        # Add timestamp if missing
        if 'timestamp' not in packet.metadata:
            packet.metadata['timestamp'] = packet.timestamp.isoformat()
            
        # Add source information
        packet.metadata['source_type'] = packet.source_type.value
        packet.metadata['priority'] = packet.priority.value
        packet.metadata['processing_time'] = datetime.utcnow().isoformat()
        
        # Source-specific enrichment
        if packet.source_type == DataSourceType.SECURITY_EVENTS:
            await self._enrich_security_event(packet)
        elif packet.source_type == DataSourceType.NETWORK_TRAFFIC:
            await self._enrich_network_data(packet)
            
        packet.processing_stage = ProcessingStage.ANALYSIS
        return packet
        
    async def _analyze_packet(self, packet: DataPacket) -> DataPacket:
        """Analyze packet for patterns and anomalies"""
        
        # Basic analysis
        packet.metadata['analysis'] = {
            'packet_size': packet.size_bytes,
            'processing_latency': (datetime.utcnow() - packet.timestamp).total_seconds(),
            'retry_count': packet.retry_count
        }
        
        # Source-specific analysis
        if packet.source_type == DataSourceType.VULNERABILITY_FEED:
            packet.metadata['analysis']['severity'] = self._analyze_vulnerability_severity(packet.data)
        elif packet.source_type == DataSourceType.THREAT_INTELLIGENCE:
            packet.metadata['analysis']['threat_level'] = self._analyze_threat_level(packet.data)
            
        packet.processing_stage = ProcessingStage.STORAGE
        return packet
        
    async def _store_packet(self, packet: DataPacket) -> DataPacket:
        """Store processed packet to appropriate storage backend"""
        
        # Simulate storage operation
        storage_key = f"{packet.source_type.value}:{packet.packet_id}"
        packet.metadata['storage_key'] = storage_key
        packet.metadata['stored_at'] = datetime.utcnow().isoformat()
        
        # Mark as stored
        packet.processing_stage = ProcessingStage.NOTIFICATION
        return packet
        
    async def _normalize_vulnerability_data(self, data: Any) -> Dict[str, Any]:
        """Normalize vulnerability feed data"""
        if isinstance(data, dict):
            return {
                'cve_id': data.get('cve_id', 'unknown'),
                'severity': data.get('severity', 'unknown'),
                'description': data.get('description', ''),
                'affected_systems': data.get('affected_systems', []),
                'mitigation': data.get('mitigation', ''),
                'references': data.get('references', [])
            }
        return {'raw_data': data}
        
    async def _normalize_threat_data(self, data: Any) -> Dict[str, Any]:
        """Normalize threat intelligence data"""
        if isinstance(data, dict):
            return {
                'threat_type': data.get('type', 'unknown'),
                'indicators': data.get('indicators', []),
                'confidence': data.get('confidence', 0.5),
                'source': data.get('source', 'unknown'),
                'campaign': data.get('campaign', ''),
                'ttps': data.get('ttps', [])
            }
        return {'raw_data': data}
        
    async def _normalize_telemetry_data(self, data: Any) -> Dict[str, Any]:
        """Normalize agent telemetry data"""
        if isinstance(data, dict):
            return {
                'agent_id': data.get('agent_id', 'unknown'),
                'metrics': data.get('metrics', {}),
                'status': data.get('status', 'unknown'),
                'performance': data.get('performance', {}),
                'errors': data.get('errors', [])
            }
        return {'raw_data': data}
        
    async def _enrich_security_event(self, packet: DataPacket):
        """Enrich security event with threat intelligence"""
        # Simulate threat intelligence lookup
        packet.metadata['threat_intel'] = {
            'known_indicators': [],
            'risk_score': 0.3,
            'recommendations': []
        }
        
    async def _enrich_network_data(self, packet: DataPacket):
        """Enrich network traffic data with geolocation and reputation"""
        # Simulate geolocation and reputation lookup
        packet.metadata['network_intel'] = {
            'source_country': 'unknown',
            'reputation_score': 0.5,
            'is_malicious': False
        }
        
    def _analyze_vulnerability_severity(self, data: Dict[str, Any]) -> str:
        """Analyze vulnerability severity"""
        severity = data.get('severity', '').lower()
        if severity in ['critical', 'high']:
            return 'high_risk'
        elif severity in ['medium']:
            return 'medium_risk'
        else:
            return 'low_risk'
            
    def _analyze_threat_level(self, data: Dict[str, Any]) -> str:
        """Analyze threat level"""
        confidence = data.get('confidence', 0.0)
        if confidence > 0.8:
            return 'high_confidence'
        elif confidence > 0.5:
            return 'medium_confidence'
        else:
            return 'low_confidence'


class ParallelIngestionEngine:
    """Main parallel data ingestion engine"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 nats_url: str = "nats://localhost:4222",
                 max_workers: int = 32):
        
        self.redis_client = redis.from_url(redis_url)
        self.nats_url = nats_url
        self.nats_client: Optional[nats.NATS] = None
        self.jetstream: Optional[JetStreamContext] = None
        
        # Core components
        self.priority_queue = PriorityQueue(max_size=1000000)  # 1M packet buffer
        self.processors: Dict[ProcessingStage, DataProcessor] = {}
        self.metrics = IngestionMetrics()
        
        # Configuration
        self.max_workers = max_workers
        self.batch_size = 100
        self.processing_interval = 0.1  # 100ms
        self.backpressure_threshold = 0.8  # 80% queue full
        
        # Worker management
        self.ingestion_workers: List[asyncio.Task] = []
        self.processing_workers: List[asyncio.Task] = []
        self.running = False
        
        # Data sources
        self.data_sources: Dict[str, Callable] = {}
        self.source_streams: Dict[str, asyncio.Task] = {}
        
    async def start(self):
        """Start the ingestion engine"""
        logger.info("Starting parallel data ingestion engine")
        
        # Connect to NATS
        try:
            self.nats_client = await nats.connect(self.nats_url)
            self.jetstream = self.nats_client.jetstream()
            logger.info("Connected to NATS JetStream")
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            
        # Initialize processors
        await self._initialize_processors()
        
        # Start ingestion workers
        self.running = True
        for i in range(min(8, self.max_workers // 4)):  # 25% for ingestion
            worker = asyncio.create_task(self._ingestion_worker(f"ingest-{i}"))
            self.ingestion_workers.append(worker)
            
        # Start processing workers
        for i in range(min(16, self.max_workers // 2)):  # 50% for processing
            worker = asyncio.create_task(self._processing_worker(f"process-{i}"))
            self.processing_workers.append(worker)
            
        # Start metrics collection
        self.metrics_task = asyncio.create_task(self._metrics_collector())
        
        logger.info(f"Started {len(self.ingestion_workers)} ingestion and {len(self.processing_workers)} processing workers")
        
    async def stop(self):
        """Stop the ingestion engine"""
        logger.info("Stopping parallel data ingestion engine")
        
        self.running = False
        
        # Stop workers
        all_workers = self.ingestion_workers + self.processing_workers
        for worker in all_workers:
            worker.cancel()
        await asyncio.gather(*all_workers, return_exceptions=True)
        
        # Stop processors
        for processor in self.processors.values():
            await processor.stop()
            
        # Stop source streams
        for stream in self.source_streams.values():
            stream.cancel()
        await asyncio.gather(*self.source_streams.values(), return_exceptions=True)
        
        # Stop metrics collection
        if hasattr(self, 'metrics_task'):
            self.metrics_task.cancel()
            
        # Close connections
        if self.nats_client:
            await self.nats_client.close()
        await self.redis_client.close()
        
    async def ingest_data(self, data: Any, source_type: DataSourceType,
                         priority: IngestionPriority = IngestionPriority.MEDIUM,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Ingest single data item"""
        
        packet = DataPacket(
            packet_id=str(uuid.uuid4()),
            source_type=source_type,
            priority=priority,
            data=data,
            metadata=metadata or {}
        )
        
        success = await self.priority_queue.put(packet)
        if success:
            self.metrics.total_packets += 1
            self.metrics.packets_per_priority[priority] += 1
            self.metrics.packets_per_source[source_type] += 1
            return packet.packet_id
        else:
            raise Exception("Ingestion queue full - backpressure applied")
            
    async def ingest_batch(self, data_items: List[Dict[str, Any]]) -> List[str]:
        """Ingest batch of data items"""
        packet_ids = []
        
        for item in data_items:
            data = item.get('data')
            source_type = DataSourceType(item.get('source_type', 'api_responses'))
            priority = IngestionPriority(item.get('priority', 'medium'))
            metadata = item.get('metadata', {})
            
            try:
                packet_id = await self.ingest_data(data, source_type, priority, metadata)
                packet_ids.append(packet_id)
            except Exception as e:
                logger.error(f"Failed to ingest batch item: {e}")
                
        return packet_ids
        
    async def register_data_source(self, source_name: str, source_type: DataSourceType,
                                 stream_handler: Callable, priority: IngestionPriority = IngestionPriority.MEDIUM):
        """Register streaming data source"""
        
        async def stream_worker():
            """Worker for streaming data source"""
            logger.info(f"Started stream worker for {source_name}")
            
            while self.running:
                try:
                    # Get data from source
                    data_items = await stream_handler()
                    
                    if data_items:
                        # Process each item
                        for data in data_items if isinstance(data_items, list) else [data_items]:
                            await self.ingest_data(data, source_type, priority)
                            
                except Exception as e:
                    logger.error(f"Stream worker {source_name} error: {e}")
                    await asyncio.sleep(5)  # Back off on error
                    
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
        # Start stream worker
        self.source_streams[source_name] = asyncio.create_task(stream_worker())
        logger.info(f"Registered data source: {source_name}")
        
    async def _initialize_processors(self):
        """Initialize processing stage processors"""
        
        # Determine worker allocation based on processing complexity
        stage_workers = {
            ProcessingStage.VALIDATION: 4,
            ProcessingStage.NORMALIZATION: 8,
            ProcessingStage.ENRICHMENT: 12,
            ProcessingStage.ANALYSIS: 8,
            ProcessingStage.STORAGE: 4
        }
        
        for stage, worker_count in stage_workers.items():
            processor = DataProcessor(stage, worker_count)
            await processor.start()
            self.processors[stage] = processor
            
    async def _ingestion_worker(self, worker_id: str):
        """Worker for handling data ingestion from queue"""
        logger.debug(f"Started ingestion worker: {worker_id}")
        
        while self.running:
            try:
                # Get batch of packets
                batch = await self.priority_queue.get_batch(self.batch_size)
                
                if not batch:
                    await asyncio.sleep(self.processing_interval)
                    continue
                    
                # Process batch through pipeline
                for packet in batch:
                    await self._process_packet_pipeline(packet)
                    
            except Exception as e:
                logger.error(f"Ingestion worker {worker_id} error: {e}")
                await asyncio.sleep(1)
                
    async def _processing_worker(self, worker_id: str):
        """Worker for handling data processing pipeline"""
        logger.debug(f"Started processing worker: {worker_id}")
        
        while self.running:
            try:
                # Monitor queue sizes and apply backpressure
                queue_utilization = self.priority_queue.size() / self.priority_queue.max_size
                
                if queue_utilization > self.backpressure_threshold:
                    # Apply backpressure by slowing processing
                    await asyncio.sleep(self.processing_interval * 2)
                else:
                    await asyncio.sleep(self.processing_interval)
                    
            except Exception as e:
                logger.error(f"Processing worker {worker_id} error: {e}")
                await asyncio.sleep(1)
                
    async def _process_packet_pipeline(self, packet: DataPacket):
        """Process packet through complete pipeline"""
        
        processing_stages = [
            ProcessingStage.VALIDATION,
            ProcessingStage.NORMALIZATION,
            ProcessingStage.ENRICHMENT,
            ProcessingStage.ANALYSIS,
            ProcessingStage.STORAGE
        ]
        
        current_packet = packet
        
        for stage in processing_stages:
            if stage in self.processors:
                start_time = time.time()
                
                try:
                    current_packet = await self.processors[stage].process(current_packet)
                    if not current_packet:
                        logger.warning(f"Packet {packet.packet_id} dropped at stage {stage}")
                        return
                        
                    # Record processing time
                    processing_time = time.time() - start_time
                    self.metrics.processing_times[stage].append(processing_time)
                    
                except Exception as e:
                    self.metrics.error_counts[f"{stage}_error"] += 1
                    logger.error(f"Pipeline stage {stage} error for packet {packet.packet_id}: {e}")
                    
                    # Retry logic
                    if current_packet.retry_count < 3:
                        current_packet.retry_count += 1
                        await self.priority_queue.put(current_packet)
                    return
                    
        # Packet successfully processed
        await self._finalize_packet(current_packet)
        
    async def _finalize_packet(self, packet: DataPacket):
        """Finalize processed packet"""
        
        # Store final result
        await self._store_processed_data(packet)
        
        # Emit notification if configured
        if self.jetstream:
            await self._emit_processing_event(packet)
            
        logger.debug(f"Finalized packet {packet.packet_id} from {packet.source_type}")
        
    async def _store_processed_data(self, packet: DataPacket):
        """Store processed data to Redis"""
        try:
            key = f"processed:{packet.source_type.value}:{packet.packet_id}"
            data = {
                'packet_id': packet.packet_id,
                'source_type': packet.source_type.value,
                'priority': packet.priority.value,
                'data': packet.data,
                'metadata': packet.metadata,
                'processing_completed': datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(key, 3600, json.dumps(data))  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Failed to store processed data: {e}")
            
    async def _emit_processing_event(self, packet: DataPacket):
        """Emit processing completion event"""
        try:
            event_data = {
                'packet_id': packet.packet_id,
                'source_type': packet.source_type.value,
                'priority': packet.priority.value,
                'processing_completed': datetime.utcnow().isoformat(),
                'size_bytes': packet.size_bytes,
                'retry_count': packet.retry_count
            }
            
            subject = f"xorb.ingestion.completed.{packet.source_type.value}"
            await self.jetstream.publish(subject, json.dumps(event_data).encode())
            
        except Exception as e:
            logger.error(f"Failed to emit processing event: {e}")
            
    async def _metrics_collector(self):
        """Collect and update ingestion metrics"""
        
        while self.running:
            try:
                # Update queue metrics
                self.metrics.backlog_size = self.priority_queue.size()
                
                # Calculate throughput
                current_time = time.time()
                self.metrics.throughput_history.append({
                    'timestamp': current_time,
                    'total_packets': self.metrics.total_packets,
                    'backlog_size': self.metrics.backlog_size
                })
                
                # Store metrics to Redis
                metrics_data = {
                    'total_packets': self.metrics.total_packets,
                    'backlog_size': self.metrics.backlog_size,
                    'packets_per_priority': {k.value: v for k, v in self.metrics.packets_per_priority.items()},
                    'packets_per_source': {k.value: v for k, v in self.metrics.packets_per_source.items()},
                    'error_counts': dict(self.metrics.error_counts),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self.redis_client.setex(
                    'xorb:ingestion:metrics',
                    300,  # 5 minutes
                    json.dumps(metrics_data)
                )
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
                
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ingestion metrics"""
        
        # Calculate throughput over last minute
        recent_throughput = []
        current_time = time.time()
        
        for entry in self.metrics.throughput_history:
            if current_time - entry['timestamp'] <= 60:  # Last minute
                recent_throughput.append(entry)
                
        throughput_per_minute = 0
        if len(recent_throughput) >= 2:
            time_diff = recent_throughput[-1]['timestamp'] - recent_throughput[0]['timestamp']
            packet_diff = recent_throughput[-1]['total_packets'] - recent_throughput[0]['total_packets']
            if time_diff > 0:
                throughput_per_minute = (packet_diff / time_diff) * 60
                
        # Calculate average processing times
        avg_processing_times = {}
        for stage, times in self.metrics.processing_times.items():
            if times:
                avg_processing_times[stage.value] = sum(times[-100:]) / len(times[-100:])  # Last 100 samples
                
        return {
            'total_packets_processed': self.metrics.total_packets,
            'current_backlog_size': self.metrics.backlog_size,
            'queue_utilization': self.priority_queue.size() / self.priority_queue.max_size,
            'throughput_per_minute': throughput_per_minute,
            'packets_by_priority': {k.value: v for k, v in self.metrics.packets_per_priority.items()},
            'packets_by_source': {k.value: v for k, v in self.metrics.packets_per_source.items()},
            'error_counts': dict(self.metrics.error_counts),
            'average_processing_times': avg_processing_times,
            'queue_sizes_by_priority': self.priority_queue.size_by_priority(),
            'active_workers': {
                'ingestion': len(self.ingestion_workers),
                'processing': len(self.processing_workers),
                'source_streams': len(self.source_streams)
            },
            'processor_metrics': {
                stage.value: processor.metrics for stage, processor in self.processors.items()
            }
        }