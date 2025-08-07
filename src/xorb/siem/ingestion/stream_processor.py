"""
Real-time log stream processor
Handles high-volume log ingestion, parsing, normalization, and routing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import time

from .log_parser import LogParserFactory, ParsedLog
from .event_normalizer import EventNormalizer, NormalizedEvent
from ..correlation.correlation_engine import CorrelationEngine
from ..correlation.threat_detector import ThreatDetector


class ProcessingStatus(Enum):
    """Stream processing status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ProcessingMetrics:
    """Processing performance metrics"""
    total_events_processed: int = 0
    events_per_second: float = 0.0
    parsing_errors: int = 0
    normalization_errors: int = 0
    correlation_alerts: int = 0
    threat_detections: int = 0
    last_processed_time: Optional[datetime] = None
    processing_latency_ms: float = 0.0
    queue_size: int = 0


class LogSource:
    """Abstract log source interface"""
    
    def __init__(self, source_id: str, source_type: str):
        self.source_id = source_id
        self.source_type = source_type
        self.enabled = True
    
    async def read_logs(self) -> AsyncGenerator[str, None]:
        """Async generator for reading log lines"""
        raise NotImplementedError


class FileLogSource(LogSource):
    """File-based log source"""
    
    def __init__(self, source_id: str, file_path: str, follow: bool = True):
        super().__init__(source_id, "file")
        self.file_path = file_path
        self.follow = follow
        self.position = 0
    
    async def read_logs(self) -> AsyncGenerator[str, None]:
        """Read logs from file"""
        try:
            with open(self.file_path, 'r') as f:
                f.seek(self.position)
                
                while True:
                    line = f.readline()
                    if line:
                        self.position = f.tell()
                        yield line.strip()
                    else:
                        if self.follow:
                            await asyncio.sleep(0.1)  # Wait for new content
                        else:
                            break
        except FileNotFoundError:
            logging.error(f"Log file not found: {self.file_path}")
        except Exception as e:
            logging.error(f"Error reading log file {self.file_path}: {e}")


class SyslogSource(LogSource):
    """Syslog UDP source"""
    
    def __init__(self, source_id: str, host: str = "0.0.0.0", port: int = 514):
        super().__init__(source_id, "syslog")
        self.host = host
        self.port = port
        self.transport = None
        self.protocol = None
    
    async def read_logs(self) -> AsyncGenerator[str, None]:
        """Read logs from syslog UDP"""
        loop = asyncio.get_event_loop()
        
        class SyslogProtocol(asyncio.DatagramProtocol):
            def __init__(self):
                self.queue = asyncio.Queue()
            
            def datagram_received(self, data, addr):
                try:
                    message = data.decode('utf-8').strip()
                    asyncio.create_task(self.queue.put(message))
                except Exception as e:
                    logging.error(f"Error processing syslog message: {e}")
        
        try:
            self.transport, self.protocol = await loop.create_datagram_endpoint(
                SyslogProtocol, local_addr=(self.host, self.port)
            )
            
            while True:
                try:
                    message = await asyncio.wait_for(self.protocol.queue.get(), timeout=1.0)
                    yield message
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logging.error(f"Error setting up syslog source: {e}")
        finally:
            if self.transport:
                self.transport.close()


class KafkaSource(LogSource):
    """Kafka log source"""
    
    def __init__(self, source_id: str, topic: str, bootstrap_servers: List[str]):
        super().__init__(source_id, "kafka")
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.consumer = None
    
    async def read_logs(self) -> AsyncGenerator[str, None]:
        """Read logs from Kafka topic"""
        try:
            # This would require aiokafka or similar async Kafka client
            # For now, this is a placeholder implementation
            logging.warning("Kafka source not implemented - requires aiokafka")
            yield ""
        except Exception as e:
            logging.error(f"Error reading from Kafka: {e}")


class StreamProcessor:
    """High-performance log stream processor"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 queue_size: int = 10000,
                 batch_size: int = 100,
                 flush_interval: float = 1.0):
        
        # Core components
        self.log_parser_factory = LogParserFactory()
        self.event_normalizer = EventNormalizer()
        self.correlation_engine = CorrelationEngine()
        self.threat_detector = ThreatDetector()
        
        # Processing configuration
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Processing queues
        self.raw_log_queue = Queue(maxsize=queue_size)
        self.parsed_log_queue = Queue(maxsize=queue_size)
        self.normalized_event_queue = Queue(maxsize=queue_size)
        
        # State management
        self.status = ProcessingStatus.STOPPED
        self.metrics = ProcessingMetrics()
        self.log_sources: Dict[str, LogSource] = {}
        self.event_handlers: List[Callable[[NormalizedEvent], None]] = []
        self.alert_handlers: List[Callable[[Any], None]] = []
        
        # Worker threads
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.worker_threads = []
        self.stop_event = threading.Event()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def add_log_source(self, source: LogSource):
        """Add log source to processor"""
        self.log_sources[source.source_id] = source
        self.logger.info(f"Added log source: {source.source_id} ({source.source_type})")
    
    def remove_log_source(self, source_id: str):
        """Remove log source from processor"""
        if source_id in self.log_sources:
            del self.log_sources[source_id]
            self.logger.info(f"Removed log source: {source_id}")
    
    def add_event_handler(self, handler: Callable[[NormalizedEvent], None]):
        """Add event handler for processed events"""
        self.event_handlers.append(handler)
    
    def add_alert_handler(self, handler: Callable[[Any], None]):
        """Add alert handler for generated alerts"""
        self.alert_handlers.append(handler)
    
    async def start(self):
        """Start stream processing"""
        if self.status != ProcessingStatus.STOPPED:
            raise RuntimeError("Processor already running")
        
        self.status = ProcessingStatus.STARTING
        self.stop_event.clear()
        
        self.logger.info("Starting stream processor...")
        
        # Start worker threads
        self._start_workers()
        
        # Start log source readers
        source_tasks = []
        for source in self.log_sources.values():
            if source.enabled:
                task = asyncio.create_task(self._read_from_source(source))
                source_tasks.append(task)
        
        # Start metrics updater
        metrics_task = asyncio.create_task(self._update_metrics())
        
        self.status = ProcessingStatus.RUNNING
        self.logger.info("Stream processor started successfully")
        
        try:
            # Wait for all tasks
            await asyncio.gather(*source_tasks, metrics_task)
        except Exception as e:
            self.logger.error(f"Error in stream processor: {e}")
            self.status = ProcessingStatus.ERROR
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop stream processing"""
        if self.status == ProcessingStatus.STOPPED:
            return
        
        self.status = ProcessingStatus.STOPPING
        self.logger.info("Stopping stream processor...")
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.status = ProcessingStatus.STOPPED
        self.logger.info("Stream processor stopped")
    
    async def _read_from_source(self, source: LogSource):
        """Read logs from a specific source"""
        self.logger.info(f"Starting to read from source: {source.source_id}")
        
        try:
            async for log_line in source.read_logs():
                if self.stop_event.is_set():
                    break
                
                if log_line:
                    try:
                        self.raw_log_queue.put_nowait({
                            'log_line': log_line,
                            'source_id': source.source_id,
                            'timestamp': datetime.utcnow()
                        })
                    except:
                        # Queue full - drop message or implement backpressure
                        self.logger.warning(f"Raw log queue full, dropping message from {source.source_id}")
                        
        except Exception as e:
            self.logger.error(f"Error reading from source {source.source_id}: {e}")
    
    def _start_workers(self):
        """Start worker threads for processing pipeline"""
        # Parsing workers
        for i in range(self.max_workers // 2 or 1):
            thread = threading.Thread(target=self._parsing_worker, name=f"parser-{i}")
            thread.start()
            self.worker_threads.append(thread)
        
        # Normalization workers
        for i in range(self.max_workers // 2 or 1):
            thread = threading.Thread(target=self._normalization_worker, name=f"normalizer-{i}")
            thread.start()
            self.worker_threads.append(thread)
        
        # Correlation worker
        thread = threading.Thread(target=self._correlation_worker, name="correlator")
        thread.start()
        self.worker_threads.append(thread)
    
    def _parsing_worker(self):
        """Worker thread for log parsing"""
        self.logger.info(f"Starting parsing worker: {threading.current_thread().name}")
        
        while not self.stop_event.is_set():
            try:
                # Get raw log entry
                raw_entry = self.raw_log_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # Parse log
                parsed_log = self.log_parser_factory.parse_log(raw_entry['log_line'])
                
                if parsed_log:
                    # Add source metadata
                    parsed_log.metadata['source_id'] = raw_entry['source_id']
                    parsed_log.metadata['ingestion_time'] = raw_entry['timestamp']
                    
                    # Queue for normalization
                    self.parsed_log_queue.put_nowait(parsed_log)
                    
                    # Update metrics
                    processing_time = (time.time() - start_time) * 1000
                    self.metrics.processing_latency_ms = (
                        self.metrics.processing_latency_ms * 0.9 + processing_time * 0.1
                    )
                else:
                    self.metrics.parsing_errors += 1
                    self.logger.debug(f"Failed to parse log: {raw_entry['log_line'][:100]}")
                
            except Empty:
                continue
            except Exception as e:
                self.metrics.parsing_errors += 1
                self.logger.error(f"Error in parsing worker: {e}")
    
    def _normalization_worker(self):
        """Worker thread for event normalization"""
        self.logger.info(f"Starting normalization worker: {threading.current_thread().name}")
        
        while not self.stop_event.is_set():
            try:
                # Get parsed log
                parsed_log = self.parsed_log_queue.get(timeout=1.0)
                
                # Normalize event
                normalized_event = self.event_normalizer.normalize(parsed_log)
                
                if normalized_event:
                    # Queue for correlation
                    self.normalized_event_queue.put_nowait(normalized_event)
                    
                    # Notify event handlers
                    for handler in self.event_handlers:
                        try:
                            handler(normalized_event)
                        except Exception as e:
                            self.logger.error(f"Error in event handler: {e}")
                    
                    self.metrics.total_events_processed += 1
                    self.metrics.last_processed_time = datetime.utcnow()
                else:
                    self.metrics.normalization_errors += 1
                
            except Empty:
                continue
            except Exception as e:
                self.metrics.normalization_errors += 1
                self.logger.error(f"Error in normalization worker: {e}")
    
    def _correlation_worker(self):
        """Worker thread for correlation and threat detection"""
        self.logger.info(f"Starting correlation worker: {threading.current_thread().name}")
        
        while not self.stop_event.is_set():
            try:
                # Get normalized event
                normalized_event = self.normalized_event_queue.get(timeout=1.0)
                
                # Run correlation analysis
                correlation_alerts = self.correlation_engine.process_event(normalized_event)
                for alert in correlation_alerts:
                    self.metrics.correlation_alerts += 1
                    
                    # Notify alert handlers
                    for handler in self.alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            self.logger.error(f"Error in alert handler: {e}")
                
                # Run threat detection
                threat_detections = self.threat_detector.analyze_event(normalized_event)
                for detection in threat_detections:
                    self.metrics.threat_detections += 1
                    
                    # Notify alert handlers
                    for handler in self.alert_handlers:
                        try:
                            handler(detection)
                        except Exception as e:
                            self.logger.error(f"Error in threat detection handler: {e}")
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in correlation worker: {e}")
    
    async def _update_metrics(self):
        """Update processing metrics periodically"""
        last_event_count = 0
        last_update_time = time.time()
        
        while not self.stop_event.is_set():
            await asyncio.sleep(self.flush_interval)
            
            current_time = time.time()
            current_events = self.metrics.total_events_processed
            
            # Calculate events per second
            time_diff = current_time - last_update_time
            event_diff = current_events - last_event_count
            
            if time_diff > 0:
                self.metrics.events_per_second = event_diff / time_diff
            
            # Update queue sizes
            self.metrics.queue_size = (
                self.raw_log_queue.qsize() + 
                self.parsed_log_queue.qsize() + 
                self.normalized_event_queue.qsize()
            )
            
            last_event_count = current_events
            last_update_time = current_time
            
            # Log metrics periodically
            if self.metrics.total_events_processed % 1000 == 0 and self.metrics.total_events_processed > 0:
                self.logger.info(
                    f"Processed {self.metrics.total_events_processed} events, "
                    f"EPS: {self.metrics.events_per_second:.2f}, "
                    f"Queue: {self.metrics.queue_size}, "
                    f"Alerts: {self.metrics.correlation_alerts}, "
                    f"Threats: {self.metrics.threat_detections}"
                )
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics"""
        return self.metrics
    
    def get_status(self) -> ProcessingStatus:
        """Get current processing status"""
        return self.status
    
    def get_sources_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all log sources"""
        return {
            source_id: {
                "type": source.source_type,
                "enabled": source.enabled,
                "config": {
                    "file_path": getattr(source, 'file_path', None),
                    "host": getattr(source, 'host', None),
                    "port": getattr(source, 'port', None),
                    "topic": getattr(source, 'topic', None)
                }
            }
            for source_id, source in self.log_sources.items()
        }
    
    async def process_single_log(self, log_line: str, source_id: str = "manual") -> Optional[NormalizedEvent]:
        """Process a single log line synchronously for testing"""
        try:
            # Parse
            parsed_log = self.log_parser_factory.parse_log(log_line)
            if not parsed_log:
                return None
            
            # Add metadata
            parsed_log.metadata['source_id'] = source_id
            parsed_log.metadata['ingestion_time'] = datetime.utcnow()
            
            # Normalize
            normalized_event = self.event_normalizer.normalize(parsed_log)
            if not normalized_event:
                return None
            
            # Process correlations and threats
            correlation_alerts = self.correlation_engine.process_event(normalized_event)
            threat_detections = self.threat_detector.analyze_event(normalized_event)
            
            return normalized_event
            
        except Exception as e:
            self.logger.error(f"Error processing single log: {e}")
            return None