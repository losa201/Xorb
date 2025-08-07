#!/usr/bin/env python3
"""
XORB Resilience & Scalability Layer - Multi-Region Data Replication & Persistence
Advanced data replication with zero-downtime backup and restore mechanisms
"""

import asyncio
import json
import time
import logging
import hashlib
import gzip
import pickle
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiofiles
import aiohttp
import uuid
import os
from pathlib import Path
import sqlite3
import threading
from collections import defaultdict, deque
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplicationStrategy(Enum):
    """Data replication strategies"""
    MASTER_SLAVE = "master_slave"
    MASTER_MASTER = "master_master"
    RING_REPLICATION = "ring_replication"
    MESH_REPLICATION = "mesh_replication"
    EVENTUAL_CONSISTENCY = "eventual_consistency"

class ConsistencyLevel(Enum):
    """Data consistency levels"""
    STRONG = "strong"              # All replicas must confirm
    EVENTUAL = "eventual"          # Eventually consistent
    QUORUM = "quorum"             # Majority of replicas
    ONE = "one"                   # Any single replica
    LOCAL_QUORUM = "local_quorum" # Majority in local datacenter

class ReplicationStatus(Enum):
    """Replication status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

class BackupType(Enum):
    """Backup types"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    CONTINUOUS = "continuous"

@dataclass
class ReplicationNode:
    """Replication node information"""
    node_id: str
    region: str
    datacenter: str
    host: str
    port: int
    status: ReplicationStatus
    last_heartbeat: datetime
    lag_ms: float = 0.0
    data_version: int = 0
    storage_capacity_gb: float = 0.0
    storage_used_gb: float = 0.0
    network_bandwidth_mbps: float = 0.0
    is_master: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataRecord:
    """Data record for replication"""
    record_id: str
    data_type: str
    content: bytes
    checksum: str
    timestamp: datetime
    version: int
    source_node: str
    replicated_to: Set[str] = field(default_factory=set)
    compression_type: Optional[str] = None

@dataclass
class ReplicationTransaction:
    """Replication transaction"""
    transaction_id: str
    source_node: str
    target_nodes: List[str]
    records: List[DataRecord]
    consistency_level: ConsistencyLevel
    timestamp: datetime
    status: str = "pending"
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)

@dataclass
class BackupMetadata:
    """Backup metadata"""
    backup_id: str
    backup_type: BackupType
    source_nodes: List[str]
    backup_path: str
    size_bytes: int
    checksum: str
    created_at: datetime
    compressed: bool = True
    encrypted: bool = False
    retention_days: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

class XORBDataReplicationManager:
    """Multi-region data replication manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.manager_id = str(uuid.uuid4())
        
        # Replication topology
        self.nodes: Dict[str, ReplicationNode] = {}
        self.replication_strategy = ReplicationStrategy.MASTER_SLAVE
        self.default_consistency_level = ConsistencyLevel.QUORUM
        
        # Data management
        self.data_store: Dict[str, DataRecord] = {}
        self.pending_transactions: Dict[str, ReplicationTransaction] = {}
        self.replication_log: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.replication_metrics = {
            'total_operations': 0,
            'successful_replications': 0,
            'failed_replications': 0,
            'avg_replication_time_ms': 0.0,
            'data_transfer_mb': 0.0,
            'network_bandwidth_utilization': 0.0
        }
        
        # Configuration
        self.heartbeat_interval = self.config.get('heartbeat_interval', 30)
        self.replication_timeout = self.config.get('replication_timeout', 60)
        self.max_lag_tolerance_ms = self.config.get('max_lag_tolerance_ms', 1000)
        self.compression_enabled = self.config.get('compression_enabled', True)
        
        # Storage paths
        self.data_path = Path(self.config.get('data_path', '/tmp/xorb_data'))
        self.backup_path = Path(self.config.get('backup_path', '/tmp/xorb_backups'))
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize local storage
        self._initialize_local_storage()
        
        logger.info(f"Data Replication Manager initialized: {self.manager_id}")
    
    def _initialize_local_storage(self):
        """Initialize local storage database"""
        try:
            self.db_path = self.data_path / "replication.db"
            self.db_connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.db_lock = threading.Lock()
            
            # Create tables
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_records (
                    record_id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    content BLOB NOT NULL,
                    checksum TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    source_node TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replication_log (
                    log_id TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    record_id TEXT,
                    source_node TEXT,
                    target_node TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backup_metadata (
                    backup_id TEXT PRIMARY KEY,
                    backup_type TEXT NOT NULL,
                    source_nodes TEXT NOT NULL,
                    backup_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    compressed BOOLEAN NOT NULL,
                    encrypted BOOLEAN NOT NULL,
                    retention_days INTEGER NOT NULL,
                    metadata TEXT
                )
            """)
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to initialize local storage: {e}")
            raise e
    
    async def register_node(self, node_id: str, region: str, datacenter: str, 
                          host: str, port: int, is_master: bool = False, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a replication node"""
        try:
            node = ReplicationNode(
                node_id=node_id,
                region=region,
                datacenter=datacenter,
                host=host,
                port=port,
                status=ReplicationStatus.HEALTHY,
                last_heartbeat=datetime.now(),
                is_master=is_master,
                metadata=metadata or {}
            )
            
            self.nodes[node_id] = node
            
            # Update node status
            await self._update_node_status(node_id)
            
            logger.info(f"Registered replication node: {node_id} ({region}/{datacenter})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return False
    
    async def deregister_node(self, node_id: str) -> bool:
        """Deregister a replication node"""
        try:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Handle data migration if this was a master
                if node.is_master:
                    await self._handle_master_failover(node_id)
                
                del self.nodes[node_id]
                logger.info(f"Deregistered replication node: {node_id}")
                return True
            else:
                logger.warning(f"Node not found for deregistration: {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to deregister node {node_id}: {e}")
            return False
    
    async def store_data(self, data_type: str, content: Union[str, bytes, Dict[str, Any]], 
                        consistency_level: Optional[ConsistencyLevel] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data with replication"""
        try:
            # Generate record ID
            record_id = str(uuid.uuid4())
            
            # Serialize content
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            elif isinstance(content, dict):
                content_bytes = json.dumps(content).encode('utf-8')
            else:
                content_bytes = content
            
            # Compress if enabled
            if self.compression_enabled:
                content_bytes = gzip.compress(content_bytes)
                compression_type = "gzip"
            else:
                compression_type = None
            
            # Calculate checksum
            checksum = hashlib.sha256(content_bytes).hexdigest()
            
            # Create data record
            record = DataRecord(
                record_id=record_id,
                data_type=data_type,
                content=content_bytes,
                checksum=checksum,
                timestamp=datetime.now(),
                version=1,
                source_node=self.manager_id,
                compression_type=compression_type
            )
            
            # Store locally
            await self._store_local(record)
            
            # Replicate to other nodes
            consistency_level = consistency_level or self.default_consistency_level
            await self._replicate_data(record, consistency_level)
            
            logger.info(f"Stored data record: {record_id} ({data_type})")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            raise e
    
    async def retrieve_data(self, record_id: str, consistency_level: Optional[ConsistencyLevel] = None) -> Optional[DataRecord]:
        """Retrieve data with consistency guarantees"""
        try:
            consistency_level = consistency_level or ConsistencyLevel.ONE
            
            if consistency_level == ConsistencyLevel.ONE:
                # Read from local storage first
                record = await self._retrieve_local(record_id)
                if record:
                    return record
                
                # Try other nodes
                for node_id, node in self.nodes.items():
                    if node.status == ReplicationStatus.HEALTHY:
                        record = await self._retrieve_from_node(node_id, record_id)
                        if record:
                            return record
            
            elif consistency_level == ConsistencyLevel.QUORUM:
                # Read from majority of nodes
                responses = await self._read_from_multiple_nodes(record_id)
                if len(responses) >= (len(self.nodes) // 2 + 1):
                    # Return most recent version
                    return max(responses, key=lambda r: r.version)
            
            elif consistency_level == ConsistencyLevel.STRONG:
                # Read from all nodes and ensure consistency
                responses = await self._read_from_all_nodes(record_id)
                if responses and all(r.version == responses[0].version for r in responses):
                    return responses[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve data {record_id}: {e}")
            return None
    
    async def _store_local(self, record: DataRecord):
        """Store data record locally"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO data_records 
                    (record_id, data_type, content, checksum, timestamp, version, source_node, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id,
                    record.data_type,
                    record.content,
                    record.checksum,
                    record.timestamp.isoformat(),
                    record.version,
                    record.source_node,
                    json.dumps({'compression_type': record.compression_type})
                ))
                self.db_connection.commit()
            
            # Update in-memory store
            self.data_store[record.record_id] = record
            
        except Exception as e:
            logger.error(f"Failed to store local record {record.record_id}: {e}")
            raise e
    
    async def _retrieve_local(self, record_id: str) -> Optional[DataRecord]:
        """Retrieve data record locally"""
        try:
            # Check in-memory first
            if record_id in self.data_store:
                return self.data_store[record_id]
            
            # Check database
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT record_id, data_type, content, checksum, timestamp, version, source_node, metadata
                    FROM data_records WHERE record_id = ?
                """, (record_id,))
                
                row = cursor.fetchone()
                if row:
                    metadata = json.loads(row[7]) if row[7] else {}
                    record = DataRecord(
                        record_id=row[0],
                        data_type=row[1],
                        content=row[2],
                        checksum=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        version=row[5],
                        source_node=row[6],
                        compression_type=metadata.get('compression_type')
                    )
                    
                    # Cache in memory
                    self.data_store[record_id] = record
                    return record
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve local record {record_id}: {e}")
            return None
    
    async def _replicate_data(self, record: DataRecord, consistency_level: ConsistencyLevel):
        """Replicate data to other nodes"""
        try:
            # Select target nodes based on strategy
            target_nodes = await self._select_replication_targets(record.source_node)
            
            if not target_nodes:
                logger.warning("No target nodes available for replication")
                return
            
            # Create replication transaction
            transaction = ReplicationTransaction(
                transaction_id=str(uuid.uuid4()),
                source_node=record.source_node,
                target_nodes=[node.node_id for node in target_nodes],
                records=[record],
                consistency_level=consistency_level,
                timestamp=datetime.now()
            )
            
            self.pending_transactions[transaction.transaction_id] = transaction
            
            # Execute replication
            await self._execute_replication_transaction(transaction)
            
        except Exception as e:
            logger.error(f"Failed to replicate data {record.record_id}: {e}")
    
    async def _select_replication_targets(self, source_node: str) -> List[ReplicationNode]:
        """Select target nodes for replication"""
        try:
            healthy_nodes = [node for node in self.nodes.values() 
                           if node.status == ReplicationStatus.HEALTHY and node.node_id != source_node]
            
            if self.replication_strategy == ReplicationStrategy.MASTER_SLAVE:
                # Replicate to all slaves
                return [node for node in healthy_nodes if not node.is_master][:3]  # Limit to 3 replicas
                
            elif self.replication_strategy == ReplicationStrategy.MASTER_MASTER:
                # Replicate to all other masters
                return [node for node in healthy_nodes if node.is_master]
                
            elif self.replication_strategy == ReplicationStrategy.RING_REPLICATION:
                # Replicate to next N nodes in ring
                sorted_nodes = sorted(healthy_nodes, key=lambda n: n.node_id)
                source_index = next((i for i, n in enumerate(sorted_nodes) if n.node_id == source_node), 0)
                targets = []
                for i in range(1, min(4, len(sorted_nodes) + 1)):  # Next 3 nodes
                    target_index = (source_index + i) % len(sorted_nodes)
                    targets.append(sorted_nodes[target_index])
                return targets
                
            else:
                # Default: replicate to up to 3 random healthy nodes
                import random
                return random.sample(healthy_nodes, min(3, len(healthy_nodes)))
                
        except Exception as e:
            logger.error(f"Failed to select replication targets: {e}")
            return []
    
    async def _execute_replication_transaction(self, transaction: ReplicationTransaction):
        """Execute replication transaction"""
        try:
            start_time = time.time()
            replication_tasks = []
            
            # Create replication tasks for each target node
            for target_node_id in transaction.target_nodes:
                if target_node_id in self.nodes:
                    target_node = self.nodes[target_node_id]
                    for record in transaction.records:
                        task = self._replicate_to_node(target_node, record, transaction.transaction_id)
                        replication_tasks.append(task)
            
            # Execute replication tasks
            if replication_tasks:
                results = await asyncio.gather(*replication_tasks, return_exceptions=True)
                
                # Process results
                successful_replications = 0
                for i, result in enumerate(results):
                    target_node_id = transaction.target_nodes[i % len(transaction.target_nodes)]
                    
                    if isinstance(result, Exception):
                        transaction.failed_nodes.add(target_node_id)
                        logger.error(f"Replication failed to node {target_node_id}: {result}")
                    else:
                        transaction.completed_nodes.add(target_node_id)
                        successful_replications += 1
                
                # Check consistency requirements
                required_replications = self._get_required_replications(transaction.consistency_level, len(transaction.target_nodes))
                
                if successful_replications >= required_replications:
                    transaction.status = "completed"
                    logger.info(f"Replication transaction completed: {transaction.transaction_id}")
                else:
                    transaction.status = "failed"
                    logger.error(f"Replication transaction failed: {transaction.transaction_id} (only {successful_replications}/{required_replications} successful)")
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000  # ms
            self.replication_metrics['total_operations'] += 1
            if transaction.status == "completed":
                self.replication_metrics['successful_replications'] += 1
            else:
                self.replication_metrics['failed_replications'] += 1
            
            # Update average replication time
            current_avg = self.replication_metrics.get('avg_replication_time_ms', 0)
            total_ops = self.replication_metrics['total_operations']
            self.replication_metrics['avg_replication_time_ms'] = ((current_avg * (total_ops - 1)) + execution_time) / total_ops
            
            # Log transaction
            await self._log_replication_transaction(transaction, execution_time)
            
        except Exception as e:
            logger.error(f"Failed to execute replication transaction {transaction.transaction_id}: {e}")
            transaction.status = "failed"
    
    def _get_required_replications(self, consistency_level: ConsistencyLevel, total_targets: int) -> int:
        """Get required number of successful replications"""
        if consistency_level == ConsistencyLevel.STRONG:
            return total_targets  # All must succeed
        elif consistency_level == ConsistencyLevel.QUORUM:
            return (total_targets // 2) + 1  # Majority
        elif consistency_level == ConsistencyLevel.ONE:
            return 1  # Any one
        else:
            return (total_targets // 2) + 1  # Default to quorum
    
    async def _replicate_to_node(self, target_node: ReplicationNode, record: DataRecord, transaction_id: str) -> bool:
        """Replicate single record to target node"""
        try:
            # Prepare replication payload
            payload = {
                'transaction_id': transaction_id,
                'record': {
                    'record_id': record.record_id,
                    'data_type': record.data_type,
                    'content': record.content.hex(),  # Convert bytes to hex string
                    'checksum': record.checksum,
                    'timestamp': record.timestamp.isoformat(),
                    'version': record.version,
                    'source_node': record.source_node,
                    'compression_type': record.compression_type
                }
            }
            
            # Send replication request
            url = f"http://{target_node.host}:{target_node.port}/replicate"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.replication_timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('success'):
                            record.replicated_to.add(target_node.node_id)
                            return True
                    
                    logger.error(f"Replication to {target_node.node_id} failed: HTTP {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.error(f"Replication timeout to node {target_node.node_id}")
            return False
        except Exception as e:
            logger.error(f"Replication error to node {target_node.node_id}: {e}")
            return False
    
    async def _log_replication_transaction(self, transaction: ReplicationTransaction, execution_time_ms: float):
        """Log replication transaction"""
        try:
            log_entry = {
                'log_id': str(uuid.uuid4()),
                'transaction_id': transaction.transaction_id,
                'operation': 'replication',
                'source_node': transaction.source_node,
                'target_nodes': transaction.target_nodes,
                'timestamp': transaction.timestamp.isoformat(),
                'status': transaction.status,
                'execution_time_ms': execution_time_ms,
                'completed_nodes': list(transaction.completed_nodes),
                'failed_nodes': list(transaction.failed_nodes),
                'record_count': len(transaction.records)
            }
            
            self.replication_log.append(log_entry)
            
            # Store in database
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO replication_log 
                    (log_id, transaction_id, operation, record_id, source_node, target_node, timestamp, status, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_entry['log_id'],
                    transaction.transaction_id,
                    'replication',
                    transaction.records[0].record_id if transaction.records else None,
                    transaction.source_node,
                    ','.join(transaction.target_nodes),
                    transaction.timestamp.isoformat(),
                    transaction.status,
                    json.dumps({
                        'execution_time_ms': execution_time_ms,
                        'completed_nodes': list(transaction.completed_nodes),
                        'failed_nodes': list(transaction.failed_nodes)
                    })
                ))
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Failed to log replication transaction: {e}")
    
    async def create_backup(self, backup_type: BackupType, source_nodes: Optional[List[str]] = None,
                          retention_days: int = 30) -> str:
        """Create data backup"""
        try:
            backup_id = f"backup_{backup_type.value}_{int(time.time())}"
            
            # Default to all nodes if not specified
            if not source_nodes:
                source_nodes = list(self.nodes.keys())
            
            # Create backup directory
            backup_dir = self.backup_path / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect data from source nodes
            backup_data = {}
            
            if backup_type == BackupType.FULL:
                # Full backup - all data
                backup_data = await self._collect_full_backup_data(source_nodes)
            elif backup_type == BackupType.INCREMENTAL:
                # Incremental - changes since last backup
                backup_data = await self._collect_incremental_backup_data(source_nodes)
            elif backup_type == BackupType.DIFFERENTIAL:
                # Differential - changes since last full backup
                backup_data = await self._collect_differential_backup_data(source_nodes)
            
            # Serialize and compress backup data
            backup_content = pickle.dumps(backup_data)
            if self.compression_enabled:
                backup_content = gzip.compress(backup_content)
            
            # Write backup file
            backup_file = backup_dir / "data.backup"
            async with aiofiles.open(backup_file, 'wb') as f:
                await f.write(backup_content)
            
            # Calculate checksum
            checksum = hashlib.sha256(backup_content).hexdigest()
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                source_nodes=source_nodes,
                backup_path=str(backup_file),
                size_bytes=len(backup_content),
                checksum=checksum,
                created_at=datetime.now(),
                compressed=self.compression_enabled,
                retention_days=retention_days
            )
            
            # Store metadata
            await self._store_backup_metadata(metadata)
            
            logger.info(f"Created {backup_type.value} backup: {backup_id} ({len(backup_content)} bytes)")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise e
    
    async def restore_backup(self, backup_id: str, target_nodes: Optional[List[str]] = None) -> bool:
        """Restore data from backup"""
        try:
            # Load backup metadata
            metadata = await self._load_backup_metadata(backup_id)
            if not metadata:
                logger.error(f"Backup metadata not found: {backup_id}")
                return False
            
            # Read backup file
            backup_file = Path(metadata.backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False
            
            async with aiofiles.open(backup_file, 'rb') as f:
                backup_content = await f.read()
            
            # Verify checksum
            checksum = hashlib.sha256(backup_content).hexdigest()
            if checksum != metadata.checksum:
                logger.error(f"Backup checksum mismatch: {backup_id}")
                return False
            
            # Decompress if needed
            if metadata.compressed:
                backup_content = gzip.decompress(backup_content)
            
            # Deserialize backup data
            backup_data = pickle.loads(backup_content)
            
            # Default to all healthy nodes if not specified
            if not target_nodes:
                target_nodes = [node.node_id for node in self.nodes.values() 
                              if node.status == ReplicationStatus.HEALTHY]
            
            # Restore data to target nodes
            restore_success = await self._restore_data_to_nodes(backup_data, target_nodes)
            
            if restore_success:
                logger.info(f"Successfully restored backup: {backup_id}")
                return True
            else:
                logger.error(f"Failed to restore backup: {backup_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            return False
    
    async def _collect_full_backup_data(self, source_nodes: List[str]) -> Dict[str, Any]:
        """Collect full backup data from source nodes"""
        try:
            backup_data = {
                'records': {},
                'metadata': {
                    'backup_type': 'full',
                    'source_nodes': source_nodes,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Collect all records from local storage
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT record_id, data_type, content, checksum, timestamp, version, source_node, metadata
                    FROM data_records
                """)
                
                rows = cursor.fetchall()
                for row in rows:
                    record_data = {
                        'data_type': row[1],
                        'content': row[2],
                        'checksum': row[3],
                        'timestamp': row[4],
                        'version': row[5],
                        'source_node': row[6],
                        'metadata': row[7]
                    }
                    backup_data['records'][row[0]] = record_data
            
            return backup_data
            
        except Exception as e:
            logger.error(f"Failed to collect full backup data: {e}")
            return {}
    
    async def _collect_incremental_backup_data(self, source_nodes: List[str]) -> Dict[str, Any]:
        """Collect incremental backup data"""
        try:
            # For demonstration, collect records modified in last 24 hours
            cutoff_time = datetime.now() - timedelta(days=1)
            
            backup_data = {
                'records': {},
                'metadata': {
                    'backup_type': 'incremental',
                    'source_nodes': source_nodes,
                    'timestamp': datetime.now().isoformat(),
                    'since': cutoff_time.isoformat()
                }
            }
            
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT record_id, data_type, content, checksum, timestamp, version, source_node, metadata
                    FROM data_records WHERE timestamp > ?
                """, (cutoff_time.isoformat(),))
                
                rows = cursor.fetchall()
                for row in rows:
                    record_data = {
                        'data_type': row[1],
                        'content': row[2],
                        'checksum': row[3],
                        'timestamp': row[4],
                        'version': row[5],
                        'source_node': row[6],
                        'metadata': row[7]
                    }
                    backup_data['records'][row[0]] = record_data
            
            return backup_data
            
        except Exception as e:
            logger.error(f"Failed to collect incremental backup data: {e}")
            return {}
    
    async def _collect_differential_backup_data(self, source_nodes: List[str]) -> Dict[str, Any]:
        """Collect differential backup data"""
        try:
            # Find last full backup time
            last_full_backup = await self._find_last_full_backup()
            if not last_full_backup:
                # If no full backup, collect all data
                return await self._collect_full_backup_data(source_nodes)
            
            cutoff_time = last_full_backup.created_at
            
            backup_data = {
                'records': {},
                'metadata': {
                    'backup_type': 'differential',
                    'source_nodes': source_nodes,
                    'timestamp': datetime.now().isoformat(),
                    'since_full_backup': cutoff_time.isoformat()
                }
            }
            
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT record_id, data_type, content, checksum, timestamp, version, source_node, metadata
                    FROM data_records WHERE timestamp > ?
                """, (cutoff_time.isoformat(),))
                
                rows = cursor.fetchall()
                for row in rows:
                    record_data = {
                        'data_type': row[1],
                        'content': row[2],
                        'checksum': row[3],
                        'timestamp': row[4],
                        'version': row[5],
                        'source_node': row[6],
                        'metadata': row[7]
                    }
                    backup_data['records'][row[0]] = record_data
            
            return backup_data
            
        except Exception as e:
            logger.error(f"Failed to collect differential backup data: {e}")
            return {}
    
    async def _store_backup_metadata(self, metadata: BackupMetadata):
        """Store backup metadata"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO backup_metadata 
                    (backup_id, backup_type, source_nodes, backup_path, size_bytes, checksum, 
                     created_at, compressed, encrypted, retention_days, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.backup_id,
                    metadata.backup_type.value,
                    json.dumps(metadata.source_nodes),
                    metadata.backup_path,
                    metadata.size_bytes,
                    metadata.checksum,
                    metadata.created_at.isoformat(),
                    metadata.compressed,
                    metadata.encrypted,
                    metadata.retention_days,
                    json.dumps(metadata.metadata)
                ))
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Failed to store backup metadata: {e}")
            raise e
    
    async def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load backup metadata"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT backup_id, backup_type, source_nodes, backup_path, size_bytes, checksum, 
                           created_at, compressed, encrypted, retention_days, metadata
                    FROM backup_metadata WHERE backup_id = ?
                """, (backup_id,))
                
                row = cursor.fetchone()
                if row:
                    return BackupMetadata(
                        backup_id=row[0],
                        backup_type=BackupType(row[1]),
                        source_nodes=json.loads(row[2]),
                        backup_path=row[3],
                        size_bytes=row[4],
                        checksum=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        compressed=bool(row[7]),
                        encrypted=bool(row[8]),
                        retention_days=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load backup metadata {backup_id}: {e}")
            return None
    
    async def _find_last_full_backup(self) -> Optional[BackupMetadata]:
        """Find the most recent full backup"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT backup_id, backup_type, source_nodes, backup_path, size_bytes, checksum, 
                           created_at, compressed, encrypted, retention_days, metadata
                    FROM backup_metadata 
                    WHERE backup_type = 'full' 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row:
                    return BackupMetadata(
                        backup_id=row[0],
                        backup_type=BackupType(row[1]),
                        source_nodes=json.loads(row[2]),
                        backup_path=row[3],
                        size_bytes=row[4],
                        checksum=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        compressed=bool(row[7]),
                        encrypted=bool(row[8]),
                        retention_days=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find last full backup: {e}")
            return None
    
    async def cleanup_expired_backups(self) -> int:
        """Clean up expired backups"""
        try:
            cleanup_count = 0
            current_time = datetime.now()
            
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT backup_id, backup_path, created_at, retention_days
                    FROM backup_metadata
                """)
                
                rows = cursor.fetchall()
                for row in rows:
                    backup_id = row[0]
                    backup_path = row[1]
                    created_at = datetime.fromisoformat(row[2])
                    retention_days = row[3]
                    
                    # Check if backup is expired
                    expiry_time = created_at + timedelta(days=retention_days)
                    if current_time > expiry_time:
                        # Delete backup file
                        backup_file = Path(backup_path)
                        if backup_file.exists():
                            backup_file.parent.rmdir() if backup_file.parent.exists() else None
                        
                        # Delete metadata
                        cursor.execute("DELETE FROM backup_metadata WHERE backup_id = ?", (backup_id,))
                        cleanup_count += 1
                        
                        logger.info(f"Cleaned up expired backup: {backup_id}")
                
                self.db_connection.commit()
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired backups: {e}")
            return 0
    
    async def get_replication_status(self) -> Dict[str, Any]:
        """Get comprehensive replication status"""
        try:
            # Node statistics
            node_stats = {}
            for node_id, node in self.nodes.items():
                node_stats[node_id] = {
                    'status': node.status.value,
                    'region': node.region,
                    'datacenter': node.datacenter,
                    'is_master': node.is_master,
                    'lag_ms': node.lag_ms,
                    'last_heartbeat': node.last_heartbeat.isoformat(),
                    'storage_used_gb': node.storage_used_gb,
                    'storage_capacity_gb': node.storage_capacity_gb
                }
            
            # Recent transactions
            recent_transactions = []
            for transaction in list(self.pending_transactions.values())[-10:]:
                recent_transactions.append({
                    'transaction_id': transaction.transaction_id,
                    'source_node': transaction.source_node,
                    'target_nodes': transaction.target_nodes,
                    'status': transaction.status,
                    'timestamp': transaction.timestamp.isoformat(),
                    'completed_nodes': list(transaction.completed_nodes),
                    'failed_nodes': list(transaction.failed_nodes)
                })
            
            # Backup statistics
            backup_stats = await self._get_backup_statistics()
            
            return {
                'manager_id': self.manager_id,
                'replication_strategy': self.replication_strategy.value,
                'default_consistency_level': self.default_consistency_level.value,
                'nodes': node_stats,
                'metrics': self.replication_metrics,
                'recent_transactions': recent_transactions,
                'backup_statistics': backup_stats,
                'data_records_count': len(self.data_store),
                'pending_transactions_count': len(self.pending_transactions),
                'configuration': {
                    'heartbeat_interval': self.heartbeat_interval,
                    'replication_timeout': self.replication_timeout,
                    'max_lag_tolerance_ms': self.max_lag_tolerance_ms,
                    'compression_enabled': self.compression_enabled
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get replication status: {e}")
            return {'error': str(e)}
    
    async def _get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                # Total backups
                cursor.execute("SELECT COUNT(*) FROM backup_metadata")
                total_backups = cursor.fetchone()[0]
                
                # Backup size
                cursor.execute("SELECT SUM(size_bytes) FROM backup_metadata")
                total_size = cursor.fetchone()[0] or 0
                
                # Recent backups
                recent_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute("SELECT COUNT(*) FROM backup_metadata WHERE created_at > ?", (recent_cutoff,))
                recent_backups = cursor.fetchone()[0]
                
                # Backup types
                cursor.execute("SELECT backup_type, COUNT(*) FROM backup_metadata GROUP BY backup_type")
                backup_types = dict(cursor.fetchall())
                
                return {
                    'total_backups': total_backups,
                    'total_size_bytes': total_size,
                    'total_size_gb': total_size / (1024 ** 3) if total_size else 0,
                    'recent_backups_7d': recent_backups,
                    'backup_type_distribution': backup_types
                }
                
        except Exception as e:
            logger.error(f"Failed to get backup statistics: {e}")
            return {}

# Example usage and testing
async def main():
    """Example usage of XORB Data Replication Manager"""
    try:
        print("💾 XORB Data Replication Manager initializing...")
        
        # Initialize replication manager
        replication_manager = XORBDataReplicationManager({
            'heartbeat_interval': 30,
            'replication_timeout': 60,
            'compression_enabled': True,
            'data_path': '/tmp/xorb_data',
            'backup_path': '/tmp/xorb_backups'
        })
        
        # Register nodes
        await replication_manager.register_node("node_1", "us-east-1", "dc1", "localhost", 9001, is_master=True)
        await replication_manager.register_node("node_2", "us-west-2", "dc2", "localhost", 9002)
        await replication_manager.register_node("node_3", "eu-west-1", "dc3", "localhost", 9003)
        
        print("✅ Replication nodes registered")
        
        # Store data with replication
        print("\n💾 Testing data storage and replication...")
        record_id = await replication_manager.store_data(
            data_type="agent_state",
            content={"agent_id": "test_agent", "state": "active", "performance": 0.85},
            consistency_level=ConsistencyLevel.QUORUM
        )
        
        print(f"✅ Data stored with ID: {record_id}")
        
        # Retrieve data
        retrieved_record = await replication_manager.retrieve_data(record_id, ConsistencyLevel.ONE)
        if retrieved_record:
            print(f"✅ Data retrieved: {retrieved_record.data_type}")
        
        # Create backup
        print("\n📦 Creating backup...")
        backup_id = await replication_manager.create_backup(BackupType.FULL, retention_days=30)
        print(f"✅ Backup created: {backup_id}")
        
        # Get status
        status = await replication_manager.get_replication_status()
        print(f"\n📊 Replication Status:")
        print(f"- Registered Nodes: {len(status['nodes'])}")
        print(f"- Total Operations: {status['metrics']['total_operations']}")
        print(f"- Successful Replications: {status['metrics']['successful_replications']}")
        print(f"- Data Records: {status['data_records_count']}")
        print(f"- Total Backups: {status['backup_statistics']['total_backups']}")
        print(f"- Backup Size: {status['backup_statistics']['total_size_gb']:.2f} GB")
        
        # Cleanup expired backups
        print("\n🧹 Cleaning up expired backups...")
        cleaned_count = await replication_manager.cleanup_expired_backups()
        print(f"✅ Cleaned up {cleaned_count} expired backups")
        
        print(f"\n✅ XORB Data Replication Manager demonstration completed!")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())