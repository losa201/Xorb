"""
Enterprise Performance Optimization
High-performance, scalable implementation for Fortune 500 deployments
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import json
import psutil
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    database_pool_usage: float = 0.0
    queue_depth: int = 0
    throughput_mbps: float = 0.0


@dataclass
class ResourcePool:
    """Resource pool configuration"""
    pool_type: str
    min_size: int
    max_size: int
    current_size: int = 0
    active_resources: int = 0
    idle_timeout: int = 300
    max_lifetime: int = 3600
    health_check_interval: int = 30


class HighPerformanceCache:
    """Enterprise-grade distributed cache"""
    
    def __init__(self, redis_urls: List[str], strategy: CacheStrategy = CacheStrategy.LRU):
        self.redis_urls = redis_urls
        self.strategy = strategy
        self.redis_pools = {}
        self.local_cache = {}
        self.cache_stats = defaultdict(int)
        self.ttl_index = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.max_local_cache_size = 10000
        
    async def initialize(self):
        """Initialize Redis connection pools"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using local cache only")
            return
        
        for i, redis_url in enumerate(self.redis_urls):
            try:
                pool = redis.ConnectionPool.from_url(
                    redis_url,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    socket_keepalive_options={}
                )
                self.redis_pools[f"pool_{i}"] = redis.Redis(connection_pool=pool)
                await self.redis_pools[f"pool_{i}"].ping()
                logger.info(f"Connected to Redis pool {i}: {redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis {redis_url}: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback strategy"""
        # Try local cache first
        if key in self.local_cache:
            self._update_access_stats(key, hit=True, source="local")
            return self.local_cache[key]
        
        # Try Redis pools
        if REDIS_AVAILABLE and self.redis_pools:
            for pool_name, redis_client in self.redis_pools.items():
                try:
                    value = await redis_client.get(key)
                    if value:
                        # Deserialize and cache locally
                        deserialized = json.loads(value.decode())
                        await self._set_local_cache(key, deserialized)
                        self._update_access_stats(key, hit=True, source="redis")
                        return deserialized
                except Exception as e:
                    logger.error(f"Redis get error for {key}: {e}")
                    continue
        
        self._update_access_stats(key, hit=False)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with replication"""
        serialized = json.dumps(value)
        
        # Set in local cache
        await self._set_local_cache(key, value, ttl)
        
        # Set in Redis pools
        if REDIS_AVAILABLE and self.redis_pools:
            success_count = 0
            for pool_name, redis_client in self.redis_pools.items():
                try:
                    await redis_client.setex(key, ttl, serialized)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Redis set error for {key}: {e}")
            
            return success_count > 0
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache layers"""
        # Delete from local cache
        self.local_cache.pop(key, None)
        self.ttl_index.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, 0)
        
        # Delete from Redis
        if REDIS_AVAILABLE and self.redis_pools:
            for pool_name, redis_client in self.redis_pools.items():
                try:
                    await redis_client.delete(key)
                except Exception as e:
                    logger.error(f"Redis delete error for {key}: {e}")
        
        return True
    
    async def _set_local_cache(self, key: str, value: Any, ttl: int = 3600):
        """Set value in local cache with eviction"""
        # Check TTL expiration
        await self._cleanup_expired_keys()
        
        # Evict if cache is full
        if len(self.local_cache) >= self.max_local_cache_size:
            await self._evict_keys()
        
        self.local_cache[key] = value
        self.ttl_index[key] = time.time() + ttl
        self.access_times[key] = time.time()
        self.access_counts[key] += 1
    
    async def _cleanup_expired_keys(self):
        """Remove expired keys from local cache"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.ttl_index.items()
            if expiry < current_time
        ]
        
        for key in expired_keys:
            self.local_cache.pop(key, None)
            self.ttl_index.pop(key, None)
            self.access_times.pop(key, None)
            self.access_counts.pop(key, 0)
    
    async def _evict_keys(self):
        """Evict keys based on strategy"""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self.access_times:
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                await self.delete(lru_key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            if self.access_counts:
                lfu_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
                await self.delete(lfu_key)
    
    def _update_access_stats(self, key: str, hit: bool, source: str = "unknown"):
        """Update cache statistics"""
        if hit:
            self.cache_stats[f"{source}_hits"] += 1
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
        else:
            self.cache_stats["misses"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = sum(self.cache_stats.values())
        hit_rate = (self.cache_stats["local_hits"] + self.cache_stats["redis_hits"]) / max(total_requests, 1)
        
        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "local_cache_size": len(self.local_cache),
            "redis_pools": len(self.redis_pools),
            "stats": dict(self.cache_stats)
        }


class DatabaseConnectionManager:
    """Enterprise database connection management"""
    
    def __init__(self, database_urls: List[str], read_replicas: List[str] = None):
        self.database_urls = database_urls
        self.read_replicas = read_replicas or []
        self.write_pools = {}
        self.read_pools = {}
        self.connection_stats = defaultdict(int)
        self.health_status = {}
        self.load_balancer = DatabaseLoadBalancer()
        
    async def initialize(self):
        """Initialize database connection pools"""
        if not ASYNCPG_AVAILABLE:
            logger.error("asyncpg not available for database connections")
            return
        
        # Initialize write pools (primary databases)
        for i, db_url in enumerate(self.database_urls):
            try:
                pool = await asyncpg.create_pool(
                    db_url,
                    min_size=5,
                    max_size=20,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300,
                    command_timeout=30
                )
                self.write_pools[f"write_pool_{i}"] = pool
                self.health_status[f"write_pool_{i}"] = True
                logger.info(f"Created write pool {i}")
            except Exception as e:
                logger.error(f"Failed to create write pool {i}: {e}")
                self.health_status[f"write_pool_{i}"] = False
        
        # Initialize read pools (read replicas)
        for i, replica_url in enumerate(self.read_replicas):
            try:
                pool = await asyncpg.create_pool(
                    replica_url,
                    min_size=3,
                    max_size=15,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300,
                    command_timeout=30
                )
                self.read_pools[f"read_pool_{i}"] = pool
                self.health_status[f"read_pool_{i}"] = True
                logger.info(f"Created read pool {i}")
            except Exception as e:
                logger.error(f"Failed to create read pool {i}: {e}")
                self.health_status[f"read_pool_{i}"] = False
    
    async def get_write_connection(self):
        """Get connection for write operations"""
        pool_name = self.load_balancer.select_write_pool(self.write_pools, self.health_status)
        if pool_name and pool_name in self.write_pools:
            try:
                connection = await self.write_pools[pool_name].acquire()
                self.connection_stats[f"{pool_name}_acquired"] += 1
                return connection, pool_name
            except Exception as e:
                logger.error(f"Failed to acquire write connection: {e}")
                self.health_status[pool_name] = False
        
        raise Exception("No healthy write pools available")
    
    async def get_read_connection(self):
        """Get connection for read operations"""
        # Try read replicas first
        if self.read_pools:
            pool_name = self.load_balancer.select_read_pool(self.read_pools, self.health_status)
            if pool_name and pool_name in self.read_pools:
                try:
                    connection = await self.read_pools[pool_name].acquire()
                    self.connection_stats[f"{pool_name}_acquired"] += 1
                    return connection, pool_name
                except Exception as e:
                    logger.error(f"Failed to acquire read connection: {e}")
                    self.health_status[pool_name] = False
        
        # Fallback to write pools
        return await self.get_write_connection()
    
    async def release_connection(self, connection, pool_name: str):
        """Release connection back to pool"""
        try:
            if pool_name in self.write_pools:
                await self.write_pools[pool_name].release(connection)
            elif pool_name in self.read_pools:
                await self.read_pools[pool_name].release(connection)
            
            self.connection_stats[f"{pool_name}_released"] += 1
        except Exception as e:
            logger.error(f"Failed to release connection: {e}")
    
    async def health_check(self):
        """Check health of all database pools"""
        for pool_name, pool in {**self.write_pools, **self.read_pools}.items():
            try:
                async with pool.acquire() as connection:
                    await connection.execute("SELECT 1")
                self.health_status[pool_name] = True
            except Exception as e:
                logger.error(f"Health check failed for {pool_name}: {e}")
                self.health_status[pool_name] = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        pool_stats = {}
        
        for pool_name, pool in {**self.write_pools, **self.read_pools}.items():
            pool_stats[pool_name] = {
                "size": pool.get_size(),
                "idle": pool.get_idle_size(),
                "healthy": self.health_status.get(pool_name, False)
            }
        
        return {
            "pools": pool_stats,
            "connection_stats": dict(self.connection_stats)
        }


class DatabaseLoadBalancer:
    """Load balancer for database connections"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED):
        self.strategy = strategy
        self.connection_counts = defaultdict(int)
        self.last_used = defaultdict(float)
        
    def select_write_pool(self, pools: Dict[str, Any], health_status: Dict[str, bool]) -> Optional[str]:
        """Select write pool based on strategy"""
        healthy_pools = [name for name, healthy in health_status.items() if healthy and name in pools]
        
        if not healthy_pools:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return min(healthy_pools, key=lambda x: self.last_used[x])
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_pools, key=lambda x: self.connection_counts[x])
        
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            # Prefer pools with better performance
            return healthy_pools[0]  # Simplified - would include more metrics
        
        return healthy_pools[0]
    
    def select_read_pool(self, pools: Dict[str, Any], health_status: Dict[str, bool]) -> Optional[str]:
        """Select read pool based on strategy"""
        return self.select_write_pool(pools, health_status)


class AsyncTaskQueue:
    """High-performance async task queue"""
    
    def __init__(self, max_workers: int = 100, max_queue_size: int = 10000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_queue = asyncio.Queue(maxsize=max_queue_size)
        self.workers = []
        self.worker_stats = defaultdict(int)
        self.task_stats = defaultdict(int)
        self.running = False
        
    async def start(self):
        """Start task queue workers"""
        self.running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} task queue workers")
    
    async def stop(self):
        """Stop task queue workers"""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Task queue workers stopped")
    
    async def enqueue_task(self, task_func: Callable, *args, **kwargs) -> bool:
        """Enqueue task for async execution"""
        try:
            task_item = {
                "func": task_func,
                "args": args,
                "kwargs": kwargs,
                "enqueued_at": time.time()
            }
            
            await self.task_queue.put(task_item)
            self.task_stats["enqueued"] += 1
            return True
        
        except asyncio.QueueFull:
            self.task_stats["queue_full"] += 1
            return False
    
    async def _worker(self, worker_id: str):
        """Worker coroutine for processing tasks"""
        while self.running:
            try:
                # Get task with timeout
                task_item = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                start_time = time.time()
                
                # Execute task
                try:
                    if asyncio.iscoroutinefunction(task_item["func"]):
                        await task_item["func"](*task_item["args"], **task_item["kwargs"])
                    else:
                        task_item["func"](*task_item["args"], **task_item["kwargs"])
                    
                    self.worker_stats[f"{worker_id}_completed"] += 1
                    self.task_stats["completed"] += 1
                
                except Exception as e:
                    logger.error(f"Task execution error in {worker_id}: {e}")
                    self.worker_stats[f"{worker_id}_errors"] += 1
                    self.task_stats["errors"] += 1
                
                # Update timing stats
                execution_time = time.time() - start_time
                self.worker_stats[f"{worker_id}_total_time"] += execution_time
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task queue statistics"""
        return {
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_workers": len(self.workers),
            "task_stats": dict(self.task_stats),
            "worker_stats": dict(self.worker_stats)
        }


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 90.0,
            "response_time": 2.0,
            "error_rate": 5.0,
            "cache_hit_rate": 70.0
        }
        self.alert_handlers = []
        self.monitoring_active = False
        
    async def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Network metrics
        network = psutil.net_io_counters()
        throughput_mbps = (network.bytes_sent + network.bytes_recv) / (1024 * 1024)
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            throughput_mbps=throughput_mbps,
            # Additional metrics would be collected from cache, database, etc.
        )
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.average_response_time > self.alert_thresholds["response_time"]:
            alerts.append(f"High response time: {metrics.average_response_time:.1f}s")
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert_message: str):
        """Send performance alert"""
        logger.warning(f"PERFORMANCE ALERT: {alert_message}")
        
        for handler in self.alert_handlers:
            try:
                await handler(alert_message)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get latest metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for specified minutes"""
        # Return last N metrics (simplified)
        return list(self.metrics_history[-minutes*6:])  # 6 metrics per minute


class EnterprisePerformanceManager:
    """Main performance management orchestrator"""
    
    def __init__(self):
        self.cache = None
        self.db_manager = None
        self.task_queue = None
        self.performance_monitor = None
        self.optimization_strategies = {}
        
    async def initialize(self, config: Dict[str, Any]):
        """Initialize all performance components"""
        
        # Initialize high-performance cache
        redis_urls = config.get("redis_urls", ["redis://localhost:6379"])
        self.cache = HighPerformanceCache(redis_urls)
        await self.cache.initialize()
        
        # Initialize database connection manager
        db_urls = config.get("database_urls", ["postgresql://localhost/xorb"])
        read_replicas = config.get("read_replicas", [])
        self.db_manager = DatabaseConnectionManager(db_urls, read_replicas)
        await self.db_manager.initialize()
        
        # Initialize async task queue
        max_workers = config.get("max_workers", 100)
        self.task_queue = AsyncTaskQueue(max_workers=max_workers)
        await self.task_queue.start()
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        await self.performance_monitor.start_monitoring()
        
        # Set up uvloop if available
        if UVLOOP_AVAILABLE:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for improved async performance")
        
        logger.info("Enterprise performance manager initialized")
    
    async def shutdown(self):
        """Shutdown all performance components"""
        if self.task_queue:
            await self.task_queue.stop()
        
        if self.performance_monitor:
            await self.performance_monitor.stop_monitoring()
        
        if self.db_manager:
            # Close database pools
            for pool in {**self.db_manager.write_pools, **self.db_manager.read_pools}.values():
                await pool.close()
        
        logger.info("Enterprise performance manager shutdown complete")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "cache_stats": self.cache.get_stats() if self.cache else {},
            "database_stats": self.db_manager.get_stats() if self.db_manager else {},
            "task_queue_stats": self.task_queue.get_stats() if self.task_queue else {},
            "current_metrics": self.performance_monitor.get_current_metrics().__dict__ if self.performance_monitor else {}
        }
        
        return stats


# Global performance manager instance
performance_manager: Optional[EnterprisePerformanceManager] = None


async def get_performance_manager() -> EnterprisePerformanceManager:
    """Get global performance manager"""
    global performance_manager
    if performance_manager is None:
        performance_manager = EnterprisePerformanceManager()
        
        # Default configuration
        config = {
            "redis_urls": [os.getenv("REDIS_URL", "redis://localhost:6379")],
            "database_urls": [os.getenv("DATABASE_URL", "postgresql://localhost/xorb")],
            "read_replicas": [],
            "max_workers": int(os.getenv("MAX_WORKERS", "100"))
        }
        
        await performance_manager.initialize(config)
    
    return performance_manager


# Performance optimization utilities
async def optimize_query_performance(query: str, params: tuple = None) -> str:
    """Optimize database query performance"""
    # Add query hints and optimizations
    optimized_query = query
    
    # Add query caching hints
    if "SELECT" in query.upper():
        query_hash = hashlib.md5(f"{query}{params}".encode()).hexdigest()
        optimized_query = f"/* QUERY_ID: {query_hash} */ {query}"
    
    return optimized_query


async def cache_result(key: str, result: Any, ttl: int = 3600) -> bool:
    """Cache result with performance manager"""
    manager = await get_performance_manager()
    if manager.cache:
        return await manager.cache.set(key, result, ttl)
    return False


async def get_cached_result(key: str) -> Optional[Any]:
    """Get cached result with performance manager"""
    manager = await get_performance_manager()
    if manager.cache:
        return await manager.cache.get(key)
    return None