"""
High-Performance Database Connection Pool for Xorb 2.0
Implements intelligent connection management and caching
"""

import asyncio
import asyncpg
import aioredis
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import structlog
import time
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger(__name__)

# Metrics
db_connections_active = Gauge('xorb_db_connections_active', 'Active database connections')
db_connections_total = Counter('xorb_db_connections_total', 'Total database connections created')
db_query_duration = Histogram('xorb_db_query_duration_seconds', 'Database query duration')
db_pool_exhausted = Counter('xorb_db_pool_exhausted_total', 'Database pool exhaustion events')

class ConnectionPoolManager:
    """Advanced connection pool manager with intelligent scaling"""
    
    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str,
        min_size: int = 5,
        max_size: int = 20,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0,
        command_timeout: float = 60.0
    ):
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url
        self.min_size = min_size
        self.max_size = max_size
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        self.command_timeout = command_timeout
        
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_pool: Optional[aioredis.ConnectionPool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Connection health tracking
        self.connection_health: Dict[str, float] = {}
        self.query_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize both PostgreSQL and Redis connection pools"""
        logger.info("Initializing connection pools", 
                   postgres_min=self.min_size, postgres_max=self.max_size)
        
        # Initialize PostgreSQL pool with optimization
        self.postgres_pool = await asyncpg.create_pool(
            self.postgres_dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            max_queries=self.max_queries,
            max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
            command_timeout=self.command_timeout,
            server_settings={
                'application_name': 'xorb_v2',
                'tcp_keepalives_idle': '300',
                'tcp_keepalives_interval': '30',
                'tcp_keepalives_count': '3',
            }
        )
        
        # Initialize Redis pool
        self.redis_pool = aioredis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=20,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={
                'TCP_KEEPIDLE': 300,
                'TCP_KEEPINTVL': 30,
                'TCP_KEEPCNT': 3,
            }
        )
        
        self.redis_client = aioredis.Redis(connection_pool=self.redis_pool)
        
        # Test connections
        await self._test_connections()
        
        logger.info("Connection pools initialized successfully")
    
    async def _test_connections(self):
        """Test database connectivity"""
        # Test PostgreSQL
        async with self.postgres_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
            
        # Test Redis
        await self.redis_client.ping()
        
        logger.info("Database connections verified")
    
    @asynccontextmanager
    async def get_postgres_connection(self):
        """Get PostgreSQL connection with monitoring"""
        start_time = time.time()
        connection = None
        
        try:
            if self.postgres_pool.get_size() >= self.max_size:
                db_pool_exhausted.inc()
                logger.warning("PostgreSQL pool exhausted", 
                             current_size=self.postgres_pool.get_size())
            
            connection = await self.postgres_pool.acquire()
            db_connections_active.inc()
            db_connections_total.inc()
            
            # Track connection health
            conn_id = id(connection)
            self.connection_health[str(conn_id)] = time.time()
            
            yield connection
            
        finally:
            if connection:
                db_connections_active.dec()
                duration = time.time() - start_time
                db_query_duration.observe(duration)
                
                # Release connection back to pool
                await self.postgres_pool.release(connection)
    
    async def execute_query(
        self, 
        query: str, 
        *args, 
        cache_key: Optional[str] = None,
        cache_ttl: int = 300
    ) -> List[Dict[str, Any]]:
        """Execute query with intelligent caching"""
        
        # Check cache first
        if cache_key:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.debug("Query cache hit", cache_key=cache_key)
                return cached_result
        
        # Execute query
        async with self.get_postgres_connection() as conn:
            start_time = time.time()
            
            try:
                rows = await conn.fetch(query, *args)
                result = [dict(row) for row in rows]
                
                # Cache result if requested
                if cache_key:
                    await self._cache_result(cache_key, result, cache_ttl)
                
                query_time = time.time() - start_time
                logger.debug("Query executed", 
                           query=query[:100], 
                           duration=query_time,
                           rows_returned=len(result))
                
                return result
                
            except Exception as e:
                query_time = time.time() - start_time
                logger.error("Query execution failed", 
                           query=query[:100],
                           duration=query_time,
                           error=str(e))
                raise
    
    async def execute_transaction(self, queries: List[tuple]) -> List[Any]:
        """Execute multiple queries in a transaction"""
        async with self.get_postgres_connection() as conn:
            async with conn.transaction():
                results = []
                for query, args in queries:
                    if args:
                        result = await conn.fetch(query, *args)
                    else:
                        result = await conn.fetch(query)
                    results.append([dict(row) for row in result])
                return results
    
    async def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached query result from Redis"""
        try:
            cached_data = await self.redis_client.get(f"query_cache:{cache_key}")
            if cached_data:
                import json
                return json.loads(cached_data)
        except Exception as e:
            logger.warning("Cache retrieval failed", cache_key=cache_key, error=str(e))
        return None
    
    async def _cache_result(self, cache_key: str, result: List[Dict[str, Any]], ttl: int):
        """Cache query result in Redis"""
        try:
            import json
            serialized_data = json.dumps(result, default=str)
            await self.redis_client.setex(f"query_cache:{cache_key}", ttl, serialized_data)
            logger.debug("Query result cached", cache_key=cache_key, ttl=ttl)
        except Exception as e:
            logger.warning("Cache storage failed", cache_key=cache_key, error=str(e))
    
    async def get_redis_client(self) -> aioredis.Redis:
        """Get Redis client"""
        return self.redis_client
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        pg_stats = {
            'size': self.postgres_pool.get_size(),
            'min_size': self.postgres_pool.get_min_size(),
            'max_size': self.postgres_pool.get_max_size(),
            'free_size': self.postgres_pool.get_idle_size(),
        }
        
        redis_stats = {
            'created_connections': self.redis_pool.created_connections,
            'available_connections': len(self.redis_pool._available_connections),
            'in_use_connections': len(self.redis_pool._in_use_connections),
        }
        
        return {
            'postgres': pg_stats,
            'redis': redis_stats,
            'cache_size': len(self.query_cache),
            'healthy_connections': len([
                ts for ts in self.connection_health.values() 
                if time.time() - ts < 300
            ])
        }
    
    async def cleanup_unhealthy_connections(self):
        """Clean up unhealthy connections"""
        current_time = time.time()
        unhealthy_count = 0
        
        for conn_id, last_seen in list(self.connection_health.items()):
            if current_time - last_seen > 600:  # 10 minutes
                del self.connection_health[conn_id]
                unhealthy_count += 1
        
        if unhealthy_count > 0:
            logger.info("Cleaned up unhealthy connections", count=unhealthy_count)
    
    async def close(self):
        """Close all connection pools"""
        logger.info("Closing connection pools")
        
        if self.postgres_pool:
            await self.postgres_pool.close()
            
        if self.redis_client:
            await self.redis_client.close()
            
        if self.redis_pool:
            await self.redis_pool.disconnect()
        
        logger.info("Connection pools closed")

# Global connection pool instance
_connection_pool: Optional[ConnectionPoolManager] = None

async def get_connection_pool() -> ConnectionPoolManager:
    """Get the global connection pool instance"""
    global _connection_pool
    
    if _connection_pool is None:
        import os
        postgres_dsn = os.getenv("POSTGRES_DSN", "postgresql://xorb:password@localhost/xorb")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        _connection_pool = ConnectionPoolManager(postgres_dsn, redis_url)
        await _connection_pool.initialize()
    
    return _connection_pool

async def initialize_connection_pool():
    """Initialize the global connection pool"""
    await get_connection_pool()

async def close_connection_pool():
    """Close the global connection pool"""
    global _connection_pool
    if _connection_pool:
        await _connection_pool.close()
        _connection_pool = None