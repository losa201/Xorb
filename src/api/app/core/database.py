"""
Production-grade database management with connection pooling and optimizations
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable, TypeVar
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import json

import asyncpg
from asyncpg import Pool, Connection, Record
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import event, text
from sqlalchemy.pool import NullPool
import structlog

from .logging import get_logger, database_logger
from .metrics import get_metrics_service, timed_operation

logger = get_logger(__name__)
T = TypeVar('T')


class DatabaseBackend(Enum):
    """Database backend types"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    # Connection settings
    database_url: str = "postgresql://user:pass@localhost/dbname"
    backend: DatabaseBackend = DatabaseBackend.POSTGRESQL
    
    # Connection pool settings
    min_pool_size: int = 5
    max_pool_size: int = 20
    pool_timeout: int = 30
    pool_recycle_seconds: int = 3600
    pool_pre_ping: bool = True
    
    # Query settings
    command_timeout: int = 60
    statement_cache_size: int = 1024
    max_cached_statement_lifetime: int = 300
    
    # Performance settings
    enable_query_logging: bool = False
    slow_query_threshold_seconds: float = 1.0
    enable_query_metrics: bool = True
    enable_connection_metrics: bool = True
    
    # Health check settings
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5
    
    # Migration settings
    enable_auto_migration: bool = False
    migration_timeout_seconds: int = 300


class DatabaseHealthMonitor:
    """Monitor database health and performance"""
    
    def __init__(self, pool: Pool, config: DatabaseConfig):
        self.pool = pool
        self.config = config
        self.metrics_service = get_metrics_service()
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_health_check = 0
        self.health_status = True
    
    async def start_monitoring(self):
        """Start health monitoring task"""
        if self.monitoring_task:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Database health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info("Database health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check failed", error=str(e))
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _perform_health_check(self):
        """Perform database health check"""
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                await asyncio.wait_for(
                    conn.fetchval("SELECT 1"),
                    timeout=self.config.health_check_timeout_seconds
                )
            
            self.health_status = True
            health_check_duration = time.time() - start_time
            
            # Record metrics
            if self.metrics_service:
                self.metrics_service.custom_metrics.record_histogram(
                    "database_health_check_duration_seconds",
                    health_check_duration
                )
                self.metrics_service.custom_metrics.set_gauge(
                    "database_health_status",
                    1.0,
                    {"status": "healthy"}
                )
            
            self.last_health_check = time.time()
            
        except Exception as e:
            self.health_status = False
            health_check_duration = time.time() - start_time
            
            database_logger.error("Database health check failed", 
                                error=str(e),
                                duration=health_check_duration)
            
            # Record failure metrics
            if self.metrics_service:
                self.metrics_service.custom_metrics.record_histogram(
                    "database_health_check_duration_seconds",
                    health_check_duration,
                    {"status": "error"}
                )
                self.metrics_service.custom_metrics.set_gauge(
                    "database_health_status",
                    0.0,
                    {"status": "unhealthy"}
                )
    
    def get_health_info(self) -> Dict[str, Any]:
        """Get current health information"""
        pool_stats = {
            "size": self.pool.get_size(),
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool.get_max_size(),
            "idle_connections": self.pool.get_idle_size(),
        }
        
        return {
            "healthy": self.health_status,
            "last_check": self.last_health_check,
            "pool_stats": pool_stats,
            "check_interval": self.config.health_check_interval_seconds
        }


class QueryProfiler:
    """Profile and analyze database queries"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.metrics_service = get_metrics_service()
        self.slow_queries: List[Dict[str, Any]] = []
        self.max_slow_queries = 100
    
    def profile_query(self, query: str, duration: float, success: bool = True):
        """Profile a database query"""
        # Log slow queries
        if duration > self.config.slow_query_threshold_seconds:
            slow_query_info = {
                "query": query[:500],  # Truncate long queries
                "duration": duration,
                "timestamp": time.time(),
                "success": success
            }
            
            self.slow_queries.append(slow_query_info)
            
            # Keep only recent slow queries
            if len(self.slow_queries) > self.max_slow_queries:
                self.slow_queries = self.slow_queries[-self.max_slow_queries:]
            
            database_logger.warning("Slow query detected",
                                  query=query[:200],
                                  duration=duration,
                                  success=success)
        
        # Record metrics
        if self.config.enable_query_metrics and self.metrics_service:
            operation = self._extract_operation(query)
            
            self.metrics_service.custom_metrics.record_histogram(
                "database_query_duration_seconds",
                duration,
                {"operation": operation, "success": str(success)}
            )
            
            self.metrics_service.custom_metrics.increment_counter(
                "database_queries_total",
                1,
                {"operation": operation, "success": str(success)}
            )
    
    def _extract_operation(self, query: str) -> str:
        """Extract operation type from SQL query"""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            return 'select'
        elif query_lower.startswith('insert'):
            return 'insert'
        elif query_lower.startswith('update'):
            return 'update'
        elif query_lower.startswith('delete'):
            return 'delete'
        elif query_lower.startswith('create'):
            return 'create'
        elif query_lower.startswith('drop'):
            return 'drop'
        elif query_lower.startswith('alter'):
            return 'alter'
        else:
            return 'other'
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent slow queries"""
        return sorted(
            self.slow_queries,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]


class DatabaseManager:
    """Production database manager with advanced features"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[Pool] = None
        self.sqlalchemy_engine = None
        self.session_factory = None
        self.health_monitor: Optional[DatabaseHealthMonitor] = None
        self.query_profiler = QueryProfiler(config)
        self.connection_count = 0
        
        database_logger.info("Database manager initialized", config=config.__dict__)
    
    async def initialize(self):
        """Initialize database connections and services"""
        try:
            await self._create_connection_pool()
            await self._setup_sqlalchemy()
            
            # Start health monitoring
            if self.pool:
                self.health_monitor = DatabaseHealthMonitor(self.pool, self.config)
                await self.health_monitor.start_monitoring()
            
            database_logger.info("Database manager initialized successfully")
            
        except Exception as e:
            database_logger.error("Database initialization failed", error=str(e))
            raise
    
    async def _create_connection_pool(self):
        """Create asyncpg connection pool"""
        if self.config.backend != DatabaseBackend.POSTGRESQL:
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.command_timeout,
                statement_cache_size=self.config.statement_cache_size,
                max_cached_statement_lifetime=self.config.max_cached_statement_lifetime,
                setup=self._setup_connection
            )
            
            database_logger.info("Connection pool created",
                                min_size=self.config.min_pool_size,
                                max_size=self.config.max_pool_size)
            
        except Exception as e:
            database_logger.error("Failed to create connection pool", error=str(e))
            raise
    
    async def _setup_connection(self, conn: Connection):
        """Setup individual database connection"""
        # Set connection-level settings
        await conn.execute("SET timezone = 'UTC'")
        await conn.execute("SET statement_timeout = $1", self.config.command_timeout * 1000)
        
        # Enable connection logging if configured
        if self.config.enable_query_logging:
            conn.add_log_listener(self._log_query)
    
    def _log_query(self, conn, query, args, timeout, elapsed):
        """Log database queries"""
        database_logger.debug("Query executed",
                             query=query[:200],
                             duration=elapsed,
                             timeout=timeout)
    
    async def _setup_sqlalchemy(self):
        """Setup SQLAlchemy async engine and session factory"""
        engine_kwargs = {
            "poolclass": NullPool,  # Use asyncpg pool instead
            "echo": self.config.enable_query_logging,
        }
        
        if self.config.backend == DatabaseBackend.POSTGRESQL:
            # Use asyncpg for PostgreSQL
            sqlalchemy_url = self.config.database_url.replace('postgresql://', 'postgresql+asyncpg://')
        else:
            # Use aiosqlite for SQLite
            sqlalchemy_url = self.config.database_url.replace('sqlite://', 'sqlite+aiosqlite://')
        
        self.sqlalchemy_engine = create_async_engine(sqlalchemy_url, **engine_kwargs)
        
        self.session_factory = async_sessionmaker(
            self.sqlalchemy_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        database_logger.info("SQLAlchemy engine created")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        start_time = time.time()
        connection = None
        
        try:
            connection = await self.pool.acquire()
            self.connection_count += 1
            
            # Record connection metrics
            if self.config.enable_connection_metrics and self.query_profiler.metrics_service:
                self.query_profiler.metrics_service.custom_metrics.increment_counter(
                    "database_connections_acquired",
                    1
                )
            
            yield connection
            
        except Exception as e:
            database_logger.error("Database connection error", error=str(e))
            raise
        
        finally:
            if connection:
                await self.pool.release(connection)
                self.connection_count -= 1
                
                connection_duration = time.time() - start_time
                
                # Record connection duration
                if self.config.enable_connection_metrics and self.query_profiler.metrics_service:
                    self.query_profiler.metrics_service.custom_metrics.record_histogram(
                        "database_connection_duration_seconds",
                        connection_duration
                    )
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get SQLAlchemy async session"""
        if not self.session_factory:
            raise RuntimeError("Session factory not initialized")
        
        session = self.session_factory()
        
        try:
            yield session
            await session.commit()
            
        except Exception as e:
            await session.rollback()
            database_logger.error("Session error", error=str(e))
            raise
        
        finally:
            await session.close()
    
    @timed_operation("database_query")
    async def execute_query(
        self,
        query: str,
        *args,
        fetch_mode: str = "all"
    ) -> Any:
        """Execute database query with profiling"""
        start_time = time.time()
        success = True
        result = None
        
        try:
            async with self.get_connection() as conn:
                if fetch_mode == "all":
                    result = await conn.fetch(query, *args)
                elif fetch_mode == "one":
                    result = await conn.fetchrow(query, *args)
                elif fetch_mode == "val":
                    result = await conn.fetchval(query, *args)
                else:  # execute
                    result = await conn.execute(query, *args)
                
                return result
                
        except Exception as e:
            success = False
            database_logger.error("Query execution failed",
                                query=query[:200],
                                error=str(e))
            raise
        
        finally:
            duration = time.time() - start_time
            self.query_profiler.profile_query(query, duration, success)
    
    async def execute_transaction(self, operations: List[Callable]):
        """Execute multiple operations in a transaction"""
        start_time = time.time()
        success = True
        
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    for operation in operations:
                        await operation(conn)
                        
        except Exception as e:
            success = False
            database_logger.error("Transaction failed", error=str(e))
            raise
        
        finally:
            duration = time.time() - start_time
            self.query_profiler.profile_query("TRANSACTION", duration, success)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get database health status"""
        base_health = {
            "database_type": self.config.backend.value,
            "connection_count": self.connection_count,
            "pool_initialized": self.pool is not None,
            "sqlalchemy_initialized": self.sqlalchemy_engine is not None
        }
        
        if self.health_monitor:
            base_health.update(self.health_monitor.get_health_info())
        
        return base_health
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "slow_queries": self.query_profiler.get_slow_queries(),
            "slow_query_threshold": self.config.slow_query_threshold_seconds,
            "query_logging_enabled": self.config.enable_query_logging,
            "metrics_enabled": self.config.enable_query_metrics
        }
    
    async def close(self):
        """Close database connections"""
        try:
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
            
            if self.pool:
                await self.pool.close()
                database_logger.info("Connection pool closed")
            
            if self.sqlalchemy_engine:
                await self.sqlalchemy_engine.dispose()
                database_logger.info("SQLAlchemy engine disposed")
                
        except Exception as e:
            database_logger.error("Error closing database connections", error=str(e))


# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> Optional[DatabaseManager]:
    """Get global database manager instance"""
    return _database_manager


def setup_database(config: DatabaseConfig) -> DatabaseManager:
    """Setup global database manager"""
    global _database_manager
    _database_manager = DatabaseManager(config)
    return _database_manager


# Dependency for getting database session
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting database session"""
    db_manager = get_database_manager()
    if not db_manager:
        raise RuntimeError("Database manager not initialized")
    
    async with db_manager.get_session() as session:
        yield session


# Dependency for getting raw database connection
async def get_db_connection() -> AsyncGenerator[Connection, None]:
    """FastAPI dependency for getting raw database connection"""
    db_manager = get_database_manager()
    if not db_manager:
        raise RuntimeError("Database manager not initialized")
    
    async with db_manager.get_connection() as connection:
        yield connection