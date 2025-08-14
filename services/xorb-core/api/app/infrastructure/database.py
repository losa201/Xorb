"""
Optimized database connection and management with performance enhancements.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import text

from ..domain.exceptions import DatabaseConnectionError


logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration with performance optimizations."""

    def __init__(self):
        self.database_url = self._get_database_url()
        self.min_pool_size = int(os.getenv("DB_MIN_POOL_SIZE", "5"))
        self.max_pool_size = int(os.getenv("DB_MAX_POOL_SIZE", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.command_timeout = int(os.getenv("DB_COMMAND_TIMEOUT", "60"))
        self.statement_cache_size = int(os.getenv("DB_STATEMENT_CACHE_SIZE", "100"))
        self.enable_prepared_statements = os.getenv("DB_ENABLE_PREPARED_STATEMENTS", "true").lower() == "true"
        self.enable_query_logging = os.getenv("DB_ENABLE_QUERY_LOGGING", "false").lower() == "true"

        # Connection pool recycling
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
        self.pool_pre_ping = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"

    def _get_database_url(self) -> str:
        """Get database URL from environment with fallbacks."""
        return os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://xorb:changeme@localhost:5432/xorb"
        )

    def get_asyncpg_url(self) -> str:
        """Get URL for direct asyncpg connections."""
        url = self.database_url
        if url.startswith("postgresql+asyncpg://"):
            return url.replace("postgresql+asyncpg://", "postgresql://")
        return url


# Global instances
_config = DatabaseConfig()
_connection_pool: Optional[asyncpg.Pool] = None
_async_engine = None
_async_session_factory = None


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return _config


async def get_database_pool() -> asyncpg.Pool:
    """Get or create optimized asyncpg connection pool."""
    global _connection_pool

    if _connection_pool is None:
        try:
            database_url = _config.get_asyncpg_url()

            # Connection initialization for performance
            async def init_connection(conn):
                # Set connection parameters for performance
                await conn.execute("SET application_name = 'xorb-api'")
                await conn.execute("SET statement_timeout = '60s'")
                await conn.execute("SET idle_in_transaction_session_timeout = '10min'")

                # Enable prepared statement caching
                if _config.enable_prepared_statements:
                    await conn.execute("SET plan_cache_mode = 'force_generic_plan'")

                # Register pgvector extension if available
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    logger.info("pgvector extension enabled")
                except Exception as e:
                    logger.warning(f"Could not enable pgvector: {e}")

            _connection_pool = await asyncpg.create_pool(
                database_url,
                min_size=_config.min_pool_size,
                max_size=_config.max_pool_size,
                command_timeout=_config.command_timeout,
                init=init_connection,
                statement_cache_size=_config.statement_cache_size
            )

            logger.info(f"Created database pool: {_config.min_pool_size}-{_config.max_pool_size} connections")

        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise DatabaseConnectionError(f"Failed to create database pool: {e}")

    return _connection_pool


def get_async_engine():
    """Get or create SQLAlchemy async engine with optimizations."""
    global _async_engine

    if _async_engine is None:
        # Engine configuration for performance
        engine_kwargs = {
            "echo": _config.enable_query_logging,
            "future": True,
            "pool_class": QueuePool,
            "pool_size": _config.max_pool_size,
            "max_overflow": 10,
            "pool_timeout": _config.pool_timeout,
            "pool_recycle": _config.pool_recycle,
            "pool_pre_ping": _config.pool_pre_ping,

            # Connection arguments for asyncpg
            "connect_args": {
                "statement_cache_size": _config.statement_cache_size,
                "command_timeout": _config.command_timeout,
                "prepared_statement_cache_size": _config.statement_cache_size,
            }
        }

        _async_engine = create_async_engine(
            _config.database_url,
            **engine_kwargs
        )

        logger.info("Created optimized SQLAlchemy async engine")

    return _async_engine


def get_async_session_factory():
    """Get SQLAlchemy async session factory."""
    global _async_session_factory

    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    return _async_session_factory


@asynccontextmanager
async def get_async_session():
    """Get SQLAlchemy async session context manager."""
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_database_pool():
    """Close database connection pool and engine."""
    global _connection_pool, _async_engine

    if _connection_pool:
        await _connection_pool.close()
        _connection_pool = None
        logger.info("Closed database connection pool")

    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        logger.info("Closed SQLAlchemy async engine")


@asynccontextmanager
async def get_database_connection():
    """Context manager for direct asyncpg connections."""
    pool = await get_database_pool()
    async with pool.acquire() as connection:
        yield connection


async def execute_query(query: str, *args):
    """Execute a query and return results."""
    async with get_database_connection() as conn:
        return await conn.fetch(query, *args)


async def execute_command(query: str, *args):
    """Execute a command (INSERT, UPDATE, DELETE)."""
    async with get_database_connection() as conn:
        return await conn.execute(query, *args)


async def check_database_connection() -> bool:
    """Check if database connection is working."""
    try:
        async with get_database_connection() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


# Database lifespan management
@asynccontextmanager
async def database_lifespan():
    """Database lifespan context manager for FastAPI."""
    try:
        # Initialize database pool and engine
        await get_database_pool()
        get_async_engine()
        logger.info("Database connections initialized")
        yield
    finally:
        # Close database connections
        await close_database_pool()
        logger.info("Database connections closed")


# Performance monitoring
async def get_database_stats() -> Dict[str, Any]:
    """Get database connection pool statistics."""
    pool = await get_database_pool()

    return {
        "pool_size": pool.get_size(),
        "pool_min_size": pool.get_min_size(),
        "pool_max_size": pool.get_max_size(),
        "pool_idle_size": pool.get_idle_size(),
        "config": {
            "min_pool_size": _config.min_pool_size,
            "max_pool_size": _config.max_pool_size,
            "pool_timeout": _config.pool_timeout,
            "command_timeout": _config.command_timeout,
            "statement_cache_size": _config.statement_cache_size,
            "enable_prepared_statements": _config.enable_prepared_statements
        }
    }
