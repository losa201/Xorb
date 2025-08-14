"""
Database connection and session management for PostgreSQL
"""

import os
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import time

from .database_models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'postgresql+asyncpg://xorb:xorb@localhost:5432/xorb'
        )
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._initialized = False

    async def initialize(self):
        """Initialize database engine and session factory"""
        if self._initialized:
            return

        try:
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.database_url,
                echo=os.getenv('SQL_ECHO', 'false').lower() == 'true',
                pool_size=int(os.getenv('DB_POOL_SIZE', '20')),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '30')),
                pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
                pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '3600')),
                pool_pre_ping=True,  # Validate connections before use
                connect_args={
                    "server_settings": {
                        "application_name": "xorb_api",
                        "jit": "off",  # Disable JIT compilation for faster queries
                    }
                }
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )

            self._initialized = True
            logger.info("Database connection initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def create_all_tables(self):
        """Create all database tables"""
        if not self.engine:
            await self.initialize()

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    async def drop_all_tables(self):
        """Drop all database tables (use with caution)"""
        if not self.engine:
            await self.initialize()

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        if not self._initialized:
            await self.initialize()

        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

    async def health_check(self) -> dict:
        """Check database connectivity and health"""
        if not self.engine:
            return {"status": "unhealthy", "error": "Database not initialized"}

        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()

            return {
                "status": "healthy",
                "database_url": self.database_url.split('@')[1] if '@' in self.database_url else "hidden",
                "pool_size": self.engine.pool.size(),
                "checked_in": self.engine.pool.checkedin(),
                "checked_out": self.engine.pool.checkedout(),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session (dependency injection compatible)"""
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session


# Alias for backward compatibility
get_async_session = get_database_session


async def initialize_database():
    """Initialize database for the application"""
    db_manager = get_database_manager()
    await db_manager.initialize()


async def create_tables():
    """Create all database tables"""
    db_manager = get_database_manager()
    await db_manager.create_all_tables()


async def get_database_health() -> dict:
    """Get database health status"""
    db_manager = get_database_manager()
    return await db_manager.health_check()


async def get_database_stats() -> dict:
    """Get database statistics"""
    db_manager = get_database_manager()
    return await db_manager.health_check()


# Initialize database function alias
async def init_database():
    """Initialize database - alias for compatibility"""
    await initialize_database()


async def check_database_connection() -> bool:
    """Check database connection"""
    try:
        health = await get_database_health()
        return health.get("status") == "healthy"
    except Exception:
        return False


def get_database_connection():
    """Get database connection (sync version)"""
    return get_database_manager()


class ProductionDatabaseManager:
    """Enhanced database manager for production use with additional features"""

    def __init__(self, database_url: str = None):
        self.db_manager = DatabaseManager(database_url)
        self._performance_stats = {
            "queries_executed": 0,
            "total_query_time": 0.0,
            "slow_queries": 0,
            "connection_errors": 0
        }

    async def initialize(self):
        """Initialize with production optimizations"""
        await self.db_manager.initialize()

        # Enable query performance tracking
        if self.db_manager.engine:
            from sqlalchemy import event

            @event.listens_for(self.db_manager.engine.sync_engine, "before_cursor_execute")
            def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                context._query_start_time = time.time()

            @event.listens_for(self.db_manager.engine.sync_engine, "after_cursor_execute")
            def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                total = time.time() - context._query_start_time
                self._performance_stats["queries_executed"] += 1
                self._performance_stats["total_query_time"] += total

                # Track slow queries (>100ms)
                if total > 0.1:
                    self._performance_stats["slow_queries"] += 1
                    logger.warning(f"Slow query detected ({total:.3f}s): {statement[:100]}...")

    async def get_repository_session(self):
        """Get session specifically for repository use"""
        return self.db_manager.get_session()

    def get_performance_stats(self) -> dict:
        """Get database performance statistics"""
        return {
            **self._performance_stats,
            "average_query_time": (
                self._performance_stats["total_query_time"] / max(1, self._performance_stats["queries_executed"])
            ),
            "slow_query_percentage": (
                (self._performance_stats["slow_queries"] / max(1, self._performance_stats["queries_executed"])) * 100
            )
        }

    async def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            import subprocess
            import urllib.parse

            # Parse database URL for pg_dump
            parsed = urllib.parse.urlparse(self.db_manager.database_url.replace('+asyncpg', ''))

            cmd = [
                'pg_dump',
                f'--host={parsed.hostname}',
                f'--port={parsed.port or 5432}',
                f'--username={parsed.username}',
                f'--dbname={parsed.path[1:]}',  # Remove leading slash
                f'--file={backup_path}',
                '--format=custom',
                '--no-password',
                '--verbose'
            ]

            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Database backup created successfully: {backup_path}")
                return True
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Database backup error: {e}")
            return False

    async def optimize_database(self):
        """Run database optimization tasks"""
        try:
            async with self.db_manager.get_session() as session:
                # Analyze tables for query optimization
                await session.execute(text("ANALYZE;"))

                # Update table statistics
                await session.execute(text("VACUUM ANALYZE;"))

                logger.info("Database optimization completed")

        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
