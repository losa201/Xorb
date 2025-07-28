"""
XORB Infrastructure - Database Management

Centralized database connection management with async support.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

try:
    import aioredis
except ImportError:
    aioredis = None

try:
    from sqlalchemy import create_engine
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import sessionmaker
except ImportError:
    create_engine = None
    create_async_engine = None
    AsyncSession = None
    async_sessionmaker = None
    sessionmaker = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

try:
    import qdrant_client
    from qdrant_client.http import models
except ImportError:
    qdrant_client = None
    models = None

from domains.core import config
from domains.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages all database connections for XORB."""

    def __init__(self):
        self._postgres_engine = None
        self._postgres_async_engine = None
        self._redis_pool = None
        self._neo4j_driver = None
        self._qdrant_client = None
        self._session_factory = None
        self._async_session_factory = None

    async def initialize(self):
        """Initialize all database connections."""
        await self._init_postgres()
        await self._init_redis()
        await self._init_neo4j()
        await self._init_qdrant()
        logger.info("Database manager initialized")

    async def _init_postgres(self):
        """Initialize PostgreSQL connections."""
        if not create_engine or not create_async_engine:
            logger.warning("SQLAlchemy not available, skipping PostgreSQL initialization")
            return
        try:
            # Sync engine for migrations
            self._postgres_engine = create_engine(
                config.database.postgres_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            # Async engine for application use
            async_url = config.database.postgres_url.replace("postgresql://", "postgresql+asyncpg://")
            self._postgres_async_engine = create_async_engine(
                async_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            self._session_factory = sessionmaker(bind=self._postgres_engine)
            self._async_session_factory = async_sessionmaker(
                bind=self._postgres_async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            logger.info("PostgreSQL connections initialized")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize PostgreSQL: {e}")

    async def _init_redis(self):
        """Initialize Redis connection."""
        if not aioredis:
            logger.warning("aioredis not available, skipping Redis initialization")
            return
        try:
            self._redis_pool = aioredis.ConnectionPool.from_url(
                config.database.redis_url,
                max_connections=20,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )

            # Test connection
            redis = aioredis.Redis(connection_pool=self._redis_pool)
            await redis.ping()
            logger.info("Redis connection initialized")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Redis: {e}")

    async def _init_neo4j(self):
        """Initialize Neo4j connection."""
        if not GraphDatabase:
            logger.warning("Neo4j driver not available, skipping Neo4j initialization")
            return
        try:
            self._neo4j_driver = GraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_user, config.database.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )

            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self._neo4j_driver.verify_connectivity
            )
            logger.info("Neo4j connection initialized")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Neo4j: {e}")

    async def _init_qdrant(self):
        """Initialize Qdrant connection."""
        if not qdrant_client:
            logger.warning("Qdrant client not available, skipping Qdrant initialization")
            return
        try:
            self._qdrant_client = qdrant_client.QdrantClient(
                host=config.database.qdrant_host,
                port=config.database.qdrant_port,
                timeout=60
            )

            # Test connection
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self._qdrant_client.get_collections
            )
            logger.info("Qdrant connection initialized")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Qdrant: {e}")

    @asynccontextmanager
    async def get_postgres_session(self):
        """Get async PostgreSQL session."""
        if not self._async_session_factory:
            raise ConfigurationError("PostgreSQL not initialized")

        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_redis(self):
        """Get Redis client."""
        if not aioredis or not self._redis_pool:
            raise ConfigurationError("Redis not initialized or aioredis not available")
        return aioredis.Redis(connection_pool=self._redis_pool)

    def get_neo4j_session(self):
        """Get Neo4j session."""
        if not self._neo4j_driver:
            raise ConfigurationError("Neo4j not initialized")
        return self._neo4j_driver.session()

    def get_qdrant_client(self):
        """Get Qdrant client."""
        if not qdrant_client or not self._qdrant_client:
            raise ConfigurationError("Qdrant not initialized or qdrant_client not available")
        return self._qdrant_client

    async def health_check(self) -> dict[str, bool]:
        """Check health of all database connections."""
        health = {}

        # PostgreSQL
        try:
            async with self.get_postgres_session() as session:
                await session.execute("SELECT 1")
            health['postgres'] = True
        except Exception:
            health['postgres'] = False

        # Redis
        try:
            redis = await self.get_redis()
            await redis.ping()
            health['redis'] = True
        except Exception:
            health['redis'] = False

        # Neo4j
        try:
            with self.get_neo4j_session() as session:
                session.run("RETURN 1")
            health['neo4j'] = True
        except Exception:
            health['neo4j'] = False

        # Qdrant
        try:
            client = self.get_qdrant_client()
            await asyncio.get_event_loop().run_in_executor(
                None, client.get_collections
            )
            health['qdrant'] = True
        except Exception:
            health['qdrant'] = False

        return health

    async def close(self):
        """Close all database connections."""
        if self._postgres_async_engine:
            await self._postgres_async_engine.dispose()

        if self._postgres_engine:
            self._postgres_engine.dispose()

        if self._redis_pool:
            await self._redis_pool.disconnect()

        if self._neo4j_driver:
            await asyncio.get_event_loop().run_in_executor(
                None, self._neo4j_driver.close
            )

        logger.info("Database connections closed")


# Global database manager
db_manager = DatabaseManager()
