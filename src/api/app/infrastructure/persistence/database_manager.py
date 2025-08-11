"""
Database Manager - Clean Architecture Infrastructure
Manages database connections, migrations, and health
"""

import logging
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Connection, Pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Production-ready database manager with connection pooling and health monitoring.
    Provides clean abstraction over database operations.
    """
    
    def __init__(self, database_url: str, config: Optional[Dict[str, Any]] = None):
        self.database_url = database_url
        self.config = config or {}
        self._pool: Optional[Pool] = None
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._is_initialized = False
        
        # Configuration defaults
        self.min_pool_size = self.config.get('min_pool_size', 5)
        self.max_pool_size = self.config.get('max_pool_size', 20)
        self.command_timeout = self.config.get('command_timeout', 30)
        self.server_settings = self.config.get('server_settings', {
            'application_name': 'xorb_platform',
            'jit': 'off'
        })
    
    async def initialize(self) -> None:
        """Initialize database connections and engine"""
        if self._is_initialized:
            return
        
        try:
            # Create AsyncPG connection pool
            self._pool = await asyncpg.create_pool(
                dsn=self.database_url,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                command_timeout=self.command_timeout,
                server_settings=self.server_settings
            )
            
            # Create SQLAlchemy async engine
            self._engine = create_async_engine(
                self.database_url,
                pool_size=self.max_pool_size,
                max_overflow=0,
                pool_timeout=30,
                pool_recycle=3600,
                echo=self.config.get('sql_echo', False)
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connections
            await self.health_check()
            
            self._is_initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self) -> None:
        """Cleanup database connections"""
        try:
            if self._pool:
                await self._pool.close()
                self._pool = None
            
            if self._engine:
                await self._engine.dispose()
                self._engine = None
            
            self._session_factory = None
            self._is_initialized = False
            
            logger.info("Database manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """Get raw AsyncPG connection from pool"""
        if not self._pool:
            raise RuntimeError("Database manager not initialized")
        
        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                # Log error but let it propagate
                logger.error(f"Database operation error: {e}")
                raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get SQLAlchemy async session"""
        if not self._session_factory:
            raise RuntimeError("Database manager not initialized")
        
        async with self._session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute raw SQL query with parameters"""
        async with self.get_connection() as conn:
            if params:
                return await conn.fetch(query, *params.values())
            else:
                return await conn.fetch(query)
    
    async def execute_command(
        self, 
        command: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute SQL command (INSERT, UPDATE, DELETE)"""
        async with self.get_connection() as conn:
            if params:
                return await conn.execute(command, *params.values())
            else:
                return await conn.execute(command)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity"""
        health_status = {
            'database': 'unhealthy',
            'pool_status': None,
            'connection_test': False,
            'query_test': False,
            'error': None
        }
        
        try:
            if not self._pool:
                health_status['error'] = "Database pool not initialized"
                return health_status
            
            # Check pool status
            health_status['pool_status'] = {
                'size': self._pool.get_size(),
                'max_size': self._pool.get_max_size(),
                'min_size': self._pool.get_min_size(),
                'idle_count': self._pool.get_idle_size()
            }
            
            # Test connection
            async with self.get_connection() as conn:
                health_status['connection_test'] = True
                
                # Test simple query
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    health_status['query_test'] = True
                    health_status['database'] = 'healthy'
            
        except Exception as e:
            health_status['error'] = str(e)
            logger.error(f"Database health check failed: {e}")
        
        return health_status
    
    async def run_migration(self, migration_sql: str) -> bool:
        """Run database migration script"""
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    await conn.execute(migration_sql)
            
            logger.info("Migration executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if database manager is initialized"""
        return self._is_initialized
    
    @property
    def pool(self) -> Optional[Pool]:
        """Get AsyncPG pool (for advanced usage)"""
        return self._pool
    
    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get SQLAlchemy engine (for advanced usage)"""
        return self._engine


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _database_manager
    if _database_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_database_manager() first.")
    return _database_manager


async def initialize_database_manager(database_url: str, config: Optional[Dict[str, Any]] = None) -> DatabaseManager:
    """Initialize global database manager"""
    global _database_manager
    
    if _database_manager is not None:
        logger.warning("Database manager already initialized")
        return _database_manager
    
    _database_manager = DatabaseManager(database_url, config)
    await _database_manager.initialize()
    
    return _database_manager


async def cleanup_database_manager():
    """Cleanup global database manager"""
    global _database_manager
    
    if _database_manager is not None:
        await _database_manager.cleanup()
        _database_manager = None