"""
Production Database Manager
Replaces development database stubs with enterprise-grade database management
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, MetaData, inspect
from sqlalchemy.exc import SQLAlchemyError

from .database_schema import Base, TenantModel, UserModel, OrganizationModel

# Optional dependencies - graceful fallback if not available
try:
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory
    ALEMBIC_AVAILABLE = True
except ImportError:
    ALEMBIC_AVAILABLE = False
from .production_database_repositories import (
    DatabaseConnectionManager, ProductionUserRepository, 
    ProductionOrganizationRepository, ProductionScanSessionRepository,
    ProductionAuthTokenRepository, ProductionTenantRepository
)
from ..domain.entities import User, Organization
from ..domain.tenant_entities import Tenant, TenantPlan, TenantStatus

logger = logging.getLogger(__name__)

if not ALEMBIC_AVAILABLE:
    logger.warning("Alembic not available - database migrations will use basic table creation")


class ProductionDatabaseManager:
    """
    Enterprise-grade database manager replacing development stubs
    
    Features:
    - Automatic schema migration
    - Connection pool management
    - Health monitoring
    - Performance optimization
    - Multi-tenant support
    - Backup and recovery
    """
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'postgresql+asyncpg://xorb:xorb@localhost:5432/xorb'
        )
        self.engine: Optional[AsyncEngine] = None
        self.session_factory = None
        self.connection_manager = None
        self.is_initialized = False
        
        # Performance settings
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '20'))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '10'))
        
        logger.info(f"Database manager initialized with URL: {self._mask_url(self.database_url)}")
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of database URL for logging"""
        if '@' in url:
            parts = url.split('@')
            if len(parts) == 2:
                # Mask the credentials part
                credentials = parts[0].split('://')
                if len(credentials) == 2:
                    protocol = credentials[0]
                    creds = credentials[1]
                    if ':' in creds:
                        user = creds.split(':')[0]
                        masked = f"{protocol}://{user}:***@{parts[1]}"
                        return masked
        return url
    
    async def initialize(self) -> bool:
        """Initialize database engine and perform setup"""
        try:
            logger.info("Initializing production database manager...")
            
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 hour
                connect_args={
                    "command_timeout": 30,
                    "server_settings": {
                        "application_name": "xorb_enterprise_platform",
                        "tcp_keepalives_idle": "600",
                        "tcp_keepalives_interval": "30", 
                        "tcp_keepalives_count": "3",
                        "statement_timeout": "300000",  # 5 minutes
                        "lock_timeout": "30000",  # 30 seconds
                    }
                }
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize connection manager
            self.connection_manager = DatabaseConnectionManager(self.database_url)
            await self.connection_manager.initialize()
            
            # Test connection
            await self._test_connection()
            
            # Run migrations
            await self._run_migrations()
            
            # Create default data
            await self._create_default_data()
            
            self.is_initialized = True
            logger.info("Production database manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            return False
    
    async def _test_connection(self):
        """Test database connection"""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT version()"))
                version = result.scalar()
                logger.info(f"Connected to PostgreSQL: {version}")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def _run_migrations(self):
        """Run database migrations using Alembic (if available)"""
        try:
            logger.info("Running database migrations...")
            
            if ALEMBIC_AVAILABLE:
                logger.info("Alembic available - running full migrations")
                # Run Alembic migrations programmatically
                from alembic.config import Config
                from alembic import command
                import tempfile
                import os
                
                # Create temporary alembic config
                alembic_cfg = Config()
                
                # Set migration script location
                migrations_dir = os.path.join(os.path.dirname(__file__), "..", "..", "migrations")
                if os.path.exists(migrations_dir):
                    alembic_cfg.set_main_option("script_location", migrations_dir)
                    alembic_cfg.set_main_option("sqlalchemy.url", str(self.database_url))
                    
                    # Run migrations to head
                    command.upgrade(alembic_cfg, "head")
                    logger.info("Alembic migrations completed successfully")
                else:
                    logger.warning(f"Migrations directory not found at {migrations_dir}, using basic schema")
            else:
                logger.info("Alembic not available - using basic table creation")
            
            # Create all tables if they don't exist
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database migrations completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def _create_default_data(self):
        """Create default tenant and admin user for production"""
        try:
            async with self.get_session() as session:
                # Check if default tenant exists
                existing_tenant = await session.execute(
                    text("SELECT id FROM tenants WHERE slug = 'default' LIMIT 1")
                )
                
                if not existing_tenant.scalar():
                    logger.info("Creating default tenant...")
                    
                    # Create default tenant
                    tenant_repo = ProductionTenantRepository(self.connection_manager)
                    default_tenant = await tenant_repo.create_tenant({
                        'name': 'Default Organization',
                        'slug': 'default',
                        'plan': TenantPlan.ENTERPRISE,
                        'settings': {
                            'max_users': 1000,
                            'max_scans_per_month': 10000,
                            'features': ['ptaas', 'compliance', 'threat_intelligence']
                        }
                    })
                    
                    # Create default admin user
                    user_repo = ProductionUserRepository(self.connection_manager)
                    admin_user = User.create(
                        username="admin",
                        email="admin@xorb.enterprise",
                        roles=["admin", "user", "security_analyst"]
                    )
                    
                    # Set default password hash (should be changed on first login)
                    admin_user.password_hash = "$2b$12$xorb.enterprise.default.admin.password.hash"
                    await user_repo.create(admin_user)
                    
                    logger.info("Default tenant and admin user created successfully")
                else:
                    logger.info("Default tenant already exists, skipping creation")
                    
        except Exception as e:
            logger.error(f"Failed to create default data: {e}")
            # Don't raise - this is not critical for initialization
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper error handling"""
        if not self.is_initialized:
            raise RuntimeError("Database manager not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.engine.begin() as conn:
                # Test basic connectivity
                await conn.execute(text("SELECT 1"))
                
                # Check connection pool status
                pool_status = {
                    "size": self.engine.pool.size(),
                    "checked_out": self.engine.pool.checkedout(),
                    "checked_in": self.engine.pool.checkedin(),
                    "overflow": self.engine.pool.overflow(),
                    "invalid": self.engine.pool.invalid()
                }
                
                # Get database stats
                stats_result = await conn.execute(text("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                        (SELECT setting FROM pg_settings WHERE name = 'max_connections') as max_connections
                """))
                
                stats = stats_result.fetchone()
                
                end_time = asyncio.get_event_loop().time()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2),
                    "pool_status": pool_status,
                    "database_size_bytes": stats[0] if stats else 0,
                    "active_connections": stats[1] if stats else 0,
                    "max_connections": stats[2] if stats else 0,
                    "engine_url": self._mask_url(str(self.engine.url))
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "engine_url": self._mask_url(str(self.engine.url)) if self.engine else "not_initialized"
            }
    
    async def get_repository_instances(self) -> Dict[str, Any]:
        """Get all production repository instances"""
        if not self.is_initialized:
            raise RuntimeError("Database manager not initialized")
        
        return {
            'user_repository': ProductionUserRepository(self.connection_manager),
            'organization_repository': ProductionOrganizationRepository(self.connection_manager),
            'scan_session_repository': ProductionScanSessionRepository(self.connection_manager),
            'auth_token_repository': ProductionAuthTokenRepository(self.connection_manager),
            'tenant_repository': ProductionTenantRepository(self.connection_manager)
        }
    
    async def backup_database(self, backup_path: str = None) -> Dict[str, Any]:
        """Create database backup"""
        try:
            import subprocess
            from datetime import datetime
            
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"/tmp/xorb_backup_{timestamp}.sql"
            
            # Extract connection details from URL
            url_parts = self.database_url.replace("postgresql+asyncpg://", "").split("@")
            if len(url_parts) != 2:
                raise ValueError("Invalid database URL format")
            
            credentials, host_db = url_parts
            username, password = credentials.split(":")
            host_port, database = host_db.split("/")
            host = host_port.split(":")[0]
            port = host_port.split(":")[1] if ":" in host_port else "5432"
            
            # Run pg_dump
            env = os.environ.copy()
            env['PGPASSWORD'] = password
            
            cmd = [
                'pg_dump',
                '-h', host,
                '-p', port,
                '-U', username,
                '-d', database,
                '-f', backup_path,
                '--verbose',
                '--no-owner',
                '--no-privileges'
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Get backup file size
                backup_size = os.path.getsize(backup_path)
                
                return {
                    "success": True,
                    "backup_path": backup_path,
                    "backup_size_bytes": backup_size,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "command": " ".join(cmd[:-2] + ["***", cmd[-1]])  # Mask password
                }
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Run database optimization tasks"""
        try:
            optimizations = []
            
            async with self.engine.begin() as conn:
                # Update table statistics
                await conn.execute(text("ANALYZE"))
                optimizations.append("Table statistics updated")
                
                # Vacuum critical tables
                critical_tables = ['users', 'scan_sessions', 'scan_findings', 'audit_logs']
                for table in critical_tables:
                    try:
                        await conn.execute(text(f"VACUUM ANALYZE {table}"))
                        optimizations.append(f"Vacuumed table: {table}")
                    except Exception as e:
                        logger.warning(f"Failed to vacuum table {table}: {e}")
                
                # Reindex if needed (be careful in production)
                if os.getenv('ALLOW_REINDEX', 'false').lower() == 'true':
                    await conn.execute(text("REINDEX DATABASE CONCURRENTLY"))
                    optimizations.append("Database reindexed")
            
            return {
                "success": True,
                "optimizations": optimizations,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        try:
            async with self.engine.begin() as conn:
                # Query performance stats
                metrics_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        n_tup_ins,
                        n_tup_upd,
                        n_tup_del
                    FROM pg_stat_user_tables
                    ORDER BY seq_scan DESC
                    LIMIT 10
                """)
                
                result = await conn.execute(metrics_query)
                table_stats = [dict(row) for row in result.fetchall()]
                
                # Index usage stats
                index_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    ORDER BY idx_scan DESC
                    LIMIT 20
                """)
                
                result = await conn.execute(index_query)
                index_stats = [dict(row) for row in result.fetchall()]
                
                # Connection stats
                conn_query = text("""
                    SELECT 
                        state,
                        count(*) as count
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                    GROUP BY state
                """)
                
                result = await conn.execute(conn_query)
                connection_stats = {row[0]: row[1] for row in result.fetchall()}
                
                return {
                    "table_statistics": table_stats,
                    "index_statistics": index_stats,
                    "connection_statistics": connection_stats,
                    "pool_status": {
                        "size": self.engine.pool.size(),
                        "checked_out": self.engine.pool.checkedout(),
                        "checked_in": self.engine.pool.checkedin()
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown database manager"""
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info("Database engine disposed successfully")
            
            self.is_initialized = False
            
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")


# Global instance for dependency injection
production_db_manager = ProductionDatabaseManager()


async def get_production_db_manager() -> ProductionDatabaseManager:
    """Get the global production database manager instance"""
    if not production_db_manager.is_initialized:
        await production_db_manager.initialize()
    return production_db_manager