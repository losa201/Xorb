#!/usr/bin/env python3
"""
XORB Database Management Script

This script provides comprehensive database management capabilities:
1. Migration management
2. Database initialization for different environments
3. Tenant management
4. Performance monitoring
5. Backup and maintenance

Usage:
    python db_management.py init --env development
    python db_management.py migrate --target head
    python db_management.py create-tenant --name "Acme Corp" --slug "acme"
    python db_management.py backup --output backup.sql
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

import asyncpg
from alembic.config import Config
from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configuration
DATABASE_CONFIGS = {
    'development': {
        'database_url': 'postgresql://xorb_dev:dev_password@localhost:5432/xorb_dev',
        'max_connections': 10,
        'ssl_mode': 'disable'
    },
    'testing': {
        'database_url': 'postgresql://xorb_test:test_password@localhost:5432/xorb_test',
        'max_connections': 5,
        'ssl_mode': 'disable'
    },
    'staging': {
        'database_url': os.getenv('STAGING_DATABASE_URL'),
        'max_connections': 20,
        'ssl_mode': 'require'
    },
    'production': {
        'database_url': os.getenv('DATABASE_URL'),
        'max_connections': 50,
        'ssl_mode': 'require'
    }
}

class DatabaseManager:
    """Comprehensive database management class"""
    
    def __init__(self, environment: str = 'development'):
        self.environment = environment
        self.config = DATABASE_CONFIGS.get(environment, DATABASE_CONFIGS['development'])
        self.database_url = self.config['database_url'] or self._get_default_url()
        
        # Initialize async engine
        self.engine = create_async_engine(
            self.database_url,
            pool_size=self.config['max_connections'],
            pool_pre_ping=True,
            echo=environment == 'development'
        )
        
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    def _get_default_url(self) -> str:
        """Get default database URL for development"""
        return "postgresql://postgres:postgres@localhost:5432/xorb"
    
    async def initialize_database(self) -> bool:
        """Initialize database with extensions and basic setup"""
        try:
            print(f"🚀 Initializing database for {self.environment} environment...")
            
            # Create database if it doesn't exist (development only)
            if self.environment == 'development':
                await self._create_database_if_not_exists()
            
            # Connect and set up extensions
            async with self.engine.begin() as conn:
                # Enable required extensions
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\""))
                
                # Try to enable pgvector (may not be available in all environments)
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"vector\""))
                    print("✅ pgvector extension enabled")
                except Exception as e:
                    print(f"⚠️  pgvector extension not available: {e}")
                
                print("✅ Database extensions initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            return False
    
    async def _create_database_if_not_exists(self):
        """Create database if it doesn't exist (development only)"""
        # Extract database name from URL
        db_name = self.database_url.split('/')[-1]
        base_url = '/'.join(self.database_url.split('/')[:-1]) + '/postgres'
        
        try:
            # Connect to postgres database to create our database
            conn = await asyncpg.connect(base_url)
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )
            
            if not exists:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                print(f"✅ Created database: {db_name}")
            
            await conn.close()
            
        except Exception as e:
            print(f"Database creation skipped: {e}")
    
    def run_migrations(self, target: str = "head") -> bool:
        """Run Alembic migrations"""
        try:
            print(f"🔄 Running migrations to {target}...")
            
            # Configure Alembic
            alembic_cfg = Config(str(Path(__file__).parent / "alembic.ini"))
            alembic_cfg.set_main_option("sqlalchemy.url", self.database_url.replace('+asyncpg', ''))
            
            # Run migrations
            command.upgrade(alembic_cfg, target)
            print("✅ Migrations completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    async def create_tenant(self, name: str, slug: str, plan: str = "PROFESSIONAL") -> Optional[str]:
        """Create a new tenant"""
        try:
            print(f"👥 Creating tenant: {name} ({slug})")
            
            async with self.async_session() as session:
                # Insert new tenant
                result = await session.execute(
                    text("""
                        INSERT INTO tenants (name, slug, plan, status, settings)
                        VALUES (:name, :slug, :plan, 'ACTIVE', '{}')
                        RETURNING id
                    """),
                    {
                        "name": name,
                        "slug": slug,
                        "plan": plan
                    }
                )
                
                tenant_id = result.scalar()
                await session.commit()
                
                print(f"✅ Created tenant with ID: {tenant_id}")
                return str(tenant_id)
                
        except Exception as e:
            print(f"❌ Tenant creation failed: {e}")
            return None
    
    async def create_admin_user(self, tenant_id: str, email: str, username: str, password: str) -> bool:
        """Create an admin user for a tenant"""
        try:
            # Import unified auth service for password hashing
            from app.services.unified_auth_service_consolidated import UnifiedAuthService
            import redis.asyncio as redis
            from app.domain.repositories import UserRepository, AuthTokenRepository
            
            # Create auth service instance for password hashing
            redis_client = redis.from_url("redis://localhost:6379/0")
            user_repo = UserRepository(self.async_session())
            token_repo = AuthTokenRepository(self.async_session())
            auth_service = UnifiedAuthService(
                user_repository=user_repo,
                token_repository=token_repo,
                redis_client=redis_client,
                secret_key="temp-key-for-admin-creation"
            )
            
            password_hash = await auth_service.hash_password(password)
            
            async with self.async_session() as session:
                await session.execute(
                    text("""
                        INSERT INTO users (tenant_id, email, username, password_hash, is_active, roles)
                        VALUES (:tenant_id, :email, :username, :password_hash, true, :roles)
                    """),
                    {
                        "tenant_id": tenant_id,
                        "email": email,
                        "username": username,
                        "password_hash": password_hash,
                        "roles": json.dumps(["admin", "user"])
                    }
                )
                await session.commit()
                
                print(f"✅ Created admin user: {username}")
                return True
                
        except Exception as e:
            print(f"❌ User creation failed: {e}")
            return False
    
    async def get_tenant_stats(self) -> Dict[str, Any]:
        """Get comprehensive tenant statistics"""
        try:
            async with self.async_session() as session:
                result = await session.execute(text("""
                    SELECT 
                        COUNT(*) as total_tenants,
                        COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_tenants,
                        COUNT(CASE WHEN plan = 'ENTERPRISE' THEN 1 END) as enterprise_tenants,
                        COUNT(CASE WHEN plan = 'PROFESSIONAL' THEN 1 END) as professional_tenants,
                        COUNT(CASE WHEN plan = 'STARTER' THEN 1 END) as starter_tenants
                    FROM tenants
                """))
                
                stats = result.fetchone()
                
                # Get additional metrics
                threat_stats = await session.execute(text("""
                    SELECT 
                        COUNT(*) as total_indicators,
                        COUNT(CASE WHEN confidence_score > 0.8 THEN 1 END) as high_confidence,
                        COUNT(DISTINCT tenant_id) as tenants_with_indicators
                    FROM threat_indicators
                """))
                
                threat_data = threat_stats.fetchone()
                
                return {
                    "tenants": {
                        "total": stats[0],
                        "active": stats[1],
                        "enterprise": stats[2],
                        "professional": stats[3],
                        "starter": stats[4]
                    },
                    "threat_intelligence": {
                        "total_indicators": threat_data[0],
                        "high_confidence": threat_data[1],
                        "active_tenants": threat_data[2]
                    },
                    "generated_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {}
    
    async def backup_tenant_data(self, tenant_id: str, output_file: str) -> bool:
        """Backup all data for a specific tenant"""
        try:
            print(f"💾 Backing up tenant data: {tenant_id}")
            
            async with self.async_session() as session:
                # Get all tenant data
                tables = [
                    "tenants", "users", "organizations", "threat_feeds",
                    "threat_indicators", "embedding_vectors", "security_incidents",
                    "vulnerabilities", "attack_patterns", "threat_actors", "audit_logs"
                ]
                
                backup_data = {}
                
                for table in tables:
                    try:
                        result = await session.execute(
                            text(f"SELECT * FROM {table} WHERE tenant_id = :tenant_id"),
                            {"tenant_id": tenant_id}
                        )
                        
                        rows = result.fetchall()
                        backup_data[table] = [dict(row._mapping) for row in rows]
                        
                    except Exception as table_error:
                        print(f"⚠️  Skipping table {table}: {table_error}")
                        backup_data[table] = []
                
                # Write to file
                with open(output_file, 'w') as f:
                    json.dump(backup_data, f, indent=2, default=str)
                
                print(f"✅ Backup completed: {output_file}")
                return True
                
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    async def cleanup_old_data(self, retention_days: int = 90) -> Dict[str, int]:
        """Cleanup old audit logs and temporary data"""
        try:
            print(f"🧹 Cleaning up data older than {retention_days} days...")
            
            cleanup_results = {}
            
            async with self.async_session() as session:
                # Cleanup audit logs
                result = await session.execute(
                    text("SELECT cleanup_old_audit_logs(:retention_days)"),
                    {"retention_days": retention_days}
                )
                
                cleanup_results["audit_logs"] = result.scalar()
                
                # Additional cleanup operations can be added here
                await session.commit()
                
                print(f"✅ Cleanup completed: {cleanup_results}")
                return cleanup_results
                
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return {}
    
    async def close(self):
        """Close database connections"""
        await self.engine.dispose()


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="XORB Database Management")
    parser.add_argument("--env", default="development", 
                       choices=["development", "testing", "staging", "production"],
                       help="Environment configuration")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize database")
    
    # Migration command
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument("--target", default="head", help="Migration target")
    
    # Create tenant command
    tenant_parser = subparsers.add_parser("create-tenant", help="Create new tenant")
    tenant_parser.add_argument("--name", required=True, help="Tenant name")
    tenant_parser.add_argument("--slug", required=True, help="Tenant slug")
    tenant_parser.add_argument("--plan", default="PROFESSIONAL", help="Tenant plan")
    tenant_parser.add_argument("--admin-email", help="Admin user email")
    tenant_parser.add_argument("--admin-username", help="Admin username")
    tenant_parser.add_argument("--admin-password", help="Admin password")
    
    # Statistics command
    subparsers.add_parser("stats", help="Show database statistics")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup tenant data")
    backup_parser.add_argument("--tenant-id", required=True, help="Tenant ID")
    backup_parser.add_argument("--output", required=True, help="Output file")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old data")
    cleanup_parser.add_argument("--retention-days", type=int, default=90, 
                               help="Data retention days")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize database manager
    db_manager = DatabaseManager(args.env)
    
    try:
        if args.command == "init":
            success = await db_manager.initialize_database()
            if success:
                success = db_manager.run_migrations()
            sys.exit(0 if success else 1)
            
        elif args.command == "migrate":
            success = db_manager.run_migrations(args.target)
            sys.exit(0 if success else 1)
            
        elif args.command == "create-tenant":
            tenant_id = await db_manager.create_tenant(args.name, args.slug, args.plan)
            if tenant_id and args.admin_email:
                await db_manager.create_admin_user(
                    tenant_id, args.admin_email, 
                    args.admin_username or args.admin_email.split('@')[0],
                    args.admin_password or "admin123"
                )
            sys.exit(0 if tenant_id else 1)
            
        elif args.command == "stats":
            stats = await db_manager.get_tenant_stats()
            print("\n📊 Database Statistics:")
            print(json.dumps(stats, indent=2))
            
        elif args.command == "backup":
            success = await db_manager.backup_tenant_data(args.tenant_id, args.output)
            sys.exit(0 if success else 1)
            
        elif args.command == "cleanup":
            results = await db_manager.cleanup_old_data(args.retention_days)
            print(f"🧹 Cleanup results: {results}")
            
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())