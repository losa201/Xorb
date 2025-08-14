"""
Multi-Tenant Architecture Manager
Enterprise-grade multi-tenancy with data isolation, resource management, and tenant provisioning
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import hashlib
from pathlib import Path
import aioredis
import asyncpg
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    PROVISIONING = "provisioning"
    DEPROVISIONING = "deprovisioning"
    ARCHIVED = "archived"


class TenantTier(Enum):
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class ResourceType(Enum):
    API_CALLS = "api_calls"
    STORAGE = "storage"
    SCANS = "scans"
    USERS = "users"
    REPORTS = "reports"
    ASSETS = "assets"
    INTEGRATIONS = "integrations"


@dataclass
class ResourceQuota:
    """Resource quota configuration"""
    resource_type: ResourceType
    limit: int
    current_usage: int = 0
    reset_period: str = "monthly"  # daily, weekly, monthly, yearly
    last_reset: datetime = field(default_factory=datetime.now)

    def is_exceeded(self) -> bool:
        return self.current_usage >= self.limit

    def get_usage_percentage(self) -> float:
        return (self.current_usage / self.limit * 100) if self.limit > 0 else 0


@dataclass
class TenantConfiguration:
    """Tenant-specific configuration"""
    id: str
    name: str
    slug: str  # URL-safe identifier
    status: TenantStatus
    tier: TenantTier

    # Organization details
    organization_name: str
    contact_email: str
    contact_phone: Optional[str] = None

    # Resource quotas and limits
    resource_quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)

    # Feature flags
    enabled_features: Set[str] = field(default_factory=set)
    disabled_features: Set[str] = field(default_factory=set)

    # Database and storage configuration
    database_schema: str = ""
    storage_prefix: str = ""

    # Security settings
    ip_whitelist: List[str] = field(default_factory=list)
    sso_required: bool = False
    mfa_required: bool = False
    session_timeout_minutes: int = 60

    # Customization
    custom_domain: Optional[str] = None
    branding_config: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    subscription_expires_at: Optional[datetime] = None

    # Usage tracking
    last_activity_at: Optional[datetime] = None
    total_api_calls: int = 0
    total_users: int = 0

    def get_database_connection_params(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get tenant-specific database connection parameters"""
        params = base_config.copy()
        if self.database_schema:
            params["options"] = f"-c search_path={self.database_schema}"
        return params

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled for this tenant"""
        if feature in self.disabled_features:
            return False
        return feature in self.enabled_features

    def is_subscription_expired(self) -> bool:
        """Check if tenant subscription is expired"""
        if not self.subscription_expires_at:
            return False
        return datetime.now() > self.subscription_expires_at


@dataclass
class TenantContext:
    """Runtime tenant context"""
    tenant_id: str
    tenant_config: TenantConfiguration
    database_pool: Optional[asyncpg.Pool] = None
    redis_client: Optional[aioredis.Redis] = None

    # Request context
    current_user_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None


class MultiTenantManager:
    """Multi-tenant architecture manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tenants: Dict[str, TenantConfiguration] = {}
        self.tenant_contexts: Dict[str, TenantContext] = {}

        # Database connections
        self.master_db_pool: Optional[asyncpg.Pool] = None
        self.tenant_db_pools: Dict[str, asyncpg.Pool] = {}

        # Redis clients
        self.master_redis: Optional[aioredis.Redis] = None
        self.tenant_redis_clients: Dict[str, aioredis.Redis] = {}

        # Tier configurations
        self.tier_configs = self._initialize_tier_configs()

    def _initialize_tier_configs(self) -> Dict[TenantTier, Dict[str, Any]]:
        """Initialize tenant tier configurations"""
        return {
            TenantTier.BASIC: {
                "resource_quotas": {
                    ResourceType.API_CALLS: 10000,
                    ResourceType.STORAGE: 1024 * 1024 * 1024,  # 1GB
                    ResourceType.SCANS: 50,
                    ResourceType.USERS: 5,
                    ResourceType.REPORTS: 100,
                    ResourceType.ASSETS: 100,
                    ResourceType.INTEGRATIONS: 3
                },
                "features": [
                    "vulnerability_scanning",
                    "basic_reporting",
                    "email_notifications"
                ]
            },
            TenantTier.PROFESSIONAL: {
                "resource_quotas": {
                    ResourceType.API_CALLS: 100000,
                    ResourceType.STORAGE: 10 * 1024 * 1024 * 1024,  # 10GB
                    ResourceType.SCANS: 500,
                    ResourceType.USERS: 25,
                    ResourceType.REPORTS: 1000,
                    ResourceType.ASSETS: 1000,
                    ResourceType.INTEGRATIONS: 10
                },
                "features": [
                    "vulnerability_scanning",
                    "advanced_reporting",
                    "compliance_automation",
                    "api_access",
                    "slack_integration",
                    "email_notifications",
                    "webhook_notifications"
                ]
            },
            TenantTier.ENTERPRISE: {
                "resource_quotas": {
                    ResourceType.API_CALLS: 1000000,
                    ResourceType.STORAGE: 100 * 1024 * 1024 * 1024,  # 100GB
                    ResourceType.SCANS: 5000,
                    ResourceType.USERS: 100,
                    ResourceType.REPORTS: 10000,
                    ResourceType.ASSETS: 10000,
                    ResourceType.INTEGRATIONS: 50
                },
                "features": [
                    "vulnerability_scanning",
                    "advanced_reporting",
                    "compliance_automation",
                    "threat_intelligence",
                    "api_access",
                    "sso_integration",
                    "custom_integrations",
                    "priority_support",
                    "custom_branding",
                    "white_label",
                    "advanced_analytics",
                    "real_time_monitoring"
                ]
            }
        }

    async def initialize(self):
        """Initialize multi-tenant manager"""
        logger.info("Initializing Multi-Tenant Manager...")

        # Initialize master database connection
        await self._initialize_master_database()

        # Initialize master Redis connection
        await self._initialize_master_redis()

        # Load existing tenants
        await self._load_tenants()

        # Start background tasks
        asyncio.create_task(self._quota_monitoring_loop())
        asyncio.create_task(self._tenant_health_monitoring_loop())
        asyncio.create_task(self._cleanup_inactive_tenants_loop())

        logger.info(f"Multi-Tenant Manager initialized with {len(self.tenants)} tenants")

    async def _initialize_master_database(self):
        """Initialize master database connection"""
        db_config = self.config.get("database", {})
        dsn = db_config.get("dsn", "postgresql://localhost/xorb_master")

        self.master_db_pool = await asyncpg.create_pool(
            dsn,
            min_size=5,
            max_size=20,
            command_timeout=60
        )

        # Create tenant management tables
        await self._create_tenant_tables()

    async def _initialize_master_redis(self):
        """Initialize master Redis connection"""
        redis_url = self.config.get("redis_url", "redis://localhost:6379")
        self.master_redis = await aioredis.from_url(f"{redis_url}/0")  # Use DB 0 for master

    async def _create_tenant_tables(self):
        """Create tenant management database tables"""
        async with self.master_db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(100) UNIQUE NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    tier VARCHAR(50) NOT NULL,
                    organization_name VARCHAR(255) NOT NULL,
                    contact_email VARCHAR(255) NOT NULL,
                    contact_phone VARCHAR(50),
                    database_schema VARCHAR(100),
                    storage_prefix VARCHAR(100),
                    custom_domain VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    subscription_expires_at TIMESTAMP WITH TIME ZONE,
                    last_activity_at TIMESTAMP WITH TIME ZONE,
                    total_api_calls BIGINT DEFAULT 0,
                    total_users INTEGER DEFAULT 0,
                    config JSONB
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tenant_resource_usage (
                    tenant_id VARCHAR(36) NOT NULL,
                    resource_type VARCHAR(50) NOT NULL,
                    current_usage BIGINT NOT NULL,
                    quota_limit BIGINT NOT NULL,
                    last_reset TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    reset_period VARCHAR(20) NOT NULL,
                    PRIMARY KEY (tenant_id, resource_type),
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);
                CREATE INDEX IF NOT EXISTS idx_tenants_slug ON tenants(slug);
                CREATE INDEX IF NOT EXISTS idx_tenant_usage_tenant_id ON tenant_resource_usage(tenant_id);
            """)

    async def _load_tenants(self):
        """Load existing tenants from database"""
        async with self.master_db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM tenants WHERE status != 'archived'")

            for row in rows:
                config_data = json.loads(row["config"]) if row["config"] else {}

                tenant_config = TenantConfiguration(
                    id=row["id"],
                    name=row["name"],
                    slug=row["slug"],
                    status=TenantStatus(row["status"]),
                    tier=TenantTier(row["tier"]),
                    organization_name=row["organization_name"],
                    contact_email=row["contact_email"],
                    contact_phone=row["contact_phone"],
                    database_schema=row["database_schema"],
                    storage_prefix=row["storage_prefix"],
                    custom_domain=row["custom_domain"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    subscription_expires_at=row["subscription_expires_at"],
                    last_activity_at=row["last_activity_at"],
                    total_api_calls=row["total_api_calls"],
                    total_users=row["total_users"]
                )

                # Load additional config from JSON
                if config_data:
                    tenant_config.enabled_features = set(config_data.get("enabled_features", []))
                    tenant_config.disabled_features = set(config_data.get("disabled_features", []))
                    tenant_config.ip_whitelist = config_data.get("ip_whitelist", [])
                    tenant_config.sso_required = config_data.get("sso_required", False)
                    tenant_config.mfa_required = config_data.get("mfa_required", False)
                    tenant_config.session_timeout_minutes = config_data.get("session_timeout_minutes", 60)
                    tenant_config.branding_config = config_data.get("branding_config", {})
                    tenant_config.custom_settings = config_data.get("custom_settings", {})

                # Load resource quotas
                await self._load_tenant_resource_quotas(tenant_config)

                self.tenants[tenant_config.id] = tenant_config

                # Initialize tenant database connection
                await self._initialize_tenant_database(tenant_config)

                # Initialize tenant Redis connection
                await self._initialize_tenant_redis(tenant_config)

    async def _load_tenant_resource_quotas(self, tenant_config: TenantConfiguration):
        """Load resource quotas for tenant"""
        async with self.master_db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM tenant_resource_usage WHERE tenant_id = $1",
                tenant_config.id
            )

            for row in rows:
                resource_type = ResourceType(row["resource_type"])
                quota = ResourceQuota(
                    resource_type=resource_type,
                    limit=row["quota_limit"],
                    current_usage=row["current_usage"],
                    reset_period=row["reset_period"],
                    last_reset=row["last_reset"]
                )
                tenant_config.resource_quotas[resource_type] = quota

    async def create_tenant(self, tenant_data: Dict[str, Any]) -> TenantConfiguration:
        """Create new tenant"""
        tenant_id = str(uuid.uuid4())
        slug = self._generate_tenant_slug(tenant_data["name"])

        # Validate tier
        tier = TenantTier(tenant_data.get("tier", "basic"))
        tier_config = self.tier_configs[tier]

        # Create tenant configuration
        tenant_config = TenantConfiguration(
            id=tenant_id,
            name=tenant_data["name"],
            slug=slug,
            status=TenantStatus.PROVISIONING,
            tier=tier,
            organization_name=tenant_data["organization_name"],
            contact_email=tenant_data["contact_email"],
            contact_phone=tenant_data.get("contact_phone"),
            database_schema=f"tenant_{slug}",
            storage_prefix=f"tenant/{slug}/",
            enabled_features=set(tier_config["features"])
        )

        # Set subscription expiry for trial
        if tenant_data.get("trial", False):
            tenant_config.status = TenantStatus.TRIAL
            tenant_config.subscription_expires_at = datetime.now() + timedelta(days=30)

        # Create resource quotas
        for resource_type, limit in tier_config["resource_quotas"].items():
            quota = ResourceQuota(
                resource_type=ResourceType(resource_type.value),
                limit=limit
            )
            tenant_config.resource_quotas[ResourceType(resource_type.value)] = quota

        try:
            # Save to database
            await self._save_tenant_to_database(tenant_config)

            # Provision tenant resources
            await self._provision_tenant_resources(tenant_config)

            # Update status to active
            tenant_config.status = TenantStatus.ACTIVE
            await self._update_tenant_status(tenant_config.id, TenantStatus.ACTIVE)

            # Cache tenant
            self.tenants[tenant_id] = tenant_config

            logger.info(f"Created tenant {tenant_id} ({slug}) successfully")
            return tenant_config

        except Exception as e:
            logger.error(f"Failed to create tenant {tenant_id}: {e}")
            # Cleanup on failure
            await self._cleanup_failed_tenant_creation(tenant_config)
            raise

    async def _save_tenant_to_database(self, tenant_config: TenantConfiguration):
        """Save tenant configuration to database"""
        config_json = {
            "enabled_features": list(tenant_config.enabled_features),
            "disabled_features": list(tenant_config.disabled_features),
            "ip_whitelist": tenant_config.ip_whitelist,
            "sso_required": tenant_config.sso_required,
            "mfa_required": tenant_config.mfa_required,
            "session_timeout_minutes": tenant_config.session_timeout_minutes,
            "branding_config": tenant_config.branding_config,
            "custom_settings": tenant_config.custom_settings
        }

        async with self.master_db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenants (
                    id, name, slug, status, tier, organization_name, contact_email,
                    contact_phone, database_schema, storage_prefix, custom_domain,
                    subscription_expires_at, config
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                tenant_config.id,
                tenant_config.name,
                tenant_config.slug,
                tenant_config.status.value,
                tenant_config.tier.value,
                tenant_config.organization_name,
                tenant_config.contact_email,
                tenant_config.contact_phone,
                tenant_config.database_schema,
                tenant_config.storage_prefix,
                tenant_config.custom_domain,
                tenant_config.subscription_expires_at,
                json.dumps(config_json)
            )

            # Save resource quotas
            for resource_type, quota in tenant_config.resource_quotas.items():
                await conn.execute("""
                    INSERT INTO tenant_resource_usage (
                        tenant_id, resource_type, current_usage, quota_limit, reset_period
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    tenant_config.id,
                    resource_type.value,
                    quota.current_usage,
                    quota.limit,
                    quota.reset_period
                )

    async def _provision_tenant_resources(self, tenant_config: TenantConfiguration):
        """Provision resources for new tenant"""
        # Create tenant database schema
        await self._create_tenant_database_schema(tenant_config)

        # Initialize tenant database connection
        await self._initialize_tenant_database(tenant_config)

        # Initialize tenant Redis connection
        await self._initialize_tenant_redis(tenant_config)

        # Create default tenant data
        await self._create_default_tenant_data(tenant_config)

    async def _create_tenant_database_schema(self, tenant_config: TenantConfiguration):
        """Create isolated database schema for tenant"""
        schema_name = tenant_config.database_schema

        async with self.master_db_pool.acquire() as conn:
            # Create schema
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

            # Set search path for subsequent operations
            await conn.execute(f'SET search_path TO "{schema_name}", public')

            # Create tenant-specific tables
            await self._create_tenant_tables(conn, tenant_config)

    async def _create_tenant_tables(self, conn, tenant_config: TenantConfiguration):
        """Create tenant-specific database tables"""
        # Users table
        await conn.execute("""
            CREATE TABLE users (
                id VARCHAR(36) PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255),
                role VARCHAR(50) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_login TIMESTAMP WITH TIME ZONE,
                sso_provider VARCHAR(50),
                sso_user_id VARCHAR(255)
            )
        """)

        # Assets table
        await conn.execute("""
            CREATE TABLE assets (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL,
                ip_address INET,
                hostname VARCHAR(255),
                description TEXT,
                tags JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Scans table
        await conn.execute("""
            CREATE TABLE scans (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL,
                status VARCHAR(50) NOT NULL,
                asset_id VARCHAR(36),
                configuration JSONB,
                results JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
        """)

        # Reports table
        await conn.execute("""
            CREATE TABLE reports (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL,
                format VARCHAR(20) NOT NULL,
                filters JSONB,
                generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                file_path VARCHAR(500)
            )
        """)

        # Create indexes
        await conn.execute("CREATE INDEX idx_users_email ON users(email)")
        await conn.execute("CREATE INDEX idx_assets_type ON assets(type)")
        await conn.execute("CREATE INDEX idx_scans_status ON scans(status)")
        await conn.execute("CREATE INDEX idx_scans_asset_id ON scans(asset_id)")

    async def _initialize_tenant_database(self, tenant_config: TenantConfiguration):
        """Initialize dedicated database connection for tenant"""
        db_config = self.config.get("database", {})
        connection_params = tenant_config.get_database_connection_params(db_config)

        tenant_pool = await asyncpg.create_pool(
            connection_params.get("dsn", "postgresql://localhost/xorb"),
            min_size=2,
            max_size=10,
            command_timeout=30,
            server_settings=connection_params.get("options", {})
        )

        self.tenant_db_pools[tenant_config.id] = tenant_pool

    async def _initialize_tenant_redis(self, tenant_config: TenantConfiguration):
        """Initialize dedicated Redis connection for tenant"""
        redis_url = self.config.get("redis_url", "redis://localhost:6379")

        # Use different Redis DB for each tenant (limited to 16 databases)
        tenant_db = hash(tenant_config.id) % 15 + 1  # Use DBs 1-15 for tenants

        tenant_redis = await aioredis.from_url(f"{redis_url}/{tenant_db}")
        self.tenant_redis_clients[tenant_config.id] = tenant_redis

    async def _create_default_tenant_data(self, tenant_config: TenantConfiguration):
        """Create default data for new tenant"""
        # Create admin user
        admin_user_id = str(uuid.uuid4())
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()  # Default password

        async with self.tenant_db_pools[tenant_config.id].acquire() as conn:
            await conn.execute("""
                INSERT INTO users (id, username, email, password_hash, role)
                VALUES ($1, $2, $3, $4, $5)
            """,
                admin_user_id,
                "admin",
                tenant_config.contact_email,
                password_hash,
                "admin"
            )

    def _generate_tenant_slug(self, name: str) -> str:
        """Generate URL-safe tenant slug"""
        import re
        slug = re.sub(r'[^a-zA-Z0-9-]', '-', name.lower().strip())
        slug = re.sub(r'-+', '-', slug).strip('-')

        # Ensure uniqueness
        base_slug = slug
        counter = 1
        while any(t.slug == slug for t in self.tenants.values()):
            slug = f"{base_slug}-{counter}"
            counter += 1

        return slug

    async def get_tenant_context(self, tenant_identifier: str) -> Optional[TenantContext]:
        """Get tenant context by ID or slug"""
        tenant_config = None

        # Try by ID first
        if tenant_identifier in self.tenants:
            tenant_config = self.tenants[tenant_identifier]
        else:
            # Try by slug
            for tenant in self.tenants.values():
                if tenant.slug == tenant_identifier:
                    tenant_config = tenant
                    break

        if not tenant_config:
            return None

        # Create or get cached context
        if tenant_config.id not in self.tenant_contexts:
            context = TenantContext(
                tenant_id=tenant_config.id,
                tenant_config=tenant_config,
                database_pool=self.tenant_db_pools.get(tenant_config.id),
                redis_client=self.tenant_redis_clients.get(tenant_config.id)
            )
            self.tenant_contexts[tenant_config.id] = context

        return self.tenant_contexts[tenant_config.id]

    @asynccontextmanager
    async def tenant_database_transaction(self, tenant_id: str):
        """Get tenant database transaction context"""
        if tenant_id not in self.tenant_db_pools:
            raise ValueError(f"No database pool for tenant {tenant_id}")

        pool = self.tenant_db_pools[tenant_id]
        async with pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def check_resource_quota(self, tenant_id: str, resource_type: ResourceType,
                                 amount: int = 1) -> bool:
        """Check if tenant can consume specified resource amount"""
        tenant_config = self.tenants.get(tenant_id)
        if not tenant_config:
            return False

        quota = tenant_config.resource_quotas.get(resource_type)
        if not quota:
            return True  # No quota set

        return quota.current_usage + amount <= quota.limit

    async def consume_resource(self, tenant_id: str, resource_type: ResourceType,
                             amount: int = 1) -> bool:
        """Consume tenant resource quota"""
        if not await self.check_resource_quota(tenant_id, resource_type, amount):
            return False

        tenant_config = self.tenants[tenant_id]
        quota = tenant_config.resource_quotas[resource_type]
        quota.current_usage += amount

        # Update in database
        async with self.master_db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE tenant_resource_usage
                SET current_usage = $1
                WHERE tenant_id = $2 AND resource_type = $3
            """, quota.current_usage, tenant_id, resource_type.value)

        return True

    async def reset_resource_quotas(self, tenant_id: str):
        """Reset resource quotas based on reset periods"""
        tenant_config = self.tenants.get(tenant_id)
        if not tenant_config:
            return

        now = datetime.now()

        for resource_type, quota in tenant_config.resource_quotas.items():
            should_reset = False

            if quota.reset_period == "daily":
                should_reset = (now - quota.last_reset).days >= 1
            elif quota.reset_period == "weekly":
                should_reset = (now - quota.last_reset).days >= 7
            elif quota.reset_period == "monthly":
                should_reset = (now - quota.last_reset).days >= 30
            elif quota.reset_period == "yearly":
                should_reset = (now - quota.last_reset).days >= 365

            if should_reset:
                quota.current_usage = 0
                quota.last_reset = now

                # Update in database
                async with self.master_db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE tenant_resource_usage
                        SET current_usage = 0, last_reset = $1
                        WHERE tenant_id = $2 AND resource_type = $3
                    """, now, tenant_id, resource_type.value)

    async def _update_tenant_status(self, tenant_id: str, status: TenantStatus):
        """Update tenant status"""
        async with self.master_db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE tenants SET status = $1, updated_at = NOW()
                WHERE id = $2
            """, status.value, tenant_id)

        if tenant_id in self.tenants:
            self.tenants[tenant_id].status = status

    async def suspend_tenant(self, tenant_id: str, reason: str = ""):
        """Suspend tenant"""
        await self._update_tenant_status(tenant_id, TenantStatus.SUSPENDED)
        logger.warning(f"Suspended tenant {tenant_id}: {reason}")

    async def activate_tenant(self, tenant_id: str):
        """Activate suspended tenant"""
        await self._update_tenant_status(tenant_id, TenantStatus.ACTIVE)
        logger.info(f"Activated tenant {tenant_id}")

    async def _quota_monitoring_loop(self):
        """Background loop for quota monitoring"""
        while True:
            try:
                for tenant_id in list(self.tenants.keys()):
                    await self.reset_resource_quotas(tenant_id)

                    # Check for quota violations
                    tenant_config = self.tenants[tenant_id]
                    for resource_type, quota in tenant_config.resource_quotas.items():
                        if quota.is_exceeded():
                            logger.warning(f"Tenant {tenant_id} exceeded {resource_type.value} quota")

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Error in quota monitoring loop: {e}")
                await asyncio.sleep(300)

    async def _tenant_health_monitoring_loop(self):
        """Background loop for tenant health monitoring"""
        while True:
            try:
                for tenant_config in self.tenants.values():
                    # Check subscription expiry
                    if tenant_config.is_subscription_expired():
                        if tenant_config.status == TenantStatus.TRIAL:
                            await self.suspend_tenant(tenant_config.id, "Trial expired")

                await asyncio.sleep(86400)  # Run daily

            except Exception as e:
                logger.error(f"Error in tenant health monitoring: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_inactive_tenants_loop(self):
        """Background loop for cleaning up inactive tenants"""
        while True:
            try:
                cutoff_date = datetime.now() - timedelta(days=365)  # 1 year inactive

                for tenant_config in list(self.tenants.values()):
                    if (tenant_config.last_activity_at and
                        tenant_config.last_activity_at < cutoff_date and
                        tenant_config.status == TenantStatus.SUSPENDED):

                        # Archive very old inactive tenants
                        await self._archive_tenant(tenant_config.id)

                await asyncio.sleep(86400 * 7)  # Run weekly

            except Exception as e:
                logger.error(f"Error in tenant cleanup: {e}")
                await asyncio.sleep(3600)

    async def _archive_tenant(self, tenant_id: str):
        """Archive inactive tenant"""
        await self._update_tenant_status(tenant_id, TenantStatus.ARCHIVED)

        # Cleanup connections
        if tenant_id in self.tenant_db_pools:
            await self.tenant_db_pools[tenant_id].close()
            del self.tenant_db_pools[tenant_id]

        if tenant_id in self.tenant_redis_clients:
            await self.tenant_redis_clients[tenant_id].close()
            del self.tenant_redis_clients[tenant_id]

        if tenant_id in self.tenant_contexts:
            del self.tenant_contexts[tenant_id]

        if tenant_id in self.tenants:
            del self.tenants[tenant_id]

        logger.info(f"Archived tenant {tenant_id}")

    async def _cleanup_failed_tenant_creation(self, tenant_config: TenantConfiguration):
        """Cleanup resources after failed tenant creation"""
        try:
            # Remove from database
            async with self.master_db_pool.acquire() as conn:
                await conn.execute("DELETE FROM tenants WHERE id = $1", tenant_config.id)
                await conn.execute("DELETE FROM tenant_resource_usage WHERE tenant_id = $1", tenant_config.id)

                # Drop schema if created
                if tenant_config.database_schema:
                    await conn.execute(f'DROP SCHEMA IF EXISTS "{tenant_config.database_schema}" CASCADE')

        except Exception as e:
            logger.error(f"Error cleaning up failed tenant creation: {e}")

    async def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant usage metrics"""
        tenant_config = self.tenants.get(tenant_id)
        if not tenant_config:
            return {}

        metrics = {
            "tenant_id": tenant_id,
            "name": tenant_config.name,
            "status": tenant_config.status.value,
            "tier": tenant_config.tier.value,
            "total_api_calls": tenant_config.total_api_calls,
            "total_users": tenant_config.total_users,
            "last_activity": tenant_config.last_activity_at.isoformat() if tenant_config.last_activity_at else None,
            "resource_usage": {}
        }

        for resource_type, quota in tenant_config.resource_quotas.items():
            metrics["resource_usage"][resource_type.value] = {
                "current": quota.current_usage,
                "limit": quota.limit,
                "percentage": quota.get_usage_percentage(),
                "exceeded": quota.is_exceeded()
            }

        return metrics

    async def shutdown(self):
        """Shutdown multi-tenant manager"""
        # Close all tenant database pools
        for pool in self.tenant_db_pools.values():
            await pool.close()

        # Close all tenant Redis clients
        for client in self.tenant_redis_clients.values():
            await client.close()

        # Close master connections
        if self.master_db_pool:
            await self.master_db_pool.close()

        if self.master_redis:
            await self.master_redis.close()

        logger.info("Multi-Tenant Manager shutdown complete")


# Factory function
def create_multi_tenant_manager(config: Dict[str, Any]) -> MultiTenantManager:
    """Create and configure multi-tenant manager"""
    default_config = {
        "database": {
            "dsn": "postgresql://localhost/xorb_master"
        },
        "redis_url": "redis://localhost:6379",
        "default_tier": "basic"
    }

    final_config = {**default_config, **config}
    return MultiTenantManager(final_config)
