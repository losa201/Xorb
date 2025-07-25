#!/usr/bin/env python3
"""
Xorb Feature Flag Service
Dynamic feature flags with usage tier integration and A/B testing capabilities
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum

import asyncpg
import aioredis
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge
from nats.aio.client import Client as NATS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
feature_flag_evaluations = Counter('xorb_feature_flag_evaluations_total', 'Feature flag evaluations', ['flag', 'result'])
feature_flag_cache_hits = Counter('xorb_feature_flag_cache_hits_total', 'Feature flag cache hits')
ab_test_assignments = Counter('xorb_ab_test_assignments_total', 'A/B test assignments', ['test', 'variant'])
feature_usage_by_tier = Counter('xorb_feature_usage_by_tier_total', 'Feature usage by tier', ['feature', 'tier'])

app = FastAPI(title="Xorb Feature Flag Service", version="1.0.0")

class FeatureType(Enum):
    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    JSON = "json"

class RolloutStrategy(Enum):
    ALL_USERS = "all_users"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    TIER_BASED = "tier_based"
    AB_TEST = "ab_test"

class BillingTier(Enum):
    GROWTH = "growth"
    ELITE = "elite"
    ENTERPRISE = "enterprise"

@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    key: str
    name: str
    description: str
    type: FeatureType
    default_value: Any
    enabled: bool
    rollout_strategy: RolloutStrategy
    rollout_config: Dict[str, Any]
    tier_restrictions: List[BillingTier]
    created_at: datetime
    updated_at: datetime
    created_by: str

@dataclass
class FeatureEvaluation:
    """Feature flag evaluation result"""
    key: str
    value: Any
    enabled: bool
    tier: str
    organization_id: str
    user_id: Optional[str]
    variant: Optional[str]
    evaluation_time: datetime
    source: str  # cache, database, default

class FeatureFlagService:
    """Central feature flag service with tier integration"""
    
    def __init__(self):
        self.db_pool = None
        self.redis = None
        self.nats = None
        
        # Caching
        self.flag_cache = {}
        self.tier_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Feature flag definitions
        self.default_flags = self.initialize_default_flags()
    
    def initialize_default_flags(self) -> Dict[str, FeatureFlag]:
        """Initialize default feature flags based on tiers"""
        
        flags = {}
        
        # Core scanning features
        flags["concurrent_scans"] = FeatureFlag(
            key="concurrent_scans",
            name="Concurrent Scans",
            description="Maximum number of concurrent scans allowed",
            type=FeatureType.NUMBER,
            default_value=5,
            enabled=True,
            rollout_strategy=RolloutStrategy.TIER_BASED,
            rollout_config={
                "growth": 5,
                "elite": 20,
                "enterprise": 100
            },
            tier_restrictions=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        flags["custom_rules"] = FeatureFlag(
            key="custom_rules",
            name="Custom Scanning Rules",
            description="Ability to create custom vulnerability scanning rules",
            type=FeatureType.BOOLEAN,
            default_value=False,
            enabled=True,
            rollout_strategy=RolloutStrategy.TIER_BASED,
            rollout_config={
                "growth": False,
                "elite": True,
                "enterprise": True
            },
            tier_restrictions=[BillingTier.ELITE, BillingTier.ENTERPRISE],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        flags["api_rate_limit"] = FeatureFlag(
            key="api_rate_limit",
            name="API Rate Limit",
            description="API requests per hour limit",
            type=FeatureType.NUMBER,
            default_value=1000,
            enabled=True,
            rollout_strategy=RolloutStrategy.TIER_BASED,
            rollout_config={
                "growth": 10000,
                "elite": 50000,
                "enterprise": -1  # unlimited
            },
            tier_restrictions=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        flags["advanced_analytics"] = FeatureFlag(
            key="advanced_analytics",
            name="Advanced Analytics",
            description="Access to advanced analytics and reporting features",
            type=FeatureType.BOOLEAN,
            default_value=False,
            enabled=True,
            rollout_strategy=RolloutStrategy.TIER_BASED,
            rollout_config={
                "growth": False,
                "elite": True,
                "enterprise": True
            },
            tier_restrictions=[BillingTier.ELITE, BillingTier.ENTERPRISE],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        flags["white_label"] = FeatureFlag(
            key="white_label",
            name="White Label UI",
            description="Custom branding and white-label interface",
            type=FeatureType.BOOLEAN,
            default_value=False,
            enabled=True,
            rollout_strategy=RolloutStrategy.TIER_BASED,
            rollout_config={
                "growth": False,
                "elite": False,
                "enterprise": True
            },
            tier_restrictions=[BillingTier.ENTERPRISE],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        flags["sso_integration"] = FeatureFlag(
            key="sso_integration",
            name="SSO Integration",
            description="Single Sign-On integration support",
            type=FeatureType.BOOLEAN,
            default_value=False,
            enabled=True,
            rollout_strategy=RolloutStrategy.TIER_BASED,
            rollout_config={
                "growth": False,
                "elite": False,
                "enterprise": True
            },
            tier_restrictions=[BillingTier.ENTERPRISE],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        # Beta features for A/B testing
        flags["new_triage_ui"] = FeatureFlag(
            key="new_triage_ui",
            name="New Triage UI",
            description="Beta version of the new triage interface",
            type=FeatureType.BOOLEAN,
            default_value=False,
            enabled=True,
            rollout_strategy=RolloutStrategy.AB_TEST,
            rollout_config={
                "test_name": "triage_ui_redesign",
                "variants": {
                    "control": {"enabled": False, "weight": 50},
                    "treatment": {"enabled": True, "weight": 50}
                }
            },
            tier_restrictions=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        flags["ai_powered_prioritization"] = FeatureFlag(
            key="ai_powered_prioritization",
            name="AI-Powered Vulnerability Prioritization",
            description="Use AI to automatically prioritize vulnerabilities",
            type=FeatureType.BOOLEAN,
            default_value=False,
            enabled=True,
            rollout_strategy=RolloutStrategy.PERCENTAGE,
            rollout_config={
                "percentage": 25  # Gradual rollout to 25% of users
            },
            tier_restrictions=[BillingTier.ELITE, BillingTier.ENTERPRISE],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        return flags
    
    async def initialize(self):
        """Initialize the feature flag service"""
        logger.info("Initializing Feature Flag Service...")
        
        # Database connection
        database_url = os.getenv("DATABASE_URL", "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=5, max_size=20)
        
        # Redis connection
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.redis = await aioredis.create_redis_pool(redis_url)
        
        # NATS connection
        self.nats = NATS()
        await self.nats.connect(os.getenv("NATS_URL", "nats://nats:4222"))
        
        # Create database tables
        await self.create_database_tables()
        
        # Load feature flags from database
        await self.load_feature_flags()
        
        # Subscribe to tier changes
        await self.nats.subscribe("billing.tier_changed", cb=self.handle_tier_change)
        
        logger.info("Feature Flag Service initialized")
    
    async def create_database_tables(self):
        """Create feature flag database tables"""
        async with self.db_pool.acquire() as conn:
            # Feature flags table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_flags (
                    key VARCHAR(100) PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    type VARCHAR(20) NOT NULL,
                    default_value JSONB NOT NULL,
                    enabled BOOLEAN DEFAULT true,
                    rollout_strategy VARCHAR(50) NOT NULL,
                    rollout_config JSONB NOT NULL,
                    tier_restrictions JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    created_by VARCHAR(100)
                )
            """)
            
            # Feature evaluations log
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_evaluations (
                    id SERIAL PRIMARY KEY,
                    flag_key VARCHAR(100) NOT NULL,
                    organization_id UUID,
                    user_id UUID,
                    tier VARCHAR(50),
                    value JSONB NOT NULL,
                    variant VARCHAR(50),
                    evaluation_time TIMESTAMP DEFAULT NOW(),
                    source VARCHAR(20) NOT NULL
                )
            """)
            
            # A/B test assignments
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_assignments (
                    id SERIAL PRIMARY KEY,
                    test_name VARCHAR(100) NOT NULL,
                    organization_id UUID,
                    user_id UUID,
                    variant VARCHAR(50) NOT NULL,
                    assigned_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(test_name, organization_id, user_id)
                )
            """)
            
            # Create indices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_flag_org ON feature_evaluations(flag_key, organization_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ab_assignments_test ON ab_test_assignments(test_name)")
    
    async def load_feature_flags(self):
        """Load feature flags from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM feature_flags")
                
                for row in rows:
                    flag = FeatureFlag(
                        key=row['key'],
                        name=row['name'],
                        description=row['description'],
                        type=FeatureType(row['type']),
                        default_value=row['default_value'],
                        enabled=row['enabled'],
                        rollout_strategy=RolloutStrategy(row['rollout_strategy']),
                        rollout_config=row['rollout_config'],
                        tier_restrictions=[BillingTier(t) for t in row['tier_restrictions']] if row['tier_restrictions'] else [],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        created_by=row['created_by']
                    )
                    self.flag_cache[flag.key] = flag
                
                # Insert default flags if they don't exist
                for key, flag in self.default_flags.items():
                    if key not in self.flag_cache:
                        await self.create_feature_flag(flag)
                        self.flag_cache[key] = flag
                
                logger.info(f"Loaded {len(self.flag_cache)} feature flags")
                
        except Exception as e:
            logger.error(f"Failed to load feature flags: {e}")
    
    async def create_feature_flag(self, flag: FeatureFlag):
        """Create a new feature flag in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feature_flags (
                    key, name, description, type, default_value, enabled,
                    rollout_strategy, rollout_config, tier_restrictions, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (key) DO NOTHING
            """, 
            flag.key, flag.name, flag.description, flag.type.value,
            json.dumps(flag.default_value), flag.enabled, flag.rollout_strategy.value,
            json.dumps(flag.rollout_config), json.dumps([t.value for t in flag.tier_restrictions]),
            flag.created_by)
    
    async def get_organization_tier(self, organization_id: str) -> str:
        """Get organization billing tier"""
        
        # Check cache first
        cache_key = f"tier:{organization_id}"
        if cache_key in self.tier_cache:
            cached_tier, cached_time = self.tier_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_tier
        
        # Check Redis cache
        try:
            cached_tier = await self.redis.get(cache_key)
            if cached_tier:
                self.tier_cache[cache_key] = (cached_tier.decode(), time.time())
                return cached_tier.decode()
        except Exception as e:
            logger.error(f"Redis tier cache error: {e}")
        
        # Query database
        try:
            async with self.db_pool.acquire() as conn:
                tier = await conn.fetchval("""
                    SELECT tier FROM subscriptions 
                    WHERE organization_id = $1 AND status = 'active'
                    ORDER BY created_at DESC LIMIT 1
                """, organization_id)
                
                if tier:
                    # Cache the result
                    self.tier_cache[cache_key] = (tier, time.time())
                    await self.redis.setex(cache_key, self.cache_ttl, tier)
                    return tier
                else:
                    # Default to growth tier
                    return "growth"
                    
        except Exception as e:
            logger.error(f"Failed to get organization tier: {e}")
            return "growth"
    
    async def evaluate_feature_flag(
        self, 
        flag_key: str, 
        organization_id: str, 
        user_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> FeatureEvaluation:
        """Evaluate a feature flag for given context"""
        
        start_time = time.time()
        
        try:
            # Get feature flag
            flag = await self.get_feature_flag(flag_key)
            if not flag:
                return FeatureEvaluation(
                    key=flag_key,
                    value=None,
                    enabled=False,
                    tier="unknown",
                    organization_id=organization_id,
                    user_id=user_id,
                    variant=None,
                    evaluation_time=datetime.now(),
                    source="default"
                )
            
            # Get organization tier
            tier = await self.get_organization_tier(organization_id)
            
            # Check if feature is enabled
            if not flag.enabled:
                result = FeatureEvaluation(
                    key=flag_key,
                    value=flag.default_value,
                    enabled=False,
                    tier=tier,
                    organization_id=organization_id,
                    user_id=user_id,
                    variant=None,
                    evaluation_time=datetime.now(),
                    source="disabled"
                )
            else:
                # Evaluate based on rollout strategy
                result = await self.evaluate_rollout_strategy(
                    flag, organization_id, user_id, tier, context
                )
            
            # Log evaluation
            await self.log_evaluation(result)
            
            # Update metrics
            feature_flag_evaluations.labels(
                flag=flag_key, 
                result="enabled" if result.enabled else "disabled"
            ).inc()
            
            feature_usage_by_tier.labels(feature=flag_key, tier=tier).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Feature flag evaluation failed for {flag_key}: {e}")
            return FeatureEvaluation(
                key=flag_key,
                value=None,
                enabled=False,
                tier="unknown",
                organization_id=organization_id,
                user_id=user_id,
                variant=None,
                evaluation_time=datetime.now(),
                source="error"
            )
    
    async def get_feature_flag(self, flag_key: str) -> Optional[FeatureFlag]:
        """Get feature flag from cache or database"""
        
        # Check cache first
        if flag_key in self.flag_cache:
            feature_flag_cache_hits.inc()
            return self.flag_cache[flag_key]
        
        # Query database
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM feature_flags WHERE key = $1", flag_key
                )
                
                if row:
                    flag = FeatureFlag(
                        key=row['key'],
                        name=row['name'],
                        description=row['description'],
                        type=FeatureType(row['type']),
                        default_value=row['default_value'],
                        enabled=row['enabled'],
                        rollout_strategy=RolloutStrategy(row['rollout_strategy']),
                        rollout_config=row['rollout_config'],
                        tier_restrictions=[BillingTier(t) for t in row['tier_restrictions']] if row['tier_restrictions'] else [],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        created_by=row['created_by']
                    )
                    
                    # Cache the flag
                    self.flag_cache[flag_key] = flag
                    return flag
                    
        except Exception as e:
            logger.error(f"Failed to get feature flag {flag_key}: {e}")
        
        return None
    
    async def evaluate_rollout_strategy(
        self, 
        flag: FeatureFlag, 
        organization_id: str, 
        user_id: Optional[str], 
        tier: str,
        context: Optional[Dict]
    ) -> FeatureEvaluation:
        """Evaluate feature flag based on rollout strategy"""
        
        base_result = FeatureEvaluation(
            key=flag.key,
            value=flag.default_value,
            enabled=False,
            tier=tier,
            organization_id=organization_id,
            user_id=user_id,
            variant=None,
            evaluation_time=datetime.now(),
            source="strategy"
        )
        
        # Check tier restrictions first
        if flag.tier_restrictions:
            try:
                tier_enum = BillingTier(tier)
                if tier_enum not in flag.tier_restrictions:
                    base_result.source = "tier_restricted"
                    return base_result
            except ValueError:
                base_result.source = "invalid_tier"
                return base_result
        
        # Evaluate based on strategy
        if flag.rollout_strategy == RolloutStrategy.ALL_USERS:
            base_result.enabled = True
            base_result.value = flag.default_value
            
        elif flag.rollout_strategy == RolloutStrategy.TIER_BASED:
            tier_value = flag.rollout_config.get(tier, flag.default_value)
            base_result.enabled = True
            base_result.value = tier_value
            
        elif flag.rollout_strategy == RolloutStrategy.PERCENTAGE:
            percentage = flag.rollout_config.get("percentage", 0)
            hash_input = f"{flag.key}:{organization_id}"
            hash_value = hash(hash_input) % 100
            
            if hash_value < percentage:
                base_result.enabled = True
                base_result.value = flag.rollout_config.get("value", flag.default_value)
                
        elif flag.rollout_strategy == RolloutStrategy.USER_LIST:
            user_list = flag.rollout_config.get("users", [])
            if user_id and user_id in user_list:
                base_result.enabled = True
                base_result.value = flag.rollout_config.get("value", flag.default_value)
                
        elif flag.rollout_strategy == RolloutStrategy.AB_TEST:
            variant = await self.get_ab_test_variant(
                flag.rollout_config.get("test_name", flag.key),
                organization_id,
                user_id,
                flag.rollout_config.get("variants", {})
            )
            
            if variant:
                variant_config = flag.rollout_config["variants"][variant]
                base_result.enabled = variant_config.get("enabled", False)
                base_result.value = variant_config.get("value", flag.default_value)
                base_result.variant = variant
        
        return base_result
    
    async def get_ab_test_variant(
        self, 
        test_name: str, 
        organization_id: str, 
        user_id: Optional[str],
        variants: Dict[str, Any]
    ) -> Optional[str]:
        """Get A/B test variant assignment"""
        
        try:
            # Check existing assignment
            async with self.db_pool.acquire() as conn:
                existing = await conn.fetchval("""
                    SELECT variant FROM ab_test_assignments
                    WHERE test_name = $1 AND organization_id = $2 AND user_id = $3
                """, test_name, organization_id, user_id)
                
                if existing:
                    return existing
                
                # Assign new variant based on weights
                total_weight = sum(v.get("weight", 0) for v in variants.values())
                if total_weight == 0:
                    return None
                
                # Use hash for consistent assignment
                hash_input = f"{test_name}:{organization_id}:{user_id or 'anonymous'}"
                hash_value = hash(hash_input) % total_weight
                
                current_weight = 0
                selected_variant = None
                
                for variant_name, variant_config in variants.items():
                    current_weight += variant_config.get("weight", 0)
                    if hash_value < current_weight:
                        selected_variant = variant_name
                        break
                
                if selected_variant:
                    # Store assignment
                    await conn.execute("""
                        INSERT INTO ab_test_assignments (test_name, organization_id, user_id, variant)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (test_name, organization_id, user_id) DO NOTHING
                    """, test_name, organization_id, user_id, selected_variant)
                    
                    # Update metrics
                    ab_test_assignments.labels(test=test_name, variant=selected_variant).inc()
                
                return selected_variant
                
        except Exception as e:
            logger.error(f"A/B test variant assignment failed: {e}")
            return None
    
    async def log_evaluation(self, evaluation: FeatureEvaluation):
        """Log feature flag evaluation"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO feature_evaluations (
                        flag_key, organization_id, user_id, tier, value, variant, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, 
                evaluation.key, evaluation.organization_id, evaluation.user_id,
                evaluation.tier, json.dumps(evaluation.value), evaluation.variant, evaluation.source)
        except Exception as e:
            logger.error(f"Failed to log evaluation: {e}")
    
    async def handle_tier_change(self, msg):
        """Handle billing tier changes"""
        try:
            data = json.loads(msg.data.decode())
            organization_id = data.get('organization_id')
            new_tier = data.get('new_tier')
            
            if organization_id and new_tier:
                # Invalidate tier cache
                cache_key = f"tier:{organization_id}"
                if cache_key in self.tier_cache:
                    del self.tier_cache[cache_key]
                
                await self.redis.delete(cache_key)
                
                logger.info(f"Invalidated tier cache for org {organization_id}, new tier: {new_tier}")
                
        except Exception as e:
            logger.error(f"Failed to handle tier change: {e}")
    
    async def get_feature_flags_for_organization(self, organization_id: str) -> Dict[str, Any]:
        """Get all applicable feature flags for an organization"""
        
        tier = await self.get_organization_tier(organization_id)
        result = {}
        
        for flag_key in self.flag_cache.keys():
            evaluation = await self.evaluate_feature_flag(flag_key, organization_id)
            result[flag_key] = {
                "enabled": evaluation.enabled,
                "value": evaluation.value,
                "variant": evaluation.variant
            }
        
        return {
            "organization_id": organization_id,
            "tier": tier,
            "features": result
        }

# Initialize service
feature_service = FeatureFlagService()

@app.on_event("startup")
async def startup_event():
    await feature_service.initialize()

# API Endpoints
@app.get("/api/v1/features/{organization_id}")
async def get_organization_features(organization_id: str):
    """Get all features for an organization"""
    return await feature_service.get_feature_flags_for_organization(organization_id)

@app.get("/api/v1/features/{organization_id}/{flag_key}")
async def evaluate_feature(organization_id: str, flag_key: str, user_id: Optional[str] = None):
    """Evaluate a specific feature flag"""
    evaluation = await feature_service.evaluate_feature_flag(flag_key, organization_id, user_id)
    return {
        "key": evaluation.key,
        "enabled": evaluation.enabled,
        "value": evaluation.value,
        "variant": evaluation.variant,
        "tier": evaluation.tier
    }

@app.get("/api/v1/flags")
async def list_feature_flags():
    """List all feature flags"""
    flags = []
    for flag in feature_service.flag_cache.values():
        flags.append({
            "key": flag.key,
            "name": flag.name,
            "description": flag.description,
            "type": flag.type.value,
            "enabled": flag.enabled,
            "rollout_strategy": flag.rollout_strategy.value,
            "tier_restrictions": [t.value for t in flag.tier_restrictions]
        })
    return {"flags": flags}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "feature-flags"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)