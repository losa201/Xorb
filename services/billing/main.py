#!/usr/bin/env python3
"""
Xorb Tiered Billing Engine
Handles subscription tiers, metered usage, and payment processing
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import stripe
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, validator
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
billing_operations = Counter('xorb_billing_operations_total', 'Total billing operations', ['operation', 'tier'])
billing_duration = Histogram('xorb_billing_operation_duration_seconds', 'Billing operation duration')
monthly_revenue = Gauge('xorb_monthly_revenue_dollars', 'Monthly revenue in dollars')
active_subscriptions = Gauge('xorb_active_subscriptions_total', 'Active subscriptions', ['tier'])

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

app = FastAPI(title="Xorb Billing Engine", version="1.0.0")

# Billing Models
class BillingTier(BaseModel):
    """Subscription tier definition"""
    name: str
    price_monthly: Decimal
    stripe_price_id: str
    features: Dict[str, any]
    usage_limits: Dict[str, int]

class UsageMetric(BaseModel):
    """Usage metric tracking"""
    metric_name: str
    value: int
    timestamp: datetime
    organization_id: str

class BillingEvent(BaseModel):
    """Billing event for audit trail"""
    event_type: str
    organization_id: str
    amount: Optional[Decimal] = None
    metadata: Dict[str, any] = {}

# Tier definitions
BILLING_TIERS = {
    "growth": BillingTier(
        name="Growth",
        price_monthly=Decimal("99.00"),
        stripe_price_id=os.getenv("STRIPE_GROWTH_PRICE_ID", "price_growth"),
        features={
            "scans_per_month": 1000,
            "concurrent_scans": 5,
            "api_calls_per_day": 10000,
            "findings_storage_months": 12,
            "support_level": "email",
            "custom_rules": False,
            "api_access": True,
            "researcher_portal": True
        },
        usage_limits={
            "scans": 1000,
            "api_calls": 10000,
            "storage_gb": 50
        }
    ),
    "elite": BillingTier(
        name="Elite",
        price_monthly=Decimal("299.00"),
        stripe_price_id=os.getenv("STRIPE_ELITE_PRICE_ID", "price_elite"),
        features={
            "scans_per_month": 5000,
            "concurrent_scans": 20,
            "api_calls_per_day": 50000,
            "findings_storage_months": 24,
            "support_level": "priority",
            "custom_rules": True,
            "api_access": True,
            "researcher_portal": True,
            "advanced_analytics": True,
            "custom_integrations": True
        },
        usage_limits={
            "scans": 5000,
            "api_calls": 50000,
            "storage_gb": 200
        }
    ),
    "enterprise": BillingTier(
        name="Enterprise",
        price_monthly=Decimal("999.00"),
        stripe_price_id=os.getenv("STRIPE_ENTERPRISE_PRICE_ID", "price_enterprise"),
        features={
            "scans_per_month": -1,  # unlimited
            "concurrent_scans": 100,
            "api_calls_per_day": -1,  # unlimited
            "findings_storage_months": -1,  # unlimited
            "support_level": "dedicated",
            "custom_rules": True,
            "api_access": True,
            "researcher_portal": True,
            "advanced_analytics": True,
            "custom_integrations": True,
            "sso": True,
            "white_label": True,
            "dedicated_instance": True
        },
        usage_limits={
            "scans": -1,  # unlimited
            "api_calls": -1,  # unlimited
            "storage_gb": -1  # unlimited
        }
    )
}

class BillingEngine:
    """Core billing engine with tiered pricing and usage tracking"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize database connections"""
        if self.initialized:
            return
        
        # PostgreSQL connection
        database_url = os.getenv("DATABASE_URL", "postgresql://xorb:xorb_secure_2024@localhost:5432/xorb_ptaas")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=5, max_size=20)
        
        # Redis connection for usage caching
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(redis_url)
        
        # Create billing tables if they don't exist
        await self.create_billing_tables()
        
        self.initialized = True
        logger.info("Billing engine initialized")
    
    async def create_billing_tables(self):
        """Create billing-related database tables"""
        async with self.db_pool.acquire() as conn:
            # Subscriptions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id SERIAL PRIMARY KEY,
                    organization_id UUID NOT NULL,
                    tier VARCHAR(50) NOT NULL,
                    stripe_subscription_id VARCHAR(100) UNIQUE,
                    stripe_customer_id VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'active',
                    current_period_start TIMESTAMP,
                    current_period_end TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Usage metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_metrics (
                    id SERIAL PRIMARY KEY,
                    organization_id UUID NOT NULL,
                    metric_name VARCHAR(50) NOT NULL,
                    value INTEGER NOT NULL,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    billing_period DATE NOT NULL
                )
            """)
            
            # Billing events table (audit trail)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS billing_events (
                    id SERIAL PRIMARY KEY,
                    organization_id UUID NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    amount DECIMAL(10,2),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Overages table (usage beyond limits)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS overages (
                    id SERIAL PRIMARY KEY,
                    organization_id UUID NOT NULL,
                    metric_name VARCHAR(50) NOT NULL,
                    overage_amount INTEGER NOT NULL,
                    billing_period DATE NOT NULL,
                    rate_per_unit DECIMAL(10,4),
                    total_charge DECIMAL(10,2),
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_org ON subscriptions(organization_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_metrics_org_period ON usage_metrics(organization_id, billing_period)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_billing_events_org ON billing_events(organization_id)")
    
    async def get_organization_subscription(self, organization_id: str) -> Optional[Dict]:
        """Get current subscription for organization"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM subscriptions 
                WHERE organization_id = $1 AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
            """, organization_id)
            
            if row:
                return dict(row)
            return None
    
    async def create_subscription(self, organization_id: str, tier: str, stripe_customer_id: str) -> Dict:
        """Create new subscription"""
        if tier not in BILLING_TIERS:
            raise ValueError(f"Invalid tier: {tier}")
        
        tier_config = BILLING_TIERS[tier]
        
        try:
            # Create Stripe subscription
            stripe_subscription = stripe.Subscription.create(
                customer=stripe_customer_id,
                items=[{"price": tier_config.stripe_price_id}],
                metadata={
                    "organization_id": organization_id,
                    "tier": tier
                }
            )
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                subscription_id = await conn.fetchval("""
                    INSERT INTO subscriptions (
                        organization_id, tier, stripe_subscription_id, 
                        stripe_customer_id, current_period_start, current_period_end
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, 
                organization_id, tier, stripe_subscription.id, stripe_customer_id,
                datetime.fromtimestamp(stripe_subscription.current_period_start),
                datetime.fromtimestamp(stripe_subscription.current_period_end)
                )
            
            # Log billing event
            await self.log_billing_event(organization_id, "subscription_created", 
                                        tier_config.price_monthly, {"tier": tier})
            
            # Update metrics
            billing_operations.labels(operation="subscription_create", tier=tier).inc()
            active_subscriptions.labels(tier=tier).inc()
            
            logger.info(f"Created {tier} subscription for org {organization_id}")
            return {
                "id": subscription_id,
                "stripe_subscription_id": stripe_subscription.id,
                "tier": tier,
                "status": "active"
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating subscription: {e}")
            raise HTTPException(status_code=400, detail=f"Payment error: {str(e)}")
    
    async def record_usage(self, organization_id: str, metric_name: str, value: int):
        """Record usage metric for billing"""
        current_period = datetime.now().replace(day=1).date()  # First day of current month
        
        # Cache in Redis for real-time usage tracking
        cache_key = f"usage:{organization_id}:{metric_name}:{current_period}"
        await self.redis_client.incrby(cache_key, value)
        await self.redis_client.expire(cache_key, 86400 * 35)  # Expire after 35 days
        
        # Store in database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO usage_metrics (organization_id, metric_name, value, billing_period)
                VALUES ($1, $2, $3, $4)
            """, organization_id, metric_name, value, current_period)
        
        # Check for usage limit violations
        await self.check_usage_limits(organization_id, metric_name)
    
    async def get_current_usage(self, organization_id: str, metric_name: str) -> int:
        """Get current usage for the billing period"""
        current_period = datetime.now().replace(day=1).date()
        
        # Try Redis cache first
        cache_key = f"usage:{organization_id}:{metric_name}:{current_period}"
        cached_value = await self.redis_client.get(cache_key)
        
        if cached_value:
            return int(cached_value)
        
        # Fallback to database
        async with self.db_pool.acquire() as conn:
            usage = await conn.fetchval("""
                SELECT COALESCE(SUM(value), 0) FROM usage_metrics
                WHERE organization_id = $1 AND metric_name = $2 AND billing_period = $3
            """, organization_id, metric_name, current_period)
            
            # Update cache
            await self.redis_client.set(cache_key, usage or 0, ex=86400)
            return usage or 0
    
    async def check_usage_limits(self, organization_id: str, metric_name: str):
        """Check if usage exceeds tier limits and handle overages"""
        subscription = await self.get_organization_subscription(organization_id)
        if not subscription:
            return
        
        tier_config = BILLING_TIERS[subscription["tier"]]
        usage_limit = tier_config.usage_limits.get(metric_name)
        
        # Skip check for unlimited (-1) limits
        if usage_limit == -1:
            return
        
        current_usage = await self.get_current_usage(organization_id, metric_name)
        
        if current_usage > usage_limit:
            overage = current_usage - usage_limit
            await self.handle_overage(organization_id, metric_name, overage)
    
    async def handle_overage(self, organization_id: str, metric_name: str, overage_amount: int):
        """Handle usage overages with additional billing"""
        current_period = datetime.now().replace(day=1).date()
        
        # Overage rates (per unit)
        overage_rates = {
            "scans": Decimal("0.10"),      # $0.10 per scan
            "api_calls": Decimal("0.001"),  # $0.001 per API call
            "storage_gb": Decimal("1.00")   # $1.00 per GB
        }
        
        rate_per_unit = overage_rates.get(metric_name, Decimal("0.01"))
        total_charge = rate_per_unit * overage_amount
        
        # Check if overage already recorded for this period
        async with self.db_pool.acquire() as conn:
            existing = await conn.fetchval("""
                SELECT id FROM overages 
                WHERE organization_id = $1 AND metric_name = $2 AND billing_period = $3
            """, organization_id, metric_name, current_period)
            
            if not existing:
                # Record new overage
                await conn.execute("""
                    INSERT INTO overages (
                        organization_id, metric_name, overage_amount, 
                        billing_period, rate_per_unit, total_charge
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, organization_id, metric_name, overage_amount, 
                     current_period, rate_per_unit, total_charge)
                
                # Log billing event
                await self.log_billing_event(organization_id, "overage_charged", 
                                            total_charge, {
                                                "metric": metric_name,
                                                "overage_amount": overage_amount,
                                                "rate_per_unit": float(rate_per_unit)
                                            })
                
                logger.warning(f"Overage recorded for org {organization_id}: "
                             f"{overage_amount} {metric_name} = ${total_charge}")
    
    async def log_billing_event(self, organization_id: str, event_type: str, 
                               amount: Optional[Decimal] = None, metadata: Dict = None):
        """Log billing event for audit trail"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO billing_events (organization_id, event_type, amount, metadata)
                VALUES ($1, $2, $3, $4)
            """, organization_id, event_type, amount, json.dumps(metadata or {}))
    
    async def calculate_monthly_bill(self, organization_id: str, billing_month: datetime) -> Dict:
        """Calculate total monthly bill including overages"""
        subscription = await self.get_organization_subscription(organization_id)
        if not subscription:
            return {"error": "No active subscription"}
        
        tier_config = BILLING_TIERS[subscription["tier"]]
        base_charge = tier_config.price_monthly
        
        # Calculate overages for the month
        billing_period = billing_month.replace(day=1).date()
        
        async with self.db_pool.acquire() as conn:
            overages = await conn.fetch("""
                SELECT metric_name, overage_amount, rate_per_unit, total_charge
                FROM overages 
                WHERE organization_id = $1 AND billing_period = $2
            """, organization_id, billing_period)
        
        overage_charges = []
        total_overages = Decimal("0.00")
        
        for overage in overages:
            overage_charge = {
                "metric": overage["metric_name"],
                "overage_amount": overage["overage_amount"],
                "rate_per_unit": overage["rate_per_unit"],
                "charge": overage["total_charge"]
            }
            overage_charges.append(overage_charge)
            total_overages += overage["total_charge"]
        
        return {
            "organization_id": organization_id,
            "billing_period": billing_period.isoformat(),
            "tier": subscription["tier"],
            "base_charge": float(base_charge),
            "overage_charges": overage_charges,
            "total_overages": float(total_overages),
            "total_amount": float(base_charge + total_overages)
        }
    
    async def get_usage_dashboard(self, organization_id: str) -> Dict:
        """Get usage dashboard data for organization"""
        subscription = await self.get_organization_subscription(organization_id)
        if not subscription:
            return {"error": "No active subscription"}
        
        tier_config = BILLING_TIERS[subscription["tier"]]
        current_period = datetime.now().replace(day=1).date()
        
        usage_data = {}
        for metric_name, limit in tier_config.usage_limits.items():
            current_usage = await self.get_current_usage(organization_id, metric_name)
            
            usage_data[metric_name] = {
                "current_usage": current_usage,
                "limit": limit if limit != -1 else "unlimited",
                "percentage_used": (current_usage / limit * 100) if limit > 0 else 0,
                "overage": max(0, current_usage - limit) if limit > 0 else 0
            }
        
        return {
            "organization_id": organization_id,
            "tier": subscription["tier"],
            "billing_period": current_period.isoformat(),
            "usage": usage_data,
            "tier_features": tier_config.features
        }

# Initialize billing engine
billing_engine = BillingEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize billing engine on startup"""
    await billing_engine.initialize()

# API Endpoints
@app.post("/api/v1/billing/subscriptions")
async def create_subscription_endpoint(
    organization_id: str,
    tier: str,
    stripe_customer_id: str,
    background_tasks: BackgroundTasks
):
    """Create new subscription"""
    with billing_duration.time():
        subscription = await billing_engine.create_subscription(
            organization_id, tier, stripe_customer_id
        )
    return subscription

@app.get("/api/v1/billing/subscriptions/{organization_id}")
async def get_subscription_endpoint(organization_id: str):
    """Get organization subscription"""
    subscription = await billing_engine.get_organization_subscription(organization_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return subscription

@app.post("/api/v1/billing/usage")
async def record_usage_endpoint(
    organization_id: str,
    metric_name: str,
    value: int,
    background_tasks: BackgroundTasks
):
    """Record usage metric"""
    background_tasks.add_task(
        billing_engine.record_usage, organization_id, metric_name, value
    )
    return {"status": "recorded"}

@app.get("/api/v1/billing/usage/{organization_id}")
async def get_usage_dashboard_endpoint(organization_id: str):
    """Get usage dashboard"""
    dashboard = await billing_engine.get_usage_dashboard(organization_id)
    return dashboard

@app.get("/api/v1/billing/bill/{organization_id}")
async def calculate_bill_endpoint(organization_id: str, month: Optional[str] = None):
    """Calculate monthly bill"""
    if month:
        billing_month = datetime.fromisoformat(month)
    else:
        billing_month = datetime.now()
    
    bill = await billing_engine.calculate_monthly_bill(organization_id, billing_month)
    return bill

@app.get("/api/v1/billing/tiers")
async def get_billing_tiers_endpoint():
    """Get available billing tiers"""
    tiers = {}
    for tier_name, tier_config in BILLING_TIERS.items():
        tiers[tier_name] = {
            "name": tier_config.name,
            "price_monthly": float(tier_config.price_monthly),
            "features": tier_config.features,
            "usage_limits": tier_config.usage_limits
        }
    return tiers

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "billing-engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)