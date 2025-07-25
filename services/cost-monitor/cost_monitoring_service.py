#!/usr/bin/env python3
"""
Xorb Cost Monitoring Service
Comprehensive cost tracking for GPT tokens, Stripe fees, S3 usage, and infrastructure
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import asyncpg
import aioredis
import boto3
import stripe
import openai
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge
from nats.aio.client import Client as NATS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Phase 5.3 Required Metrics
billing_overage_alerts_total = Counter(
    'billing_overage_alerts_total', 
    'Billing overage alerts triggered', 
    ['org_id', 'plan_type', 'overage_type']
)

# Enhanced Prometheus metrics
cost_tracking = Counter('xorb_cost_tracking_total', 'Total cost tracking events', ['service', 'type'])
monthly_costs = Gauge('xorb_monthly_costs_dollars', 'Monthly costs in dollars', ['service', 'category'])
token_usage = Counter('xorb_gpt_token_usage_total', 'GPT token usage', ['model', 'type', 'provider'])
stripe_fees = Counter('xorb_stripe_fees_total', 'Stripe fees in cents')
s3_storage_bytes = Gauge('xorb_s3_storage_bytes', 'S3 storage usage in bytes', ['bucket'])

# OpenRouter specific metrics
openrouter_requests_total = Counter('xorb_openrouter_requests_total', 'OpenRouter API requests', ['model', 'org_id'])
openrouter_tokens_total = Counter('xorb_openrouter_tokens_total', 'OpenRouter tokens used', ['model', 'type'])
plan_usage_ratio = Gauge('xorb_plan_usage_ratio', 'Plan usage ratio', ['org_id', 'resource_type', 'plan_type'])

app = FastAPI(title="Xorb Cost Monitoring Service", version="1.0.0")

class CostCategory(Enum):
    INFRASTRUCTURE = "infrastructure"
    AI_SERVICES = "ai_services"
    PAYMENT_PROCESSING = "payment_processing"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    MONITORING = "monitoring"

class ServiceProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    STRIPE = "stripe"
    AWS = "aws"
    HETZNER = "hetzner"
    GRAFANA_CLOUD = "grafana_cloud"

@dataclass
class CostEntry:
    """Individual cost entry"""
    id: str
    service: ServiceProvider
    category: CostCategory
    amount: Decimal
    currency: str
    description: str
    usage_data: Dict
    timestamp: datetime
    organization_id: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

@dataclass
class CostSummary:
    """Cost summary for a time period"""
    period_start: datetime
    period_end: datetime
    total_cost: Decimal
    breakdown_by_service: Dict[str, Decimal]
    breakdown_by_category: Dict[str, Decimal]
    breakdown_by_organization: Dict[str, Decimal]
    top_cost_drivers: List[Dict]

class CostMonitoringService:
    """Comprehensive cost monitoring service"""
    
    def __init__(self):
        self.db_pool = None
        self.redis = None
        self.nats = None
        
        # Service clients
        self.s3_client = None
        self.stripe = None
        
        # Cost tracking state
        self.current_month_costs = {}
        self.cost_cache = {}
        
        # Pricing data (updated periodically)
        self.pricing = self.initialize_pricing()
        
        # OpenRouter configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", 
            "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")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Plan limits for overage detection
        self.plan_limits = {
            'Growth': {'gpt_spend_weekly': 50.0, 'assets_max': 150},
            'Pro': {'gpt_spend_weekly': 200.0, 'assets_max': 500}, 
            'Enterprise': {'gpt_spend_weekly': 1000.0, 'assets_max': 2000}
        }
        
        # Active overage alerts
        self.active_alerts = set()
    
    def initialize_pricing(self) -> Dict:
        """Initialize pricing data for various services"""
        return {
            "openai": {
                "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
                "gpt-4-32k": {"input": 0.06, "output": 0.12},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
            },
            "anthropic": {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
            },
            "openrouter": {
                "qwen/qwen-coder:free": {"input": 0.0, "output": 0.0},  # Free model
                "qwen/qwen-coder": {"input": 0.0007, "output": 0.0007},
                "anthropic/claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "openai/gpt-4": {"input": 0.03, "output": 0.06}
            },
            "stripe": {
                "processing_fee": 0.029,  # 2.9%
                "fixed_fee": 0.30,       # $0.30 per transaction
                "international_fee": 0.015  # 1.5% additional
            },
            "aws": {
                "s3": {
                    "standard": 0.023,  # per GB/month
                    "intelligent_tiering": 0.0125,
                    "glacier": 0.004,
                    "deep_archive": 0.00099
                },
                "data_transfer": 0.09,  # per GB
                "api_requests": {
                    "get": 0.0004,  # per 1K requests
                    "put": 0.005    # per 1K requests
                }
            },
            "infrastructure": {
                "epyc_vps": 250.00,  # Monthly cost for 16vCPU/32GB EPYC VPS
                "monitoring": 50.00,  # Grafana Cloud/monitoring costs
                "dns_cdn": 20.00     # DNS and CDN costs
            }
        }
    
    async def initialize(self):
        """Initialize the cost monitoring service"""
        logger.info("Initializing Cost Monitoring Service...")
        
        # Database connection
        database_url = os.getenv("DATABASE_URL", "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=5, max_size=20)
        
        # Redis connection
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.redis = await aioredis.create_redis_pool(redis_url)
        
        # NATS connection
        self.nats = NATS()
        await self.nats.connect(os.getenv("NATS_URL", "nats://nats:4222"))
        
        # Initialize service clients
        await self.initialize_service_clients()
        
        # Create database tables
        await self.create_database_tables()
        
        # Subscribe to cost events
        await self.nats.subscribe("costs.gpt_usage", cb=self.handle_gpt_usage)
        await self.nats.subscribe("costs.openrouter_usage", cb=self.handle_openrouter_usage)
        await self.nats.subscribe("costs.stripe_fee", cb=self.handle_stripe_fee)
        
        # Start overage monitoring
        asyncio.create_task(self.monitor_overages())
        
        # Start periodic cost collection
        asyncio.create_task(self.periodic_cost_collection())
        
        logger.info("Cost Monitoring Service initialized")
    
    async def initialize_service_clients(self):
        """Initialize external service clients"""
        
        # AWS S3 client
        if os.getenv("AWS_ACCESS_KEY_ID"):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-west-2")
            )
        
        # Stripe client
        if os.getenv("STRIPE_SECRET_KEY"):
            stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
            self.stripe = stripe
        
        # OpenAI client
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
    
    async def create_database_tables(self):
        """Create cost monitoring database tables"""
        async with self.db_pool.acquire() as conn:
            # Cost entries table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_entries (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    service VARCHAR(50) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    amount DECIMAL(12,4) NOT NULL,
                    currency VARCHAR(3) DEFAULT 'USD',
                    description TEXT NOT NULL,
                    usage_data JSONB,
                    organization_id UUID,
                    period_start TIMESTAMP,
                    period_end TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Cost summaries table (monthly/daily aggregates)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_summaries (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    total_cost DECIMAL(12,4) NOT NULL,
                    breakdown_by_service JSONB NOT NULL,
                    breakdown_by_category JSONB NOT NULL,
                    breakdown_by_organization JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(period_start, period_end)
                )
            """)
            
            # Budget alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_budgets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(100) NOT NULL,
                    service VARCHAR(50),
                    category VARCHAR(50),
                    monthly_limit DECIMAL(10,2) NOT NULL,
                    alert_threshold DECIMAL(5,2) DEFAULT 80.0,
                    organization_id UUID,
                    enabled BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Cost alerts table for overage tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    org_id UUID NOT NULL,
                    alert_type VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    plan_type VARCHAR(50) NOT NULL,
                    triggered_at TIMESTAMP DEFAULT NOW(),
                    resolved_at TIMESTAMP,
                    active BOOLEAN DEFAULT true
                )
            """)
            
            # Create indices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_cost_entries_service ON cost_entries(service)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_cost_entries_period ON cost_entries(period_start, period_end)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_cost_entries_org ON cost_entries(organization_id)")
    
    async def handle_gpt_usage(self, msg):
        """Handle GPT token usage events"""
        try:
            data = json.loads(msg.data.decode())
            
            model = data.get('model', 'gpt-3.5-turbo')
            input_tokens = data.get('input_tokens', 0)
            output_tokens = data.get('output_tokens', 0)
            organization_id = data.get('organization_id')
            
            # Calculate cost
            pricing = self.pricing["openai"].get(model, self.pricing["openai"]["gpt-3.5-turbo"])
            
            input_cost = Decimal(str(input_tokens / 1000 * pricing["input"]))
            output_cost = Decimal(str(output_tokens / 1000 * pricing["output"]))
            total_cost = input_cost + output_cost
            
            # Create cost entry
            cost_entry = CostEntry(
                id=f"gpt_{int(time.time())}_{organization_id}",
                service=ServiceProvider.OPENAI,
                category=CostCategory.AI_SERVICES,
                amount=total_cost,
                currency="USD",
                description=f"GPT {model} usage: {input_tokens} input + {output_tokens} output tokens",
                usage_data={
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_breakdown": {
                        "input_cost": float(input_cost),
                        "output_cost": float(output_cost)
                    }
                },
                timestamp=datetime.now(),
                organization_id=organization_id
            )
            
            await self.record_cost_entry(cost_entry)
            
            # Update metrics
            token_usage.labels(model=model, type="input", provider="openai").inc(input_tokens)
            token_usage.labels(model=model, type="output", provider="openai").inc(output_tokens)
            
            logger.debug(f"Recorded GPT usage cost: ${total_cost} for {model}")
            
        except Exception as e:
            logger.error(f"Failed to handle GPT usage: {e}")
    
    async def handle_stripe_fee(self, msg):
        """Handle Stripe fee events"""
        try:
            data = json.loads(msg.data.decode())
            
            payment_amount = Decimal(str(data.get('amount', 0))) / 100  # Convert cents to dollars
            organization_id = data.get('organization_id')
            
            # Calculate Stripe fees
            processing_fee = payment_amount * Decimal(str(self.pricing["stripe"]["processing_fee"]))
            fixed_fee = Decimal(str(self.pricing["stripe"]["fixed_fee"]))
            total_fee = processing_fee + fixed_fee
            
            # Create cost entry
            cost_entry = CostEntry(
                id=f"stripe_{int(time.time())}_{organization_id}",
                service=ServiceProvider.STRIPE,
                category=CostCategory.PAYMENT_PROCESSING,
                amount=total_fee,
                currency="USD",
                description=f"Stripe processing fee for ${payment_amount}",
                usage_data={
                    "payment_amount": float(payment_amount),
                    "processing_fee": float(processing_fee),
                    "fixed_fee": float(fixed_fee),
                    "payment_id": data.get('payment_id')
                },
                timestamp=datetime.now(),
                organization_id=organization_id
            )
            
            await self.record_cost_entry(cost_entry)
            
            # Update metrics
            stripe_fees.inc(float(total_fee * 100))  # Convert back to cents
            
            logger.debug(f"Recorded Stripe fee: ${total_fee} for payment ${payment_amount}")
            
        except Exception as e:
            logger.error(f"Failed to handle Stripe fee: {e}")
    
    async def handle_openrouter_usage(self, msg):
        """Handle OpenRouter API usage events"""
        try:
            data = json.loads(msg.data.decode())
            
            model = data.get('model', 'qwen/qwen-coder:free')
            input_tokens = data.get('input_tokens', 0)
            output_tokens = data.get('output_tokens', 0)
            organization_id = data.get('organization_id')
            
            # Calculate cost (free for qwen-coder:free)
            pricing = self.pricing["openrouter"].get(model, {"input": 0.0, "output": 0.0})
            
            input_cost = Decimal(str(input_tokens / 1000 * pricing["input"]))
            output_cost = Decimal(str(output_tokens / 1000 * pricing["output"]))
            total_cost = input_cost + output_cost
            
            # Create cost entry
            cost_entry = CostEntry(
                id=f"openrouter_{int(time.time())}_{organization_id}",
                service=ServiceProvider.OPENAI,  # Using OPENAI enum for simplicity
                category=CostCategory.AI_SERVICES,
                amount=total_cost,
                currency="USD",
                description=f"OpenRouter {model} usage: {input_tokens} input + {output_tokens} output tokens",
                usage_data={
                    "model": model,
                    "provider": "openrouter",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_breakdown": {
                        "input_cost": float(input_cost),
                        "output_cost": float(output_cost)
                    },
                    "free_tier": pricing["input"] == 0.0 and pricing["output"] == 0.0
                },
                timestamp=datetime.now(),
                organization_id=organization_id
            )
            
            await self.record_cost_entry(cost_entry)
            
            # Update OpenRouter specific metrics
            openrouter_requests_total.labels(model=model, org_id=organization_id or "unknown").inc()
            openrouter_tokens_total.labels(model=model, type="input").inc(input_tokens)
            openrouter_tokens_total.labels(model=model, type="output").inc(output_tokens)
            
            # Update general token usage metrics
            token_usage.labels(model=model, type="input", provider="openrouter").inc(input_tokens)
            token_usage.labels(model=model, type="output", provider="openrouter").inc(output_tokens)
            
            logger.debug(f"Recorded OpenRouter usage: ${total_cost} for {model}")
            
        except Exception as e:
            logger.error(f"Failed to handle OpenRouter usage: {e}")
    
    async def monitor_overages(self):
        """Monitor for billing overages and trigger alerts"""
        while True:
            try:
                # Get all active organizations
                async with self.db_pool.acquire() as conn:
                    orgs = await conn.fetch("""
                        SELECT id, plan_type FROM orgs WHERE active = true
                    """)
                    
                    for org in orgs:
                        org_id = str(org['id'])
                        plan_type = org['plan_type']
                        
                        await self.check_gpt_overage(org_id, plan_type)
                        await self.check_asset_overage(org_id, plan_type)
                
                # Sleep for 5 minutes between checks
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Overage monitoring failed: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def check_gpt_overage(self, org_id: str, plan_type: str):
        """Check for GPT spending overage"""
        try:
            # Get this week's spending
            week_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = week_start - timedelta(days=week_start.weekday())
            
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COALESCE(SUM(amount), 0) as weekly_spend
                    FROM cost_entries 
                    WHERE organization_id = $1 
                    AND category = 'ai_services' 
                    AND created_at >= $2
                """, org_id, week_start)
                
                weekly_spend = float(result['weekly_spend'])
                limit = self.plan_limits[plan_type]['gpt_spend_weekly']
                usage_ratio = weekly_spend / limit
                
                # Update plan usage ratio metric
                plan_usage_ratio.labels(
                    org_id=org_id, 
                    resource_type='gpt_spend', 
                    plan_type=plan_type
                ).set(usage_ratio)
                
                # Check for overage (>100% of limit)
                alert_key = f"{org_id}_gpt_overage"
                if usage_ratio > 1.0:
                    if alert_key not in self.active_alerts:
                        await self.trigger_overage_alert(
                            org_id, plan_type, 'gpt_overage', 
                            f"GPT spending (${weekly_spend:.2f}) exceeded weekly limit (${limit:.2f})"
                        )
                        self.active_alerts.add(alert_key)
                else:
                    # Remove alert if usage back to normal
                    self.active_alerts.discard(alert_key)
                    
        except Exception as e:
            logger.error(f"Failed to check GPT overage for {org_id}: {e}")
    
    async def check_asset_overage(self, org_id: str, plan_type: str):
        """Check for asset count overage"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(DISTINCT id) as asset_count
                    FROM assets 
                    WHERE org_id = $1 AND active = true
                """, org_id)
                
                asset_count = int(result['asset_count'] or 0)
                limit = self.plan_limits[plan_type]['assets_max']
                usage_ratio = asset_count / limit
                
                # Update plan usage ratio metric
                plan_usage_ratio.labels(
                    org_id=org_id, 
                    resource_type='assets', 
                    plan_type=plan_type
                ).set(usage_ratio)
                
                # Check for overage (>100% of limit)
                alert_key = f"{org_id}_asset_overage"
                if usage_ratio > 1.0:
                    if alert_key not in self.active_alerts:
                        await self.trigger_overage_alert(
                            org_id, plan_type, 'asset_overage',
                            f"Asset count ({asset_count}) exceeded plan limit ({limit})"
                        )
                        self.active_alerts.add(alert_key)
                else:
                    # Remove alert if usage back to normal
                    self.active_alerts.discard(alert_key)
                    
        except Exception as e:
            logger.error(f"Failed to check asset overage for {org_id}: {e}")
    
    async def trigger_overage_alert(self, org_id: str, plan_type: str, overage_type: str, message: str):
        """Trigger billing overage alert"""
        try:
            # Update Prometheus metric
            billing_overage_alerts_total.labels(
                org_id=org_id,
                plan_type=plan_type, 
                overage_type=overage_type
            ).inc()
            
            # Store alert in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO cost_alerts (org_id, alert_type, message, plan_type, triggered_at)
                    VALUES ($1, $2, $3, $4, NOW())
                """, org_id, overage_type, message, plan_type)
            
            # Send notification via NATS
            alert_data = {
                "org_id": org_id,
                "plan_type": plan_type,
                "overage_type": overage_type,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.nats.publish("alerts.billing_overage", json.dumps(alert_data).encode())
            
            logger.warning(f"Billing overage alert triggered", 
                          org_id=org_id, type=overage_type, message=message)
            
        except Exception as e:
            logger.error(f"Failed to trigger overage alert: {e}")
    
    async def record_cost_entry(self, cost_entry: CostEntry):
        """Record a cost entry in the database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO cost_entries (
                        id, service, category, amount, currency, description,
                        usage_data, organization_id, period_start, period_end, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                cost_entry.id, cost_entry.service.value, cost_entry.category.value,
                cost_entry.amount, cost_entry.currency, cost_entry.description,
                json.dumps(cost_entry.usage_data), cost_entry.organization_id,
                cost_entry.period_start, cost_entry.period_end, cost_entry.timestamp)
            
            # Update in-memory tracking
            service_key = cost_entry.service.value
            if service_key not in self.current_month_costs:
                self.current_month_costs[service_key] = Decimal('0')
            
            self.current_month_costs[service_key] += cost_entry.amount
            
            # Update Prometheus metrics
            monthly_costs.labels(
                service=cost_entry.service.value,
                category=cost_entry.category.value
            ).set(float(self.current_month_costs[service_key]))
            
            cost_tracking.labels(
                service=cost_entry.service.value,
                type=cost_entry.category.value
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record cost entry: {e}")
    
    async def collect_s3_costs(self):
        """Collect S3 storage and transfer costs"""
        if not self.s3_client:
            return
        
        try:
            # Get list of buckets
            response = self.s3_client.list_buckets()
            
            total_storage_cost = Decimal('0')
            total_request_cost = Decimal('0')
            
            for bucket in response.get('Buckets', []):
                bucket_name = bucket['Name']
                
                # Get bucket size and object count
                try:
                    cloudwatch = boto3.client('cloudwatch', region_name=os.getenv("AWS_REGION", "us-west-2"))
                    
                    # Get storage metrics
                    storage_metrics = cloudwatch.get_metric_statistics(
                        Namespace='AWS/S3',
                        MetricName='BucketSizeBytes',
                        Dimensions=[
                            {'Name': 'BucketName', 'Value': bucket_name},
                            {'Name': 'StorageType', 'Value': 'StandardStorage'}
                        ],
                        StartTime=datetime.now() - timedelta(days=2),
                        EndTime=datetime.now(),
                        Period=86400,
                        Statistics=['Average']
                    )
                    
                    if storage_metrics['Datapoints']:
                        storage_bytes = storage_metrics['Datapoints'][-1]['Average']
                        storage_gb = storage_bytes / (1024 ** 3)  # Convert to GB
                        
                        # Calculate monthly storage cost
                        storage_cost = Decimal(str(storage_gb * self.pricing["aws"]["s3"]["standard"]))
                        total_storage_cost += storage_cost
                        
                        # Update Prometheus metrics
                        s3_storage_bytes.labels(bucket=bucket_name).set(storage_bytes)
                        
                        logger.debug(f"S3 bucket {bucket_name}: {storage_gb:.2f} GB, ${storage_cost}")
                
                except Exception as e:
                    logger.error(f"Failed to get S3 metrics for {bucket_name}: {e}")
            
            # Record monthly S3 costs
            if total_storage_cost > 0:
                cost_entry = CostEntry(
                    id=f"s3_storage_{int(time.time())}",
                    service=ServiceProvider.AWS,
                    category=CostCategory.STORAGE,
                    amount=total_storage_cost,
                    currency="USD",
                    description=f"S3 storage costs for {len(response.get('Buckets', []))} buckets",
                    usage_data={
                        "bucket_count": len(response.get('Buckets', [])),
                        "total_storage_cost": float(total_storage_cost),
                        "collection_time": datetime.now().isoformat()
                    },
                    timestamp=datetime.now(),
                    period_start=datetime.now().replace(day=1),
                    period_end=datetime.now()
                )
                
                await self.record_cost_entry(cost_entry)
            
        except Exception as e:
            logger.error(f"Failed to collect S3 costs: {e}")
    
    async def collect_infrastructure_costs(self):
        """Record fixed infrastructure costs"""
        try:
            now = datetime.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # EPYC VPS cost
            vps_cost = CostEntry(
                id=f"epyc_vps_{now.year}_{now.month}",
                service=ServiceProvider.HETZNER,
                category=CostCategory.INFRASTRUCTURE,
                amount=Decimal(str(self.pricing["infrastructure"]["epyc_vps"])),
                currency="USD",
                description="EPYC 16vCPU/32GB VPS monthly cost",
                usage_data={
                    "specs": "16 vCPU, 32GB RAM, 600GB Storage",
                    "provider": "Hetzner Cloud",
                    "region": "eu-central"
                },
                timestamp=now,
                period_start=month_start,
                period_end=month_start + timedelta(days=32)
            )
            
            await self.record_cost_entry(vps_cost)
            
            # Monitoring costs
            monitoring_cost = CostEntry(
                id=f"monitoring_{now.year}_{now.month}",
                service=ServiceProvider.GRAFANA_CLOUD,
                category=CostCategory.MONITORING,
                amount=Decimal(str(self.pricing["infrastructure"]["monitoring"])),
                currency="USD",
                description="Grafana Cloud and monitoring services",
                usage_data={
                    "services": ["Grafana Cloud", "Prometheus", "Loki", "Alertmanager"],
                    "metrics_retention": "30 days",
                    "logs_retention": "7 days"
                },
                timestamp=now,
                period_start=month_start,
                period_end=month_start + timedelta(days=32)
            )
            
            await self.record_cost_entry(monitoring_cost)
            
        except Exception as e:
            logger.error(f"Failed to collect infrastructure costs: {e}")
    
    async def periodic_cost_collection(self):
        """Periodic cost collection from various services"""
        while True:
            try:
                logger.info("Starting periodic cost collection...")
                
                # Collect S3 costs
                await self.collect_s3_costs()
                
                # Collect infrastructure costs (monthly)
                now = datetime.now()
                if now.day == 1 and now.hour < 2:  # First day of month, early morning
                    await self.collect_infrastructure_costs()
                
                # Generate monthly summaries
                if now.day == 1:  # First day of month
                    await self.generate_monthly_summary()
                
                # Sleep for 6 hours
                await asyncio.sleep(21600)
                
            except Exception as e:
                logger.error(f"Periodic cost collection failed: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def generate_monthly_summary(self):
        """Generate monthly cost summary"""
        try:
            # Get previous month's data
            now = datetime.now()
            month_start = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
            month_end = now.replace(day=1) - timedelta(days=1)
            
            async with self.db_pool.acquire() as conn:
                # Get cost entries for the month
                rows = await conn.fetch("""
                    SELECT service, category, organization_id, amount, usage_data
                    FROM cost_entries 
                    WHERE created_at >= $1 AND created_at < $2
                """, month_start, month_end)
                
                if not rows:
                    return
                
                # Aggregate costs
                total_cost = Decimal('0')
                by_service = {}
                by_category = {}
                by_organization = {}
                
                for row in rows:
                    amount = row['amount']
                    total_cost += amount
                    
                    # By service
                    service = row['service']
                    by_service[service] = by_service.get(service, Decimal('0')) + amount
                    
                    # By category
                    category = row['category']
                    by_category[category] = by_category.get(category, Decimal('0')) + amount
                    
                    # By organization
                    org_id = row['organization_id']
                    if org_id:
                        by_organization[org_id] = by_organization.get(org_id, Decimal('0')) + amount
                
                # Convert Decimal to float for JSON storage
                by_service_json = {k: float(v) for k, v in by_service.items()}
                by_category_json = {k: float(v) for k, v in by_category.items()}
                by_organization_json = {k: float(v) for k, v in by_organization.items()}
                
                # Store summary
                await conn.execute("""
                    INSERT INTO cost_summaries (
                        period_start, period_end, total_cost,
                        breakdown_by_service, breakdown_by_category, breakdown_by_organization
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (period_start, period_end) DO UPDATE SET
                        total_cost = EXCLUDED.total_cost,
                        breakdown_by_service = EXCLUDED.breakdown_by_service,
                        breakdown_by_category = EXCLUDED.breakdown_by_category,
                        breakdown_by_organization = EXCLUDED.breakdown_by_organization
                """, 
                month_start, month_end, total_cost,
                json.dumps(by_service_json), json.dumps(by_category_json), 
                json.dumps(by_organization_json))
                
                logger.info(f"Generated monthly cost summary: ${total_cost} for {month_start.strftime('%Y-%m')}")
                
        except Exception as e:
            logger.error(f"Failed to generate monthly summary: {e}")
    
    async def get_cost_dashboard_data(self, period_days: int = 30) -> Dict:
        """Get cost dashboard data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            async with self.db_pool.acquire() as conn:
                # Get total costs by service
                service_costs = await conn.fetch("""
                    SELECT service, SUM(amount) as total_cost
                    FROM cost_entries 
                    WHERE created_at >= $1 AND created_at <= $2
                    GROUP BY service
                    ORDER BY total_cost DESC
                """, start_date, end_date)
                
                # Get total costs by category
                category_costs = await conn.fetch("""
                    SELECT category, SUM(amount) as total_cost
                    FROM cost_entries 
                    WHERE created_at >= $1 AND created_at <= $2
                    GROUP BY category
                    ORDER BY total_cost DESC
                """, start_date, end_date)
                
                # Get daily cost trend
                daily_costs = await conn.fetch("""
                    SELECT DATE(created_at) as date, SUM(amount) as daily_cost
                    FROM cost_entries 
                    WHERE created_at >= $1 AND created_at <= $2
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """, start_date, end_date)
                
                # Get top cost drivers
                top_drivers = await conn.fetch("""
                    SELECT description, amount, service, category, created_at
                    FROM cost_entries 
                    WHERE created_at >= $1 AND created_at <= $2
                    ORDER BY amount DESC
                    LIMIT 10
                """, start_date, end_date)
                
                # Calculate total cost
                total_cost = sum(float(row['total_cost']) for row in service_costs)
                
                return {
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                        "days": period_days
                    },
                    "total_cost": total_cost,
                    "breakdown_by_service": [
                        {"service": row['service'], "cost": float(row['total_cost'])}
                        for row in service_costs
                    ],
                    "breakdown_by_category": [
                        {"category": row['category'], "cost": float(row['total_cost'])}
                        for row in category_costs
                    ],
                    "daily_trend": [
                        {"date": row['date'].isoformat(), "cost": float(row['daily_cost'])}
                        for row in daily_costs
                    ],
                    "top_cost_drivers": [
                        {
                            "description": row['description'],
                            "amount": float(row['amount']),
                            "service": row['service'],
                            "category": row['category'],
                            "date": row['created_at'].isoformat()
                        }
                        for row in top_drivers
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}

# Initialize service
cost_service = CostMonitoringService()

@app.on_event("startup")
async def startup_event():
    await cost_service.initialize()

# API Endpoints
@app.get("/api/v1/costs/dashboard")
async def get_cost_dashboard(period_days: int = 30):
    """Get cost dashboard data"""
    return await cost_service.get_cost_dashboard_data(period_days)

@app.get("/api/v1/costs/summary/{year}/{month}")
async def get_monthly_summary(year: int, month: int):
    """Get monthly cost summary"""
    try:
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(days=1)
        
        async with cost_service.db_pool.acquire() as conn:
            summary = await conn.fetchrow("""
                SELECT * FROM cost_summaries
                WHERE period_start = $1 AND period_end = $2
            """, month_start, month_end)
            
            if summary:
                return {
                    "period_start": summary['period_start'].isoformat(),
                    "period_end": summary['period_end'].isoformat(),
                    "total_cost": float(summary['total_cost']),
                    "breakdown_by_service": summary['breakdown_by_service'],
                    "breakdown_by_category": summary['breakdown_by_category'],
                    "breakdown_by_organization": summary['breakdown_by_organization']
                }
            else:
                raise HTTPException(status_code=404, detail="Summary not found")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cost-monitor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)