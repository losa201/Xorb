#!/usr/bin/env python3
"""
PTaaS (Penetration Testing as a Service) Main Service
Handles campaign lifecycle, researcher assignment, and test orchestration
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Xorb imports
from xorb_common.auth.jwt_auth import verify_jwt_token
from xorb_common.database.postgresql import get_db_pool
from xorb_common.events.cloud_events import CloudEventPublisher
from xorb_common.logging.structured_logger import get_logger
from xorb_common.metrics.prometheus_metrics import XorbMetrics

logger = get_logger(__name__)

# Metrics
ptaas_campaigns_total = Counter('ptaas_campaigns_total', 'Total PTaaS campaigns created', ['status'])
ptaas_test_duration = Histogram('ptaas_test_duration_seconds', 'PTaaS test execution time')
ptaas_active_researchers = Gauge('ptaas_active_researchers', 'Number of active researchers')

class CampaignStatus(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TestType(str, Enum):
    AUTOMATED = "automated"
    MANUAL = "manual"
    HYBRID = "hybrid"

class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class RulesOfEngagement:
    """Rules of Engagement for PTaaS campaigns"""
    allowed_domains: List[str]
    forbidden_domains: List[str]
    allowed_ports: List[int]
    forbidden_techniques: List[str]
    max_request_rate: int
    business_hours_only: bool
    data_exfiltration_allowed: bool
    social_engineering_allowed: bool
    physical_testing_allowed: bool
    notification_email: str
    emergency_contact: str

class PTaaSCampaignRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    target_domains: List[str] = Field(..., min_items=1)
    test_type: TestType
    severity_threshold: SeverityLevel = SeverityLevel.MEDIUM
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    max_budget: float = Field(gt=0)
    rules_of_engagement: Dict[str, Any]
    required_skills: List[str] = []
    auto_assign_researchers: bool = True

class PTaaSCampaignResponse(BaseModel):
    id: str
    name: str
    description: str
    status: CampaignStatus
    target_domains: List[str]
    test_type: TestType
    created_at: datetime
    created_by: str
    scheduled_start: Optional[datetime]
    scheduled_end: Optional[datetime]
    actual_start: Optional[datetime]
    actual_end: Optional[datetime]
    max_budget: float
    spent_budget: float
    findings_count: int
    assigned_researchers: List[Dict[str, Any]]
    rules_of_engagement: Dict[str, Any]

class PTaaSService:
    def __init__(self):
        self.app = FastAPI(title="Xorb PTaaS Service", version="1.0.0")
        self.db_pool = None
        self.redis_client = None
        self.event_publisher = None
        self.metrics = XorbMetrics("ptaas")
        self.security = HTTPBearer()
        
        self._setup_routes()

    async def startup(self):
        """Initialize service connections"""
        try:
            self.db_pool = await get_db_pool()
            self.redis_client = redis.Redis.from_url("redis://localhost:6379")
            self.event_publisher = CloudEventPublisher("ptaas-service")
            
            # Initialize database schemas
            await self._init_database()
            
            logger.info("PTaaS service started successfully")
        except Exception as e:
            logger.error(f"Failed to start PTaaS service: {e}")
            raise

    async def shutdown(self):
        """Cleanup service connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()

    async def _init_database(self):
        """Initialize database tables for PTaaS"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ptaas_campaigns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'draft',
                    target_domains JSONB NOT NULL,
                    test_type VARCHAR(20) NOT NULL,
                    severity_threshold VARCHAR(20) NOT NULL DEFAULT 'medium',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_by UUID NOT NULL,
                    scheduled_start TIMESTAMP WITH TIME ZONE,
                    scheduled_end TIMESTAMP WITH TIME ZONE,
                    actual_start TIMESTAMP WITH TIME ZONE,
                    actual_end TIMESTAMP WITH TIME ZONE,
                    max_budget DECIMAL(10,2) NOT NULL,
                    spent_budget DECIMAL(10,2) DEFAULT 0.00,
                    findings_count INTEGER DEFAULT 0,
                    rules_of_engagement JSONB NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS ptaas_campaign_researchers (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    campaign_id UUID NOT NULL REFERENCES ptaas_campaigns(id) ON DELETE CASCADE,
                    researcher_id UUID NOT NULL,
                    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    status VARCHAR(20) DEFAULT 'assigned',
                    skills JSONB DEFAULT '[]'::jsonb,
                    allocation_percentage INTEGER DEFAULT 100
                );
                
                CREATE TABLE IF NOT EXISTS ptaas_test_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    campaign_id UUID NOT NULL REFERENCES ptaas_campaigns(id) ON DELETE CASCADE,
                    researcher_id UUID,
                    session_type VARCHAR(20) NOT NULL,
                    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    ended_at TIMESTAMP WITH TIME ZONE,
                    target_url TEXT NOT NULL,
                    findings_count INTEGER DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'active',
                    metadata JSONB DEFAULT '{}'::jsonb
                );
                
                CREATE INDEX IF NOT EXISTS idx_ptaas_campaigns_status ON ptaas_campaigns(status);
                CREATE INDEX IF NOT EXISTS idx_ptaas_campaigns_created_by ON ptaas_campaigns(created_by);
                CREATE INDEX IF NOT EXISTS idx_ptaas_campaign_researchers_campaign ON ptaas_campaign_researchers(campaign_id);
            """)

    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/campaigns", response_model=PTaaSCampaignResponse)
        async def create_campaign(
            request: PTaaSCampaignRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            user_id = await self._verify_token(credentials.credentials)
            campaign = await self._create_campaign(request, user_id)
            
            # Trigger researcher assignment in background
            if request.auto_assign_researchers:
                background_tasks.add_task(self._assign_researchers, campaign["id"])
            
            ptaas_campaigns_total.labels(status="created").inc()
            
            return PTaaSCampaignResponse(**campaign)

        @self.app.get("/campaigns", response_model=List[PTaaSCampaignResponse])
        async def list_campaigns(
            status: Optional[CampaignStatus] = None,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            user_id = await self._verify_token(credentials.credentials)
            campaigns = await self._list_campaigns(user_id, status)
            return [PTaaSCampaignResponse(**campaign) for campaign in campaigns]

        @self.app.get("/campaigns/{campaign_id}", response_model=PTaaSCampaignResponse)
        async def get_campaign(
            campaign_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            user_id = await self._verify_token(credentials.credentials)
            campaign = await self._get_campaign(campaign_id, user_id)
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            return PTaaSCampaignResponse(**campaign)

        @self.app.post("/campaigns/{campaign_id}/start")
        async def start_campaign(
            campaign_id: str,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            user_id = await self._verify_token(credentials.credentials)
            await self._start_campaign(campaign_id, user_id)
            
            # Trigger test execution in background
            background_tasks.add_task(self._execute_campaign_tests, campaign_id)
            
            return {"status": "started", "campaign_id": campaign_id}

        @self.app.post("/campaigns/{campaign_id}/pause")
        async def pause_campaign(
            campaign_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            user_id = await self._verify_token(credentials.credentials)
            await self._pause_campaign(campaign_id, user_id)
            return {"status": "paused", "campaign_id": campaign_id}

        @self.app.get("/campaigns/{campaign_id}/findings")
        async def get_campaign_findings(
            campaign_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            user_id = await self._verify_token(credentials.credentials)
            findings = await self._get_campaign_findings(campaign_id, user_id)
            return {"campaign_id": campaign_id, "findings": findings}

        @self.app.get("/metrics")
        async def get_metrics():
            return generate_latest()

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "ptaas", "timestamp": datetime.utcnow()}

    async def _verify_token(self, token: str) -> str:
        """Verify JWT token and return user ID"""
        try:
            payload = verify_jwt_token(token)
            return payload.get("user_id")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid authentication token")

    async def _create_campaign(self, request: PTaaSCampaignRequest, user_id: str) -> Dict[str, Any]:
        """Create a new PTaaS campaign"""
        campaign_id = str(uuid.uuid4())
        
        # Validate Rules of Engagement
        await self._validate_roe(request.rules_of_engagement)
        
        async with self.db_pool.acquire() as conn:
            campaign_data = await conn.fetchrow("""
                INSERT INTO ptaas_campaigns (
                    id, name, description, target_domains, test_type, 
                    severity_threshold, created_by, scheduled_start, 
                    scheduled_end, max_budget, rules_of_engagement
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING *
            """, 
                campaign_id, request.name, request.description,
                json.dumps(request.target_domains), request.test_type.value,
                request.severity_threshold.value, user_id,
                request.scheduled_start, request.scheduled_end,
                request.max_budget, json.dumps(request.rules_of_engagement)
            )
        
        # Publish campaign creation event
        await self.event_publisher.publish({
            "type": "ptaas.campaign.created",
            "source": "ptaas-service",
            "data": {
                "campaign_id": campaign_id,
                "user_id": user_id,
                "test_type": request.test_type.value
            }
        })
        
        logger.info(f"Created PTaaS campaign {campaign_id} for user {user_id}")
        
        return {
            "id": str(campaign_data["id"]),
            "name": campaign_data["name"],
            "description": campaign_data["description"],
            "status": campaign_data["status"],
            "target_domains": json.loads(campaign_data["target_domains"]),
            "test_type": campaign_data["test_type"],
            "created_at": campaign_data["created_at"],
            "created_by": str(campaign_data["created_by"]),
            "scheduled_start": campaign_data["scheduled_start"],
            "scheduled_end": campaign_data["scheduled_end"],
            "actual_start": campaign_data["actual_start"],
            "actual_end": campaign_data["actual_end"],
            "max_budget": float(campaign_data["max_budget"]),
            "spent_budget": float(campaign_data["spent_budget"]),
            "findings_count": campaign_data["findings_count"],
            "assigned_researchers": [],
            "rules_of_engagement": json.loads(campaign_data["rules_of_engagement"])
        }

    async def _validate_roe(self, roe_data: Dict[str, Any]):
        """Validate Rules of Engagement"""
        required_fields = [
            "allowed_domains", "notification_email", "emergency_contact"
        ]
        
        for field in required_fields:
            if field not in roe_data:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required RoE field: {field}"
                )
        
        # Validate domain format
        allowed_domains = roe_data.get("allowed_domains", [])
        if not allowed_domains:
            raise HTTPException(
                status_code=400,
                detail="At least one allowed domain must be specified"
            )

    async def _assign_researchers(self, campaign_id: str):
        """Automatically assign researchers to campaign based on skills and availability"""
        try:
            # Get campaign details
            async with self.db_pool.acquire() as conn:
                campaign = await conn.fetchrow(
                    "SELECT * FROM ptaas_campaigns WHERE id = $1", 
                    uuid.UUID(campaign_id)
                )
                
                if not campaign:
                    logger.error(f"Campaign {campaign_id} not found for researcher assignment")
                    return
                
                # Get available researchers from researcher service
                # This would call the researcher management service
                available_researchers = await self._get_available_researchers(
                    test_type=campaign["test_type"],
                    required_skills=json.loads(campaign.get("metadata", "{}")).get("required_skills", [])
                )
                
                # Assign top researchers
                for researcher in available_researchers[:3]:  # Assign top 3
                    await conn.execute("""
                        INSERT INTO ptaas_campaign_researchers 
                        (campaign_id, researcher_id, skills, allocation_percentage)
                        VALUES ($1, $2, $3, $4)
                    """, 
                        uuid.UUID(campaign_id), 
                        uuid.UUID(researcher["id"]),
                        json.dumps(researcher.get("skills", [])),
                        33  # Split 100% among 3 researchers
                    )
                
            logger.info(f"Assigned {len(available_researchers)} researchers to campaign {campaign_id}")
            
        except Exception as e:
            logger.error(f"Failed to assign researchers to campaign {campaign_id}: {e}")

    async def _get_available_researchers(self, test_type: str, required_skills: List[str]) -> List[Dict[str, Any]]:
        """Get available researchers matching criteria"""
        # This would integrate with the researcher management service
        # For now, return mock data
        return [
            {
                "id": str(uuid.uuid4()),
                "username": "sec_researcher_1",
                "skills": ["web", "api", "sql_injection"],
                "rating": 1850,
                "availability": "available"
            },
            {
                "id": str(uuid.uuid4()),
                "username": "sec_researcher_2", 
                "skills": ["mobile", "api", "crypto"],
                "rating": 1750,
                "availability": "available"
            }
        ]

    async def _list_campaigns(self, user_id: str, status: Optional[CampaignStatus]) -> List[Dict[str, Any]]:
        """List campaigns for user"""
        query = "SELECT * FROM ptaas_campaigns WHERE created_by = $1"
        params = [uuid.UUID(user_id)]
        
        if status:
            query += " AND status = $2"
            params.append(status.value)
            
        query += " ORDER BY created_at DESC"
        
        async with self.db_pool.acquire() as conn:
            campaigns = await conn.fetch(query, *params)
            
        return [self._campaign_row_to_dict(campaign) for campaign in campaigns]

    async def _get_campaign(self, campaign_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific campaign"""
        async with self.db_pool.acquire() as conn:
            campaign = await conn.fetchrow("""
                SELECT c.*, 
                       array_agg(DISTINCT cr.researcher_id) FILTER (WHERE cr.researcher_id IS NOT NULL) as researcher_ids
                FROM ptaas_campaigns c
                LEFT JOIN ptaas_campaign_researchers cr ON c.id = cr.campaign_id
                WHERE c.id = $1 AND c.created_by = $2
                GROUP BY c.id
            """, uuid.UUID(campaign_id), uuid.UUID(user_id))
            
            if not campaign:
                return None
                
        return self._campaign_row_to_dict(campaign)

    def _campaign_row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        return {
            "id": str(row["id"]),
            "name": row["name"],
            "description": row["description"],
            "status": row["status"],
            "target_domains": json.loads(row["target_domains"]),
            "test_type": row["test_type"],
            "created_at": row["created_at"],
            "created_by": str(row["created_by"]),
            "scheduled_start": row["scheduled_start"],
            "scheduled_end": row["scheduled_end"],
            "actual_start": row["actual_start"],
            "actual_end": row["actual_end"],
            "max_budget": float(row["max_budget"]),
            "spent_budget": float(row["spent_budget"]),
            "findings_count": row["findings_count"],
            "assigned_researchers": row.get("researcher_ids", []) or [],
            "rules_of_engagement": json.loads(row["rules_of_engagement"])
        }

    async def _start_campaign(self, campaign_id: str, user_id: str):
        """Start a PTaaS campaign"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE ptaas_campaigns 
                SET status = 'active', actual_start = NOW(), updated_at = NOW()
                WHERE id = $1 AND created_by = $2 AND status = 'scheduled'
            """, uuid.UUID(campaign_id), uuid.UUID(user_id))

    async def _pause_campaign(self, campaign_id: str, user_id: str):
        """Pause a PTaaS campaign"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE ptaas_campaigns 
                SET status = 'paused', updated_at = NOW()
                WHERE id = $1 AND created_by = $2 AND status = 'active'
            """, uuid.UUID(campaign_id), uuid.UUID(user_id))

    async def _execute_campaign_tests(self, campaign_id: str):
        """Execute automated tests for campaign"""
        try:
            # This would integrate with the Xorb orchestrator
            # to trigger automated security tests
            logger.info(f"Starting automated tests for campaign {campaign_id}")
            
            # Placeholder for orchestrator integration
            await asyncio.sleep(5)  # Simulate test execution
            
            logger.info(f"Completed automated tests for campaign {campaign_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute tests for campaign {campaign_id}: {e}")

    async def _get_campaign_findings(self, campaign_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get findings for a campaign"""
        # This would integrate with the bug bounty service
        # For now return mock data
        return [
            {
                "id": str(uuid.uuid4()),
                "title": "SQL Injection in login endpoint",
                "severity": "high",
                "status": "validated",
                "researcher_id": str(uuid.uuid4()),
                "submitted_at": datetime.utcnow(),
                "cvss_score": 8.5
            }
        ]

# Service entry point
service = PTaaSService()

@service.app.on_event("startup")
async def startup_event():
    await service.startup()

@service.app.on_event("shutdown") 
async def shutdown_event():
    await service.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(service.app, host="0.0.0.0", port=8010)