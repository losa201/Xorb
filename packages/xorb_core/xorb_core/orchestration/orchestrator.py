#!/usr/bin/env python3

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
import redis.asyncio as redis

from .scheduler import CampaignScheduler
from .audit_logger import AuditLogger
from .roe_compliance import RoEValidator


class CampaignStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class CampaignPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Target:
    hostname: str
    ip_address: Optional[str] = None
    ports: List[int] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    scope: str = "in-scope"
    confidence: float = 1.0


@dataclass
class Campaign:
    id: str
    name: str
    targets: List[Target]
    status: CampaignStatus = CampaignStatus.PENDING
    priority: CampaignPriority = CampaignPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    agent_assignments: Dict[str, List[str]] = field(default_factory=dict)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.scheduler = CampaignScheduler()
        self.audit_logger = AuditLogger()
        self.roe_validator = RoEValidator()
        
        self.campaigns: Dict[str, Campaign] = {}
        self.active_agents: Dict[str, Any] = {}
        self.max_concurrent_campaigns = 3
        self.max_agents_per_campaign = 5
        
        self.logger = logging.getLogger(__name__)

    async def start(self):
        self.logger.info("Starting XORB Orchestrator")
        await self.audit_logger.log_event("orchestrator_start", {"timestamp": datetime.utcnow()})
        
        await self.redis_client.ping()
        self.logger.info("Connected to Redis")
        
        await self._load_persisted_campaigns()
        await self._start_scheduler()

    async def create_campaign(self, name: str, targets: List[Dict], priority: CampaignPriority = CampaignPriority.MEDIUM, metadata: Optional[Dict] = None) -> str:
        campaign_id = str(uuid.uuid4())
        
        target_objects = []
        for target_data in targets:
            target = Target(**target_data)
            if not await self.roe_validator.validate_target(target):
                raise ValueError(f"Target {target.hostname} violates Rules of Engagement")
            target_objects.append(target)
        
        campaign = Campaign(
            id=campaign_id,
            name=name,
            targets=target_objects,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.campaigns[campaign_id] = campaign
        await self._persist_campaign(campaign)
        
        await self.audit_logger.log_event("campaign_created", {
            "campaign_id": campaign_id,
            "name": name,
            "target_count": len(target_objects),
            "priority": priority.value
        })
        
        self.logger.info(f"Created campaign {campaign_id}: {name}")
        return campaign_id

    async def start_campaign(self, campaign_id: str) -> bool:
        if campaign_id not in self.campaigns:
            self.logger.error(f"Campaign {campaign_id} not found")
            return False
        
        campaign = self.campaigns[campaign_id]
        
        if len([c for c in self.campaigns.values() if c.status == CampaignStatus.RUNNING]) >= self.max_concurrent_campaigns:
            self.logger.warning(f"Maximum concurrent campaigns reached, queuing {campaign_id}")
            await self.scheduler.queue_campaign(campaign_id)
            return True
        
        campaign.status = CampaignStatus.RUNNING
        campaign.started_at = datetime.utcnow()
        
        await self._assign_agents(campaign)
        await self._persist_campaign(campaign)
        
        await self.audit_logger.log_event("campaign_started", {
            "campaign_id": campaign_id,
            "agent_count": sum(len(agents) for agents in campaign.agent_assignments.values())
        })
        
        self.logger.info(f"Started campaign {campaign_id}")
        return True

    async def pause_campaign(self, campaign_id: str) -> bool:
        if campaign_id not in self.campaigns:
            return False
        
        campaign = self.campaigns[campaign_id]
        if campaign.status != CampaignStatus.RUNNING:
            return False
        
        campaign.status = CampaignStatus.PAUSED
        await self._pause_agents(campaign)
        await self._persist_campaign(campaign)
        
        await self.audit_logger.log_event("campaign_paused", {"campaign_id": campaign_id})
        self.logger.info(f"Paused campaign {campaign_id}")
        return True

    async def stop_campaign(self, campaign_id: str, reason: str = "manual_stop") -> bool:
        if campaign_id not in self.campaigns:
            return False
        
        campaign = self.campaigns[campaign_id]
        campaign.status = CampaignStatus.COMPLETED if reason != "abort" else CampaignStatus.ABORTED
        campaign.completed_at = datetime.utcnow()
        
        await self._stop_agents(campaign)
        await self._persist_campaign(campaign)
        
        await self.audit_logger.log_event("campaign_stopped", {
            "campaign_id": campaign_id,
            "reason": reason,
            "duration": (campaign.completed_at - campaign.started_at).total_seconds() if campaign.started_at else 0
        })
        
        self.logger.info(f"Stopped campaign {campaign_id}: {reason}")
        return True

    async def get_campaign_status(self, campaign_id: str) -> Optional[Dict]:
        if campaign_id not in self.campaigns:
            return None
        
        campaign = self.campaigns[campaign_id]
        return {
            "id": campaign.id,
            "name": campaign.name,
            "status": campaign.status.value,
            "priority": campaign.priority.value,
            "created_at": campaign.created_at.isoformat(),
            "started_at": campaign.started_at.isoformat() if campaign.started_at else None,
            "completed_at": campaign.completed_at.isoformat() if campaign.completed_at else None,
            "target_count": len(campaign.targets),
            "findings_count": len(campaign.findings),
            "active_agents": list(campaign.agent_assignments.keys())
        }

    async def list_campaigns(self, status_filter: Optional[CampaignStatus] = None) -> List[Dict]:
        campaigns = []
        for campaign in self.campaigns.values():
            if status_filter is None or campaign.status == status_filter:
                campaigns.append(await self.get_campaign_status(campaign.id))
        
        return sorted(campaigns, key=lambda x: x["created_at"], reverse=True)

    async def add_finding(self, campaign_id: str, finding: Dict[str, Any]) -> bool:
        if campaign_id not in self.campaigns:
            return False
        
        finding["id"] = str(uuid.uuid4())
        finding["timestamp"] = datetime.utcnow().isoformat()
        finding["campaign_id"] = campaign_id
        
        self.campaigns[campaign_id].findings.append(finding)
        await self._persist_campaign(self.campaigns[campaign_id])
        
        await self.audit_logger.log_event("finding_added", {
            "campaign_id": campaign_id,
            "finding_id": finding["id"],
            "severity": finding.get("severity", "unknown")
        })
        
        self.logger.info(f"Added finding to campaign {campaign_id}: {finding.get('title', 'Unknown')}")
        return True

    async def _assign_agents(self, campaign: Campaign):
        agent_types = ["recon", "web_crawler", "vulnerability_scanner"]
        
        for agent_type in agent_types:
            if len(campaign.agent_assignments.get(agent_type, [])) >= self.max_agents_per_campaign:
                continue
            
            agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
            if agent_type not in campaign.agent_assignments:
                campaign.agent_assignments[agent_type] = []
            campaign.agent_assignments[agent_type].append(agent_id)
            
            self.logger.debug(f"Assigned {agent_type} agent {agent_id} to campaign {campaign.id}")

    async def _pause_agents(self, campaign: Campaign):
        for agent_type, agent_ids in campaign.agent_assignments.items():
            for agent_id in agent_ids:
                self.logger.debug(f"Pausing agent {agent_id}")

    async def _stop_agents(self, campaign: Campaign):
        for agent_type, agent_ids in campaign.agent_assignments.items():
            for agent_id in agent_ids:
                self.logger.debug(f"Stopping agent {agent_id}")

    async def _persist_campaign(self, campaign: Campaign):
        campaign_data = {
            "id": campaign.id,
            "name": campaign.name,
            "status": campaign.status.value,
            "priority": campaign.priority.value,
            "created_at": campaign.created_at.isoformat(),
            "started_at": campaign.started_at.isoformat() if campaign.started_at else None,
            "completed_at": campaign.completed_at.isoformat() if campaign.completed_at else None,
            "targets": [{"hostname": t.hostname, "ip_address": t.ip_address, "ports": t.ports} for t in campaign.targets],
            "agent_assignments": campaign.agent_assignments,
            "findings": campaign.findings,
            "metadata": campaign.metadata
        }
        
        await self.redis_client.hset(f"campaign:{campaign.id}", mapping={
            "data": str(campaign_data)
        })

    async def _load_persisted_campaigns(self):
        keys = await self.redis_client.keys("campaign:*")
        for key in keys:
            campaign_data = await self.redis_client.hget(key, "data")
            if campaign_data:
                data = eval(campaign_data)
                campaign_id = data["id"]
                self.logger.debug(f"Loaded persisted campaign {campaign_id}")

    async def _start_scheduler(self):
        self.scheduler.start()
        self.logger.info("Campaign scheduler started")

    async def shutdown(self):
        self.logger.info("Shutting down XORB Orchestrator")
        await self.scheduler.stop()
        
        for campaign in self.campaigns.values():
            if campaign.status == CampaignStatus.RUNNING:
                await self.stop_campaign(campaign.id, "orchestrator_shutdown")
        
        await self.redis_client.close()
        await self.audit_logger.log_event("orchestrator_shutdown", {"timestamp": datetime.utcnow()})


if __name__ == "__main__":
    import sys
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="XORB Orchestrator")
    parser.add_argument("--campaign", help="Start a test campaign")
    args = parser.parse_args()
    
    async def main():
        orchestrator = Orchestrator()
        await orchestrator.start()
        
        if args.campaign:
            campaign_id = await orchestrator.create_campaign(
                name="Test Campaign",
                targets=[{"hostname": "example.com", "ports": [80, 443]}],
                priority=CampaignPriority.HIGH
            )
            await orchestrator.start_campaign(campaign_id)
            print(f"Started campaign: {campaign_id}")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await orchestrator.shutdown()
    
    asyncio.run(main())