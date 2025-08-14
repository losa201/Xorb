from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from xorb.shared.models import (
    UnifiedUser, APIKeyModel, UnifiedTarget, UnifiedAgent, UnifiedCampaign,
    ThreatIntelligence, ScanResultModel, ExploitResultModel, EvidenceModel
)

class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_user_by_username(self, username: str) -> UnifiedUser | None:
        result = await self.session.execute(select(UnifiedUser).where(UnifiedUser.username == username))
        return result.scalars().first()

    async def create_user(self, user: UnifiedUser) -> UnifiedUser:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

class APIKeyRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_api_key_by_hash(self, key_hash: str) -> APIKeyModel | None:
        result = await self.session.execute(select(APIKeyModel).where(APIKeyModel.key_hash == key_hash))
        return result.scalars().first()

    async def create_api_key(self, api_key: APIKeyModel) -> APIKeyModel:
        self.session.add(api_key)
        await self.session.commit()
        await self.session.refresh(api_key)
        return api_key

class TargetRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_target_by_id(self, target_id: str) -> UnifiedTarget | None:
        result = await self.session.execute(select(UnifiedTarget).where(UnifiedTarget.id == target_id))
        return result.scalars().first()

    async def create_target(self, target: UnifiedTarget) -> UnifiedTarget:
        self.session.add(target)
        await self.session.commit()
        await self.session.refresh(target)
        return target

class AgentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_agent_by_id(self, agent_id: str) -> UnifiedAgent | None:
        result = await self.session.execute(select(UnifiedAgent).where(UnifiedAgent.id == agent_id))
        return result.scalars().first()

    async def create_agent(self, agent: UnifiedAgent) -> UnifiedAgent:
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)
        return agent

class CampaignRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_campaign_by_id(self, campaign_id: str) -> UnifiedCampaign | None:
        result = await self.session.execute(select(UnifiedCampaign).where(UnifiedCampaign.id == campaign_id))
        return result.scalars().first()

    async def get_all_campaigns(self) -> List[UnifiedCampaign]:
        result = await self.session.execute(select(UnifiedCampaign))
        return result.scalars().all()

    async def create_campaign(self, campaign: UnifiedCampaign) -> UnifiedCampaign:
        self.session.add(campaign)
        await self.session.commit()
        await self.session.refresh(campaign)
        return campaign

class ThreatIntelligenceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_threat_intelligence_by_id(self, threat_id: str) -> ThreatIntelligence | None:
        result = await self.session.execute(select(ThreatIntelligence).where(ThreatIntelligence.id == threat_id))
        return result.scalars().first()

    async def create_threat_intelligence(self, threat_intelligence: ThreatIntelligence) -> ThreatIntelligence:
        self.session.add(threat_intelligence)
        await self.session.commit()
        await self.session.refresh(threat_intelligence)
        return threat_intelligence

class ScanRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_scan_by_id(self, scan_id: str) -> ScanResultModel | None:
        result = await self.session.execute(select(ScanResultModel).where(ScanResultModel.id == scan_id))
        return result.scalars().first()

    async def get_scans_by_target(self, target_id: str) -> List[ScanResultModel]:
        result = await self.session.execute(select(ScanResultModel).where(ScanResultModel.target_id == target_id))
        return result.scalars().all()

    async def create_scan(self, scan: ScanResultModel) -> ScanResultModel:
        self.session.add(scan)
        await self.session.commit()
        await self.session.refresh(scan)
        return scan

    async def update_scan(self, scan: ScanResultModel) -> ScanResultModel:
        await self.session.commit()
        await self.session.refresh(scan)
        return scan

class ExploitRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_exploit_by_id(self, exploit_id: str) -> ExploitResultModel | None:
        result = await self.session.execute(select(ExploitResultModel).where(ExploitResultModel.id == exploit_id))
        return result.scalars().first()

    async def get_exploits_by_target(self, target_id: str) -> List[ExploitResultModel]:
        result = await self.session.execute(select(ExploitResultModel).where(ExploitResultModel.target_id == target_id))
        return result.scalars().all()

    async def create_exploit(self, exploit: ExploitResultModel) -> ExploitResultModel:
        self.session.add(exploit)
        await self.session.commit()
        await self.session.refresh(exploit)
        return exploit

    async def update_exploit(self, exploit: ExploitResultModel) -> ExploitResultModel:
        await self.session.commit()
        await self.session.refresh(exploit)
        return exploit

class EvidenceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_evidence_by_id(self, evidence_id: str) -> EvidenceModel | None:
        result = await self.session.execute(select(EvidenceModel).where(EvidenceModel.id == evidence_id))
        return result.scalars().first()

    async def get_evidence_by_target(self, target_id: str) -> List[EvidenceModel]:
        result = await self.session.execute(select(EvidenceModel).where(EvidenceModel.target_id == target_id))
        return result.scalars().all()

    async def get_evidence_by_scan(self, scan_id: str) -> List[EvidenceModel]:
        result = await self.session.execute(select(EvidenceModel).where(EvidenceModel.scan_id == scan_id))
        return result.scalars().all()

    async def create_evidence(self, evidence: EvidenceModel) -> EvidenceModel:
        self.session.add(evidence)
        await self.session.commit()
        await self.session.refresh(evidence)
        return evidence

    async def update_evidence(self, evidence: EvidenceModel) -> EvidenceModel:
        await self.session.commit()
        await self.session.refresh(evidence)
        return evidence
