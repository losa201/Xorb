"""
Application Services - Orchestrate Use Cases

Application services coordinate between multiple use cases
and provide higher-level business operations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Tuple

from ..domain import (
    AgentCapability,
    Campaign,
    CampaignId,
    CampaignStatus,
    Finding,
    FindingId,
    FindingStatus,
    KnowledgeAtom,
    Severity,
    TargetId
)
from .ports import (
    CacheService,
    EventPublisher,
    NotificationService
)
from .use_cases import (
    CreateCampaignCommand,
    CreateCampaignUseCase,
    CreateKnowledgeAtomCommand,
    CreateKnowledgeAtomUseCase,
    ExecuteAgentCommand,
    ExecuteAgentUseCase,
    GenerateEmbeddingCommand,
    GenerateEmbeddingUseCase,
    GetCampaignStatusQuery,
    ListActiveCampaignsQuery,
    SearchSimilarFindingsQuery,
    SearchSimilarFindingsUseCase,
    StartCampaignCommand,
    StartCampaignUseCase,
    TriageFindingCommand,
    TriageFindingUseCase
)

__all__ = [
    "CampaignApplicationService",
    "FindingApplicationService", 
    "KnowledgeApplicationService"
]


class CampaignApplicationService:
    """Application service for campaign operations"""
    
    def __init__(
        self,
        create_campaign_use_case: CreateCampaignUseCase,
        start_campaign_use_case: StartCampaignUseCase,
        execute_agent_use_case: ExecuteAgentUseCase,
        notification_service: NotificationService,
        cache_service: CacheService
    ) -> None:
        self._create_campaign_use_case = create_campaign_use_case
        self._start_campaign_use_case = start_campaign_use_case
        self._execute_agent_use_case = execute_agent_use_case
        self._notification_service = notification_service
        self._cache_service = cache_service
    
    async def create_and_start_campaign(
        self,
        name: str,
        target_id: TargetId,
        max_cost_usd: Decimal,
        max_duration_hours: int,
        max_api_calls: int,
        required_capabilities: List[AgentCapability],
        max_agents: int = 5,
        auto_start: bool = False
    ) -> CampaignId:
        """Create a campaign and optionally start it immediately"""
        
        # Create campaign
        campaign_id = await self._create_campaign_use_case.execute(
            CreateCampaignCommand(
                name=name,
                target_id=target_id,
                max_cost_usd=max_cost_usd,
                max_duration_hours=max_duration_hours,
                max_api_calls=max_api_calls,
                required_capabilities=required_capabilities,
                max_agents=max_agents
            )
        )
        
        # Start campaign if requested
        if auto_start:
            await self._start_campaign_use_case.execute(
                StartCampaignCommand(campaign_id=campaign_id)
            )
        
        return campaign_id
    
    async def execute_campaign_agents(
        self,
        campaign_id: CampaignId,
        parallel_execution: bool = True
    ) -> List[FindingId]:
        """Execute all agents for a campaign"""
        
        # This would fetch the campaign and execute all scheduled agents
        # For now, this is a placeholder that would coordinate agent execution
        
        all_findings = []
        
        # In real implementation, this would:
        # 1. Fetch campaign and scheduled agents
        # 2. Execute agents (parallel or sequential)
        # 3. Monitor budget and stop if exceeded
        # 4. Handle errors and retry logic
        # 5. Update campaign status when complete
        
        return all_findings
    
    async def get_campaign_progress(
        self, 
        campaign_id: CampaignId
    ) -> dict[str, any]:
        """Get detailed campaign progress information"""
        
        # Check cache first
        cache_key = f"campaign_progress:{campaign_id}"
        cached_progress = await self._cache_service.get(cache_key)
        if cached_progress:
            return cached_progress
        
        # This would fetch campaign details and compute progress
        progress = {
            "campaign_id": str(campaign_id),
            "status": "running",
            "agents_completed": 0,
            "agents_total": 0,
            "findings_discovered": 0,
            "budget_used_percentage": 0.0,
            "estimated_completion": None
        }
        
        # Cache for 30 seconds
        await self._cache_service.set(cache_key, progress, ttl_seconds=30)
        
        return progress
    
    async def pause_campaign(self, campaign_id: CampaignId) -> None:
        """Pause a running campaign"""
        
        # This would:
        # 1. Update campaign status to PAUSED
        # 2. Stop any running agents
        # 3. Send notifications
        
        await self._notification_service.send_campaign_update(
            campaign_id=campaign_id,
            status="paused",
            details="Campaign has been paused by user"
        )
    
    async def resume_campaign(self, campaign_id: CampaignId) -> None:
        """Resume a paused campaign"""
        
        # This would:
        # 1. Update campaign status to RUNNING
        # 2. Resume agent execution
        # 3. Send notifications
        
        await self._notification_service.send_campaign_update(
            campaign_id=campaign_id,
            status="resumed",
            details="Campaign has been resumed"
        )


class FindingApplicationService:
    """Application service for finding operations"""
    
    def __init__(
        self,
        triage_finding_use_case: TriageFindingUseCase,
        search_similar_findings_use_case: SearchSimilarFindingsUseCase,
        generate_embedding_use_case: GenerateEmbeddingUseCase,
        notification_service: NotificationService,
        cache_service: CacheService
    ) -> None:
        self._triage_finding_use_case = triage_finding_use_case
        self._search_similar_findings_use_case = search_similar_findings_use_case
        self._generate_embedding_use_case = generate_embedding_use_case
        self._notification_service = notification_service
        self._cache_service = cache_service
    
    async def bulk_triage_findings(
        self,
        finding_ids: List[FindingId],
        new_status: FindingStatus,
        reasoning: Optional[str] = None
    ) -> int:
        """Triage multiple findings at once"""
        
        successful_triages = 0
        
        for finding_id in finding_ids:
            try:
                await self._triage_finding_use_case.execute(
                    TriageFindingCommand(
                        finding_id=finding_id,
                        new_status=new_status,
                        reasoning=reasoning
                    )
                )
                successful_triages += 1
            except Exception as e:
                # Log error but continue with other findings
                await self._notification_service.send_alert(
                    title="Triage Failed",
                    message=f"Failed to triage finding {finding_id}: {str(e)}",
                    severity="warning"
                )
        
        return successful_triages
    
    async def auto_triage_duplicates(
        self,
        finding_id: FindingId,
        similarity_threshold: float = 0.9
    ) -> int:
        """Automatically mark similar findings as duplicates"""
        
        # Find similar findings
        similar_findings = await self._search_similar_findings_use_case.execute(
            SearchSimilarFindingsQuery(
                finding_id=finding_id,
                similarity_threshold=similarity_threshold,
                max_results=50
            )
        )
        
        # Mark similar findings as duplicates
        duplicate_ids = [finding.id for finding, _ in similar_findings]
        
        duplicates_marked = await self.bulk_triage_findings(
            finding_ids=duplicate_ids,
            new_status=FindingStatus.DUPLICATE,
            reasoning=f"Automatically marked as duplicate of {finding_id}"
        )
        
        return duplicates_marked
    
    async def generate_finding_embedding(
        self,
        finding_id: FindingId,
        content: str
    ) -> None:
        """Generate and cache embedding for a finding"""
        
        # Generate embedding
        embedding = await self._generate_embedding_use_case.execute(
            GenerateEmbeddingCommand(
                text=content,
                input_type="passage"  # Findings are passages
            )
        )
        
        # This would update the finding with the embedding
        # In real implementation, would fetch finding and update it
        
        # Cache the embedding
        cache_key = f"finding_embedding:{finding_id}"
        await self._cache_service.set(cache_key, embedding, ttl_seconds=86400)
    
    async def get_finding_insights(
        self,
        finding_id: FindingId
    ) -> dict[str, any]:
        """Get insights and related information for a finding"""
        
        # Check cache
        cache_key = f"finding_insights:{finding_id}"
        cached_insights = await self._cache_service.get(cache_key)
        if cached_insights:
            return cached_insights
        
        # Find similar findings
        similar_findings = await self._search_similar_findings_use_case.execute(
            SearchSimilarFindingsQuery(
                finding_id=finding_id,
                similarity_threshold=0.7,
                max_results=5
            )
        )
        
        insights = {
            "finding_id": str(finding_id),
            "similar_findings_count": len(similar_findings),
            "similar_findings": [
                {
                    "id": str(finding.id),
                    "title": finding.title,
                    "similarity": similarity,
                    "severity": finding.severity.value
                }
                for finding, similarity in similar_findings
            ],
            "recommendations": self._generate_recommendations(similar_findings),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Cache for 1 hour
        await self._cache_service.set(cache_key, insights, ttl_seconds=3600)
        
        return insights
    
    def _generate_recommendations(
        self,
        similar_findings: List[Tuple[Finding, float]]
    ) -> List[str]:
        """Generate recommendations based on similar findings"""
        
        recommendations = []
        
        if len(similar_findings) > 3:
            recommendations.append(
                "Consider if this is a duplicate of an existing finding"
            )
        
        # Check severity distribution
        severities = [finding.severity for finding, _ in similar_findings]
        critical_count = sum(1 for s in severities if s == Severity.CRITICAL)
        
        if critical_count > 1:
            recommendations.append(
                "Similar critical findings found - consider escalation"
            )
        
        return recommendations


class KnowledgeApplicationService:
    """Application service for knowledge management operations"""
    
    def __init__(
        self,
        create_knowledge_atom_use_case: CreateKnowledgeAtomUseCase,
        search_similar_findings_use_case: SearchSimilarFindingsUseCase,
        generate_embedding_use_case: GenerateEmbeddingUseCase,
        cache_service: CacheService
    ) -> None:
        self._create_knowledge_atom_use_case = create_knowledge_atom_use_case
        self._search_similar_findings_use_case = search_similar_findings_use_case
        self._generate_embedding_use_case = generate_embedding_use_case
        self._cache_service = cache_service
    
    async def create_knowledge_from_finding(
        self,
        finding: Finding,
        additional_context: Optional[str] = None
    ) -> str:
        """Create knowledge atoms from a finding"""
        
        # Prepare content
        content = f"Title: {finding.title}\n\nDescription: {finding.description}"
        if additional_context:
            content += f"\n\nContext: {additional_context}"
        
        # Determine atom type based on severity
        atom_type = "vulnerability" if finding.severity in [Severity.HIGH, Severity.CRITICAL] else "indicator"
        
        # Calculate confidence based on finding details
        confidence = 0.8 if finding.evidence else 0.6
        
        # Create knowledge atom
        atom_id = await self._create_knowledge_atom_use_case.execute(
            CreateKnowledgeAtomCommand(
                content=content,
                atom_type=atom_type,
                confidence=confidence,
                tags=[finding.severity.value, "auto-generated"],
                source=f"finding:{finding.id}"
            )
        )
        
        return str(atom_id)
    
    async def enrich_finding_with_knowledge(
        self,
        finding_id: FindingId
    ) -> dict[str, any]:
        """Enrich a finding with related knowledge"""
        
        # This would:
        # 1. Find similar knowledge atoms
        # 2. Extract relevant techniques and indicators
        # 3. Provide context and remediation suggestions
        
        enrichment = {
            "finding_id": str(finding_id),
            "related_techniques": [],
            "indicators": [],
            "remediation_suggestions": [],
            "confidence_score": 0.0
        }
        
        return enrichment
    
    async def update_knowledge_confidence(
        self,
        atom_ids: List[str],
        feedback_type: str  # "positive", "negative", "neutral"
    ) -> None:
        """Update knowledge atom confidence based on feedback"""
        
        # This would adjust confidence scores based on user feedback
        # Positive feedback increases confidence, negative decreases it
        
        confidence_adjustment = {
            "positive": 0.1,
            "negative": -0.1,
            "neutral": 0.0
        }.get(feedback_type, 0.0)
        
        # In real implementation, would fetch and update atoms
        for atom_id in atom_ids:
            cache_key = f"knowledge_feedback:{atom_id}"
            await self._cache_service.set(
                cache_key, 
                {"adjustment": confidence_adjustment, "timestamp": datetime.now(timezone.utc)},
                ttl_seconds=86400
            )