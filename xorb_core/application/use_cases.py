"""
Use Cases - Application-specific business rules

Use cases orchestrate the flow of data to and from entities,
and direct those entities to use their critical business rules
to achieve the goals of the use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Tuple
from uuid import UUID

from ..domain import (
    Agent,
    AgentCapability,
    AgentId,
    AtomId,
    AtomType,
    BudgetLimit,
    Campaign,
    CampaignId,
    CampaignStatus,
    Embedding,
    Finding,
    FindingId,
    FindingStatus,
    KnowledgeAtom,
    Severity,
    Target,
    TargetId,
    # Events
    AgentExecutionCompleted,
    AgentExecutionStarted,
    CampaignCompleted,
    CampaignStarted,
    EmbeddingGenerated,
    FindingDiscovered,
    FindingTriaged,
    KnowledgeAtomCreated,
    # Services
    AgentSelectionService,
    BudgetManagementService,
    CampaignOrchestratorService,
    SimilarityService,
    TriageService
)
from .ports import (
    AgentRepository,
    CacheService,
    CampaignRepository,
    EmbeddingService,
    EventPublisher,
    FindingRepository,
    KnowledgeAtomRepository,
    NotificationService,
    SecurityScanner,
    TargetRepository
)

__all__ = [
    # Command DTOs
    "CreateCampaignCommand",
    "StartCampaignCommand", 
    "ExecuteAgentCommand",
    "TriageFindingCommand",
    "GenerateEmbeddingCommand",
    "CreateKnowledgeAtomCommand",
    
    # Query DTOs
    "SearchSimilarFindingsQuery",
    "GetCampaignStatusQuery",
    "ListActiveCampaignsQuery",
    
    # Use Cases
    "CreateCampaignUseCase",
    "StartCampaignUseCase",
    "ExecuteAgentUseCase",
    "TriageFindingUseCase",
    "SearchSimilarFindingsUseCase",
    "GenerateEmbeddingUseCase",
    "CreateKnowledgeAtomUseCase"
]


# Command DTOs
@dataclass(frozen=True)
class CreateCampaignCommand:
    """Command to create a new campaign"""
    name: str
    target_id: TargetId
    max_cost_usd: Decimal
    max_duration_hours: int
    max_api_calls: int
    required_capabilities: List[AgentCapability]
    max_agents: int = 5


@dataclass(frozen=True)
class StartCampaignCommand:
    """Command to start a campaign"""
    campaign_id: CampaignId


@dataclass(frozen=True)
class ExecuteAgentCommand:
    """Command to execute an agent"""
    agent_id: AgentId
    campaign_id: CampaignId
    scan_config: dict[str, any]


@dataclass(frozen=True)
class TriageFindingCommand:
    """Command to triage a finding"""
    finding_id: FindingId
    new_status: FindingStatus
    reasoning: Optional[str] = None


@dataclass(frozen=True)
class GenerateEmbeddingCommand:
    """Command to generate an embedding"""
    text: str
    model: str = "nvidia/embed-qa-4"
    input_type: str = "query"


@dataclass(frozen=True)
class CreateKnowledgeAtomCommand:
    """Command to create a knowledge atom"""
    content: str
    atom_type: str
    confidence: float
    tags: List[str]
    source: Optional[str] = None


# Query DTOs
@dataclass(frozen=True)  
class SearchSimilarFindingsQuery:
    """Query to search for similar findings"""
    finding_id: FindingId
    similarity_threshold: float = 0.8
    max_results: int = 10


@dataclass(frozen=True)
class GetCampaignStatusQuery:
    """Query to get campaign status"""
    campaign_id: CampaignId


@dataclass(frozen=True)
class ListActiveCampaignsQuery:
    """Query to list active campaigns"""
    limit: int = 50


# Use Cases
class CreateCampaignUseCase:
    """Use case for creating a new campaign"""
    
    def __init__(
        self,
        target_repository: TargetRepository,
        agent_repository: AgentRepository,
        campaign_repository: CampaignRepository,
        event_publisher: EventPublisher
    ) -> None:
        self._target_repository = target_repository
        self._agent_repository = agent_repository
        self._campaign_repository = campaign_repository
        self._event_publisher = event_publisher
    
    async def execute(self, command: CreateCampaignCommand) -> CampaignId:
        """Execute the create campaign use case"""
        
        # Fetch target
        target = await self._target_repository.find_by_id(command.target_id)
        if not target:
            raise ValueError(f"Target {command.target_id} not found")
        
        # Create budget limit
        budget = BudgetLimit(
            max_cost_usd=command.max_cost_usd,
            max_duration_hours=command.max_duration_hours,
            max_api_calls=command.max_api_calls
        )
        
        # Select agents
        available_agents = await self._agent_repository.find_active_agents()
        selected_agents = AgentSelectionService.select_agents_for_target(
            target=target,
            available_agents=available_agents,
            required_capabilities=set(command.required_capabilities),
            budget_limit=budget,
            max_agents=command.max_agents
        )
        
        if not selected_agents:
            raise ValueError("No suitable agents found for campaign")
        
        # Create campaign
        campaign_id = CampaignId.generate()
        campaign = Campaign(
            id_=campaign_id,
            name=command.name,
            target=target,
            budget=budget,
            created_at=datetime.now(timezone.utc),
            status=CampaignStatus.QUEUED,
            scheduled_agents=[agent.id for agent in selected_agents]
        )
        
        # Save campaign
        await self._campaign_repository.save(campaign)
        
        # Publish event (async, fire-and-forget)
        await self._event_publisher.publish(
            CampaignStarted(
                campaign_id=campaign_id,
                target_id=command.target_id,
                scheduled_agent_count=len(selected_agents),
                estimated_duration_hours=CampaignOrchestratorService.estimate_campaign_duration(
                    selected_agents, parallel_execution=True
                ) // 60,  # Convert minutes to hours
                budget_limit_usd=float(command.max_cost_usd)
            )
        )
        
        return campaign_id


class StartCampaignUseCase:
    """Use case for starting a campaign"""
    
    def __init__(
        self,
        campaign_repository: CampaignRepository,
        event_publisher: EventPublisher,
        notification_service: NotificationService
    ) -> None:
        self._campaign_repository = campaign_repository
        self._event_publisher = event_publisher
        self._notification_service = notification_service
    
    async def execute(self, command: StartCampaignCommand) -> None:
        """Execute the start campaign use case"""
        
        # Fetch campaign
        campaign = await self._campaign_repository.find_by_id(command.campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {command.campaign_id} not found")
        
        # Check if campaign can be started
        can_start, issues = CampaignOrchestratorService.can_start_campaign(campaign)
        if not can_start:
            raise ValueError(f"Cannot start campaign: {'; '.join(issues)}")
        
        # Start campaign
        campaign.start_campaign()
        await self._campaign_repository.save(campaign)
        
        # Send notification
        await self._notification_service.send_campaign_update(
            campaign_id=command.campaign_id,
            status="started",
            details=f"Campaign '{campaign.name}' has been started"
        )


class ExecuteAgentUseCase:
    """Use case for executing an agent"""
    
    def __init__(
        self,
        agent_repository: AgentRepository,
        campaign_repository: CampaignRepository,
        finding_repository: FindingRepository,
        security_scanner: SecurityScanner,
        event_publisher: EventPublisher
    ) -> None:
        self._agent_repository = agent_repository
        self._campaign_repository = campaign_repository
        self._finding_repository = finding_repository
        self._security_scanner = security_scanner
        self._event_publisher = event_publisher
    
    async def execute(self, command: ExecuteAgentCommand) -> List[FindingId]:
        """Execute an agent and return discovered findings"""
        
        # Fetch agent and campaign
        agent = await self._agent_repository.find_by_id(command.agent_id)
        if not agent:
            raise ValueError(f"Agent {command.agent_id} not found")
        
        campaign = await self._campaign_repository.find_by_id(command.campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {command.campaign_id} not found")
        
        # Publish start event
        await self._event_publisher.publish(
            AgentExecutionStarted(
                agent_id=command.agent_id,
                campaign_id=command.campaign_id,
                target_id=campaign.target.id,
                estimated_duration_minutes=agent.average_duration_minutes,
                estimated_cost_usd=float(agent.cost_per_execution)
            )
        )
        
        try:
            # Execute security scan
            scan_results = await self._security_scanner.scan_target(
                target=campaign.target,
                scan_config=command.scan_config
            )
            
            # Process scan results into findings
            findings = []
            for result in scan_results.get("findings", []):
                finding_id = FindingId.generate()
                finding = Finding(
                    id_=finding_id,
                    campaign_id=command.campaign_id,
                    agent_id=command.agent_id,
                    title=result.get("title", "Unknown Finding"),
                    description=result.get("description", ""),
                    severity=Severity(result.get("severity", "info")),
                    created_at=datetime.now(timezone.utc),
                    evidence=result.get("evidence", {})
                )
                
                await self._finding_repository.save(finding)
                campaign.add_finding(finding_id)
                findings.append(finding_id)
                
                # Publish finding discovered event
                await self._event_publisher.publish(
                    FindingDiscovered(
                        finding_id=finding_id,
                        campaign_id=command.campaign_id,
                        agent_id=command.agent_id,
                        title=finding.title,
                        severity=finding.severity,
                        target_info=campaign.target.name
                    )
                )
            
            # Update campaign
            await self._campaign_repository.save(campaign)
            
            # Publish completion event
            await self._event_publisher.publish(
                AgentExecutionCompleted(
                    agent_id=command.agent_id,
                    campaign_id=command.campaign_id,
                    target_id=campaign.target.id,
                    actual_duration_minutes=agent.average_duration_minutes,  # Would be actual in real implementation
                    actual_cost_usd=float(agent.cost_per_execution),
                    findings_discovered=len(findings),
                    success=True
                )
            )
            
            return findings
            
        except Exception as e:
            # Publish failure event
            await self._event_publisher.publish(
                AgentExecutionCompleted(
                    agent_id=command.agent_id,
                    campaign_id=command.campaign_id,
                    target_id=campaign.target.id,
                    actual_duration_minutes=0,
                    actual_cost_usd=0.0,
                    findings_discovered=0,
                    success=False,
                    error_message=str(e)
                )
            )
            raise


class TriageFindingUseCase:
    """Use case for triaging findings"""
    
    def __init__(
        self,
        finding_repository: FindingRepository,
        event_publisher: EventPublisher
    ) -> None:
        self._finding_repository = finding_repository
        self._event_publisher = event_publisher
    
    async def execute(self, command: TriageFindingCommand) -> None:
        """Execute the triage finding use case"""
        
        # Fetch finding
        finding = await self._finding_repository.find_by_id(command.finding_id)
        if not finding:
            raise ValueError(f"Finding {command.finding_id} not found")
        
        # Store previous status
        previous_status = finding.status
        
        # Update finding status
        finding.update_status(command.new_status)
        await self._finding_repository.save(finding)
        
        # Calculate confidence score
        confidence_score = TriageService.calculate_priority_score(finding)
        
        # Publish triage event
        await self._event_publisher.publish(
            FindingTriaged(
                finding_id=command.finding_id,
                campaign_id=finding.campaign_id,
                previous_status=previous_status.value,
                new_status=command.new_status.value,
                confidence_score=confidence_score,
                reasoning=command.reasoning
            )
        )


class SearchSimilarFindingsUseCase:
    """Use case for searching similar findings"""
    
    def __init__(
        self,
        finding_repository: FindingRepository,
        cache_service: CacheService
    ) -> None:
        self._finding_repository = finding_repository
        self._cache_service = cache_service
    
    async def execute(
        self, 
        query: SearchSimilarFindingsQuery
    ) -> List[Tuple[Finding, float]]:
        """Execute the search similar findings use case"""
        
        # Check cache first
        cache_key = f"similar_findings:{query.finding_id}:{query.similarity_threshold}"
        cached_result = await self._cache_service.get(cache_key)
        if cached_result:
            return cached_result
        
        # Fetch target finding
        target_finding = await self._finding_repository.find_by_id(query.finding_id)
        if not target_finding:
            raise ValueError(f"Finding {query.finding_id} not found")
        
        if not target_finding.embedding:
            return []  # Cannot find similar findings without embedding
        
        # Find similar findings
        similar_findings = await self._finding_repository.find_similar(
            embedding=target_finding.embedding,
            threshold=query.similarity_threshold,
            limit=query.max_results
        )
        
        # Cache result for 1 hour
        await self._cache_service.set(cache_key, similar_findings, ttl_seconds=3600)
        
        return similar_findings


class GenerateEmbeddingUseCase:
    """Use case for generating embeddings"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        event_publisher: EventPublisher,
        cache_service: CacheService
    ) -> None:
        self._embedding_service = embedding_service
        self._event_publisher = event_publisher
        self._cache_service = cache_service
    
    async def execute(self, command: GenerateEmbeddingCommand) -> Embedding:
        """Execute the generate embedding use case"""
        
        # Check cache first
        cache_key = f"embedding:{command.model}:{command.input_type}:{hash(command.text)}"
        cached_embedding = await self._cache_service.get(cache_key)
        if cached_embedding:
            return cached_embedding
        
        # Generate embedding
        start_time = datetime.now(timezone.utc)
        embedding = await self._embedding_service.generate_embedding(
            text=command.text,
            model=command.model,
            input_type=command.input_type
        )
        end_time = datetime.now(timezone.utc)
        
        # Cache embedding
        await self._cache_service.set(cache_key, embedding, ttl_seconds=86400)  # 24 hours
        
        # Publish event
        await self._event_publisher.publish(
            EmbeddingGenerated(
                entity_id=UUID("00000000-0000-0000-0000-000000000000"),  # Generic ID for text embedding
                entity_type="text",
                model_name=command.model,
                vector_dimension=embedding.dimension,
                generation_time_ms=int((end_time - start_time).total_seconds() * 1000),
                cached=False
            )
        )
        
        return embedding


class CreateKnowledgeAtomUseCase:
    """Use case for creating knowledge atoms"""
    
    def __init__(
        self,
        knowledge_repository: KnowledgeAtomRepository,
        embedding_service: EmbeddingService,
        event_publisher: EventPublisher
    ) -> None:
        self._knowledge_repository = knowledge_repository
        self._embedding_service = embedding_service
        self._event_publisher = event_publisher
    
    async def execute(self, command: CreateKnowledgeAtomCommand) -> AtomId:
        """Execute the create knowledge atom use case"""
        
        # Create atom
        atom_id = AtomId.generate()
        atom = KnowledgeAtom(
            id_=atom_id,
            content=command.content,
            atom_type=AtomType(command.atom_type),
            confidence=command.confidence,
            created_at=datetime.now(timezone.utc),
            tags=set(command.tags),
            source=command.source
        )
        
        # Generate embedding for content
        embedding = await self._embedding_service.generate_embedding(
            text=command.content,
            input_type="passage"  # Knowledge atoms are passages
        )
        atom.set_embedding(embedding)
        
        # Save atom
        await self._knowledge_repository.save(atom)
        
        # Publish event
        await self._event_publisher.publish(
            KnowledgeAtomCreated(
                atom_id=atom_id,
                atom_type=command.atom_type,
                content_length=len(command.content),
                confidence=command.confidence,
                source=command.source,
                tags=tuple(command.tags)
            )
        )
        
        return atom_id