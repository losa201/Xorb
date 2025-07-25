"""
Dependency Injection Container

Wires up infrastructure implementations with application services.
This is the only place where the interface layer connects to infrastructure.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

import asyncpg
from neo4j import AsyncGraphDatabase

from ..application import (
    CampaignApplicationService,
    CreateCampaignUseCase,
    CreateKnowledgeAtomUseCase,
    ExecuteAgentUseCase,
    FindingApplicationService,
    GenerateEmbeddingUseCase,
    KnowledgeApplicationService,
    SearchSimilarFindingsUseCase,
    StartCampaignUseCase,
    TriageFindingUseCase
)
from ..infrastructure import (
    NatsEventPublisher,
    Neo4jKnowledgeAtomRepository,
    NvidiaEmbeddingService,
    PostgreSQLAgentRepository,
    PostgreSQLCampaignRepository,
    PostgreSQLFindingRepository,
    PostgreSQLTargetRepository,
    RedisCache,
    SlackNotificationService
)

__all__ = [
    "Dependencies",
    "get_campaign_service", 
    "get_finding_service",
    "get_knowledge_service"
]


class Dependencies:
    """Dependency injection container"""
    
    def __init__(self) -> None:
        self._instances: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all dependencies"""
        
        if self._initialized:
            return
        
        # Database connections
        await self._setup_databases()
        
        # External services
        await self._setup_external_services()
        
        # Repositories
        await self._setup_repositories()
        
        # Use cases
        await self._setup_use_cases()
        
        # Application services
        await self._setup_application_services()
        
        self._initialized = True
    
    async def _setup_databases(self) -> None:
        """Setup database connections"""
        
        # PostgreSQL connection pool
        postgresql_url = os.getenv(
            "POSTGRESQL_URL", 
            "postgresql://xorb:xorb@localhost:5432/xorb"
        )
        
        self._instances["postgres_pool"] = await asyncpg.create_pool(
            postgresql_url,
            min_size=5,
            max_size=20,
            command_timeout=30
        )
        
        # Neo4j driver
        neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")
        
        self._instances["neo4j_driver"] = AsyncGraphDatabase.driver(
            neo4j_url,
            auth=(neo4j_user, neo4j_password)
        )
    
    async def _setup_external_services(self) -> None:
        """Setup external service clients"""
        
        # NVIDIA Embedding Service
        nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
        if not nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is required")
        
        self._instances["embedding_service"] = NvidiaEmbeddingService(
            api_key=nvidia_api_key
        )
        
        # Redis Cache
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self._instances["cache_service"] = RedisCache(redis_url)
        
        # NATS Event Publisher
        nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
        self._instances["event_publisher"] = NatsEventPublisher(nats_url)
        
        # Slack Notification Service
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        if slack_webhook:
            self._instances["notification_service"] = SlackNotificationService(
                webhook_url=slack_webhook
            )
        else:
            # Use a no-op notification service for development
            from ..infrastructure.external import NotificationService
            
            class NoOpNotificationService(NotificationService):
                async def send_alert(self, title: str, message: str, severity: str = "info", metadata = None) -> None:
                    pass
                async def send_campaign_update(self, campaign_id, status: str, details: str) -> None:
                    pass
            
            self._instances["notification_service"] = NoOpNotificationService()
    
    async def _setup_repositories(self) -> None:
        """Setup repository implementations"""
        
        postgres_pool = self._instances["postgres_pool"]
        neo4j_driver = self._instances["neo4j_driver"]
        
        self._instances["target_repository"] = PostgreSQLTargetRepository(postgres_pool)
        self._instances["agent_repository"] = PostgreSQLAgentRepository(postgres_pool)
        self._instances["finding_repository"] = PostgreSQLFindingRepository(postgres_pool)
        self._instances["knowledge_atom_repository"] = Neo4jKnowledgeAtomRepository(neo4j_driver)
        
        # Campaign repository needs target repository
        self._instances["campaign_repository"] = PostgreSQLCampaignRepository(
            postgres_pool,
            self._instances["target_repository"]
        )
    
    async def _setup_use_cases(self) -> None:
        """Setup use case implementations"""
        
        # Get dependencies
        target_repo = self._instances["target_repository"]
        agent_repo = self._instances["agent_repository"]
        campaign_repo = self._instances["campaign_repository"]
        finding_repo = self._instances["finding_repository"]
        knowledge_repo = self._instances["knowledge_atom_repository"]
        embedding_service = self._instances["embedding_service"]
        event_publisher = self._instances["event_publisher"]
        
        # Create use cases
        self._instances["create_campaign_use_case"] = CreateCampaignUseCase(
            target_repository=target_repo,
            agent_repository=agent_repo,
            campaign_repository=campaign_repo,
            event_publisher=event_publisher
        )
        
        self._instances["start_campaign_use_case"] = StartCampaignUseCase(
            campaign_repository=campaign_repo,
            event_publisher=event_publisher
        )
        
        self._instances["execute_agent_use_case"] = ExecuteAgentUseCase(
            agent_repository=agent_repo,
            campaign_repository=campaign_repo,
            finding_repository=finding_repo,
            event_publisher=event_publisher
        )
        
        self._instances["triage_finding_use_case"] = TriageFindingUseCase(
            finding_repository=finding_repo,
            event_publisher=event_publisher
        )
        
        self._instances["search_similar_findings_use_case"] = SearchSimilarFindingsUseCase(
            finding_repository=finding_repo,
            embedding_service=embedding_service
        )
        
        self._instances["generate_embedding_use_case"] = GenerateEmbeddingUseCase(
            embedding_service=embedding_service
        )
        
        self._instances["create_knowledge_atom_use_case"] = CreateKnowledgeAtomUseCase(
            knowledge_atom_repository=knowledge_repo,
            embedding_service=embedding_service,
            event_publisher=event_publisher
        )
    
    async def _setup_application_services(self) -> None:
        """Setup application service implementations"""
        
        # Get dependencies
        create_campaign_uc = self._instances["create_campaign_use_case"]
        start_campaign_uc = self._instances["start_campaign_use_case"]
        execute_agent_uc = self._instances["execute_agent_use_case"]
        triage_finding_uc = self._instances["triage_finding_use_case"]
        search_similar_uc = self._instances["search_similar_findings_use_case"]
        generate_embedding_uc = self._instances["generate_embedding_use_case"]
        create_knowledge_uc = self._instances["create_knowledge_atom_use_case"]
        
        notification_service = self._instances["notification_service"]
        cache_service = self._instances["cache_service"]
        
        # Create application services
        self._instances["campaign_service"] = CampaignApplicationService(
            create_campaign_use_case=create_campaign_uc,
            start_campaign_use_case=start_campaign_uc,
            execute_agent_use_case=execute_agent_uc,
            notification_service=notification_service,
            cache_service=cache_service
        )
        
        self._instances["finding_service"] = FindingApplicationService(
            triage_finding_use_case=triage_finding_uc,
            search_similar_findings_use_case=search_similar_uc,
            generate_embedding_use_case=generate_embedding_uc,
            notification_service=notification_service,
            cache_service=cache_service
        )
        
        self._instances["knowledge_service"] = KnowledgeApplicationService(
            create_knowledge_atom_use_case=create_knowledge_uc,
            search_similar_findings_use_case=search_similar_uc,
            generate_embedding_use_case=generate_embedding_uc,
            cache_service=cache_service
        )
    
    def get(self, service_name: str) -> Any:
        """Get a service instance"""
        
        if not self._initialized:
            raise RuntimeError("Dependencies not initialized")
        
        if service_name not in self._instances:
            raise KeyError(f"Service not found: {service_name}")
        
        return self._instances[service_name]
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        
        if "postgres_pool" in self._instances:
            await self._instances["postgres_pool"].close()
        
        if "neo4j_driver" in self._instances:
            await self._instances["neo4j_driver"].close()


# Global dependencies instance
_dependencies: Optional[Dependencies] = None


async def get_dependencies() -> Dependencies:
    """Get or create the global dependencies instance"""
    
    global _dependencies
    
    if _dependencies is None:
        _dependencies = Dependencies()
        await _dependencies.initialize()
    
    return _dependencies


async def get_campaign_service() -> CampaignApplicationService:
    """Dependency injection for campaign service"""
    
    deps = await get_dependencies()
    return deps.get("campaign_service")


async def get_finding_service() -> FindingApplicationService:
    """Dependency injection for finding service"""
    
    deps = await get_dependencies()
    return deps.get("finding_service")


async def get_knowledge_service() -> KnowledgeApplicationService:
    """Dependency injection for knowledge service"""
    
    deps = await get_dependencies()
    return deps.get("knowledge_service")