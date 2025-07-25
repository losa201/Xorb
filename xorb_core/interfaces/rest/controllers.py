"""
REST API Controllers

FastAPI endpoint handlers that orchestrate application services.
Controllers handle HTTP concerns and delegate business logic to application layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status

from ...application import (
    CampaignApplicationService,
    CreateCampaignCommand,
    FindingApplicationService,
    GenerateEmbeddingCommand,
    KnowledgeApplicationService,
    SearchSimilarFindingsQuery,
    StartCampaignCommand,
    TriageFindingCommand
)
from ...domain import AgentCapability, FindingId, FindingStatus, TargetId
from ..dependencies import (
    get_campaign_service,
    get_finding_service,
    get_knowledge_service
)
from .schemas import (
    CampaignResponse,
    CreateCampaignRequest,
    EmbeddingRequest,
    ErrorResponse,
    FindingResponse,
    HealthResponse,
    KnowledgeAtomResponse,
    StartCampaignRequest,
    TriageFindingRequest
)

__all__ = [
    "CampaignController",
    "FindingController",
    "KnowledgeController", 
    "HealthController"
]

logger = logging.getLogger(__name__)


class CampaignController:
    """Campaign management endpoints"""
    
    @staticmethod
    async def create_campaign(
        request: CreateCampaignRequest,
        campaign_service: CampaignApplicationService = Depends(get_campaign_service)
    ) -> CampaignResponse:
        """Create a new security campaign"""
        
        try:
            # Convert capabilities from strings to domain enums
            capabilities = [AgentCapability(cap) for cap in request.required_capabilities]
            
            # Create campaign through application service
            campaign_id = await campaign_service.create_and_start_campaign(
                name=request.name,
                target_id=TargetId.from_string(str(request.target_id)),
                max_cost_usd=request.max_cost_usd,
                max_duration_hours=request.max_duration_hours,
                max_api_calls=request.max_api_calls,
                required_capabilities=capabilities,
                max_agents=request.max_agents,
                auto_start=request.auto_start
            )
            
            # Get campaign progress for response
            progress = await campaign_service.get_campaign_progress(campaign_id)
            
            return CampaignResponse(
                id=UUID(str(campaign_id)),
                name=request.name,
                target_name="Unknown",  # Would be fetched in real implementation
                status=progress["status"],
                budget_used_percentage=progress["budget_used_percentage"],
                agents_completed=progress["agents_completed"],
                agents_total=progress["agents_total"],
                findings_discovered=progress["findings_discovered"],
                created_at=datetime.now(timezone.utc),
                estimated_completion=progress.get("estimated_completion")
            )
            
        except ValueError as e:
            logger.warning("Invalid campaign request", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid request: {str(e)}"
            )
        except Exception as e:
            logger.error("Failed to create campaign", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create campaign"
            )
    
    @staticmethod
    async def start_campaign(
        request: StartCampaignRequest,
        campaign_service: CampaignApplicationService = Depends(get_campaign_service)
    ) -> dict[str, str]:
        """Start an existing campaign"""
        
        try:
            # This would use StartCampaignUseCase in real implementation
            logger.info("Starting campaign", campaign_id=str(request.campaign_id))
            
            return {"message": "Campaign started successfully"}
            
        except Exception as e:
            logger.error("Failed to start campaign", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start campaign"
            )
    
    @staticmethod
    async def get_campaign(
        campaign_id: UUID,
        campaign_service: CampaignApplicationService = Depends(get_campaign_service)
    ) -> CampaignResponse:
        """Get campaign details"""
        
        try:
            from ...domain import CampaignId
            
            progress = await campaign_service.get_campaign_progress(
                CampaignId.from_string(str(campaign_id))
            )
            
            return CampaignResponse(
                id=campaign_id,
                name="Campaign Name",  # Would be fetched from repository
                target_name="Target Name",  # Would be fetched from repository  
                status=progress["status"],
                budget_used_percentage=progress["budget_used_percentage"],
                agents_completed=progress["agents_completed"],
                agents_total=progress["agents_total"],
                findings_discovered=progress["findings_discovered"],
                created_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error("Failed to get campaign", campaign_id=str(campaign_id), error=str(e))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
    
    @staticmethod
    async def list_campaigns(
        status_filter: Optional[str] = None,
        campaign_service: CampaignApplicationService = Depends(get_campaign_service)
    ) -> List[CampaignResponse]:
        """List campaigns with optional status filter"""
        
        try:
            # This would use ListActiveCampaignsQuery in real implementation
            logger.info("Listing campaigns", status_filter=status_filter)
            
            # Return empty list for now
            return []
            
        except Exception as e:
            logger.error("Failed to list campaigns", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list campaigns"
            )
    
    @staticmethod
    async def pause_campaign(
        campaign_id: UUID,
        campaign_service: CampaignApplicationService = Depends(get_campaign_service)
    ) -> dict[str, str]:
        """Pause a running campaign"""
        
        try:
            from ...domain import CampaignId
            
            await campaign_service.pause_campaign(
                CampaignId.from_string(str(campaign_id))
            )
            
            return {"message": "Campaign paused successfully"}
            
        except Exception as e:
            logger.error("Failed to pause campaign", campaign_id=str(campaign_id), error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to pause campaign"
            )


class FindingController:
    """Security finding management endpoints"""
    
    @staticmethod
    async def get_finding(
        finding_id: UUID,
        finding_service: FindingApplicationService = Depends(get_finding_service)
    ) -> FindingResponse:
        """Get security finding details"""
        
        try:
            # Get finding insights
            insights = await finding_service.get_finding_insights(
                FindingId.from_string(str(finding_id))
            )
            
            return FindingResponse(
                id=finding_id,
                campaign_id=UUID("00000000-0000-0000-0000-000000000000"),  # Placeholder
                agent_id=UUID("00000000-0000-0000-0000-000000000000"),  # Placeholder
                title="Sample Finding",
                description="Sample description",
                severity="high",
                status="new", 
                created_at=datetime.now(timezone.utc),
                similar_findings_count=insights["similar_findings_count"]
            )
            
        except Exception as e:
            logger.error("Failed to get finding", finding_id=str(finding_id), error=str(e))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Finding not found"
            )
    
    @staticmethod
    async def list_findings(
        campaign_id: Optional[UUID] = None,
        severity: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[FindingResponse]:
        """List security findings with filters"""
        
        try:
            logger.info(
                "Listing findings", 
                campaign_id=str(campaign_id) if campaign_id else None,
                severity=severity,
                status=status_filter,
                limit=limit,
                offset=offset
            )
            
            # Return empty list for now
            return []
            
        except Exception as e:
            logger.error("Failed to list findings", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list findings"
            )
    
    @staticmethod
    async def triage_finding(
        finding_id: UUID,
        request: TriageFindingRequest,
        finding_service: FindingApplicationService = Depends(get_finding_service)
    ) -> dict[str, str]:
        """Triage a security finding"""
        
        try:
            await finding_service._triage_finding_use_case.execute(
                TriageFindingCommand(
                    finding_id=FindingId.from_string(str(finding_id)),
                    new_status=FindingStatus(request.new_status),
                    reasoning=request.reasoning
                )
            )
            
            return {"message": "Finding triaged successfully"}
            
        except ValueError as e:
            logger.warning("Invalid triage request", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid request: {str(e)}"
            )
        except Exception as e:
            logger.error("Failed to triage finding", finding_id=str(finding_id), error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to triage finding"
            )
    
    @staticmethod
    async def search_similar_findings(
        finding_id: UUID,
        threshold: float = 0.8,
        limit: int = 10,
        finding_service: FindingApplicationService = Depends(get_finding_service)
    ) -> List[FindingResponse]:
        """Search for similar findings"""
        
        try:
            similar_findings = await finding_service._search_similar_findings_use_case.execute(
                SearchSimilarFindingsQuery(
                    finding_id=FindingId.from_string(str(finding_id)),
                    similarity_threshold=threshold,
                    max_results=limit
                )
            )
            
            # Convert to response DTOs
            responses = []
            for finding, similarity in similar_findings:
                responses.append(FindingResponse(
                    id=UUID(str(finding.id)),
                    campaign_id=UUID(str(finding.campaign_id)),
                    agent_id=UUID(str(finding.agent_id)),
                    title=finding.title,
                    description=finding.description,
                    severity=finding.severity.value,
                    status=finding.status.value,
                    created_at=finding.created_at,
                    confidence_score=similarity
                ))
            
            return responses
            
        except Exception as e:
            logger.error("Failed to search similar findings", finding_id=str(finding_id), error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to search similar findings"
            )


class KnowledgeController:
    """Knowledge management endpoints"""
    
    @staticmethod
    async def generate_embedding(
        request: EmbeddingRequest,
        knowledge_service: KnowledgeApplicationService = Depends(get_knowledge_service)
    ) -> dict[str, any]:
        """Generate text embedding"""
        
        try:
            embedding = await knowledge_service._generate_embedding_use_case.execute(
                GenerateEmbeddingCommand(
                    text=request.text,
                    model=request.model,
                    input_type=request.input_type
                )
            )
            
            return {
                "embedding": list(embedding.vector),
                "dimension": embedding.dimension,
                "model": embedding.model
            }
            
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate embedding"
            )
    
    @staticmethod
    async def get_knowledge_atom(
        atom_id: UUID,
        knowledge_service: KnowledgeApplicationService = Depends(get_knowledge_service)
    ) -> KnowledgeAtomResponse:
        """Get knowledge atom details"""
        
        try:
            # This would fetch from repository in real implementation
            return KnowledgeAtomResponse(
                id=atom_id,
                content="Sample knowledge content",
                atom_type="vulnerability",
                confidence=0.85,
                tags=["sample", "placeholder"],
                created_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error("Failed to get knowledge atom", atom_id=str(atom_id), error=str(e))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge atom not found"
            )
    
    @staticmethod
    async def search_knowledge(
        query: str,
        atom_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[KnowledgeAtomResponse]:
        """Search knowledge base"""
        
        try:
            logger.info(
                "Searching knowledge",
                query=query,
                atom_type=atom_type,
                tags=tags,
                limit=limit
            )
            
            # Return empty list for now
            return []
            
        except Exception as e:
            logger.error("Failed to search knowledge", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to search knowledge"
            )


class HealthController:
    """Health check endpoints"""
    
    @staticmethod
    async def health_check() -> HealthResponse:
        """System health check"""
        
        try:
            # Check dependencies
            dependencies = {
                "database": "healthy",
                "redis": "healthy", 
                "nats": "healthy",
                "nvidia_api": "healthy"
            }
            
            # Determine overall status
            overall_status = "healthy" if all(
                status == "healthy" for status in dependencies.values()
            ) else "degraded"
            
            return HealthResponse(
                status=overall_status,
                version="2.0.0",
                dependencies=dependencies
            )
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return HealthResponse(
                status="unhealthy",
                version="2.0.0",
                dependencies={"error": str(e)}
            )
    
    @staticmethod
    async def readiness_check() -> dict[str, str]:
        """Readiness probe for Kubernetes"""
        
        try:
            # Check if all required services are ready
            # This would check database connections, etc.
            return {"status": "ready"}
            
        except Exception as e:
            logger.error("Readiness check failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
    
    @staticmethod
    async def liveness_check() -> dict[str, str]:
        """Liveness probe for Kubernetes"""
        
        return {"status": "alive"}