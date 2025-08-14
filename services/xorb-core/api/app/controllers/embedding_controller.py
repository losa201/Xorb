"""
Embedding controller - Handles embedding-related HTTP requests
"""

from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from ..container import get_container
from ..services.interfaces import EmbeddingService
from ..domain.exceptions import DomainException
from ..domain.entities import User, Organization
from .base import BaseController


class EmbeddingRequest(BaseModel):
    input: List[str] = Field(..., description="List of texts to embed", max_items=100)
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use")
    input_type: str = Field(default="query", description="Type of input text")
    encoding_format: str = Field(default="float", description="Encoding format for embeddings")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="First text for comparison")
    text2: str = Field(..., description="Second text for comparison")
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use")


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to embed", max_items=1000)
    model: str = Field(default="nvidia/embed-qa-4", description="Embedding model to use")
    batch_size: int = Field(default=50, description="Batch size for processing", le=100)
    input_type: str = Field(default="query", description="Input type for embeddings")


class EmbeddingController(BaseController):
    """Embedding controller"""

    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Setup embedding routes"""

        @self.router.post("/embeddings", response_model=EmbeddingResponse)
        async def create_embeddings_endpoint(
            request: EmbeddingRequest,
            background_tasks: BackgroundTasks,
            current_user: User = Depends(get_current_user),
            current_org: Organization = Depends(get_current_organization)
        ):
            return await self.create_embeddings(request, background_tasks, current_user, current_org)

        @self.router.get("/embeddings/models")
        async def list_embedding_models_endpoint(
            current_user: User = Depends(get_current_user)
        ):
            return await self.list_models(current_user)

        @self.router.post("/embeddings/similarity")
        async def compute_similarity_endpoint(
            request: SimilarityRequest,
            current_user: User = Depends(get_current_user)
        ):
            return await self.compute_similarity(request, current_user)

        @self.router.post("/embeddings/batch")
        async def batch_embeddings_endpoint(
            request: BatchEmbeddingRequest,
            current_user: User = Depends(get_current_user),
            current_org: Organization = Depends(get_current_organization)
        ):
            return await self.batch_embeddings(request, current_user, current_org)

        @self.router.get("/embeddings/health")
        async def embedding_service_health_endpoint():
            return await self.health_check()

    async def create_embeddings(
        self,
        request: EmbeddingRequest,
        background_tasks: BackgroundTasks,
        current_user: User,
        current_org: Organization
    ) -> EmbeddingResponse:
        """Generate embeddings for input texts"""

        try:
            container = get_container()
            embedding_service = container.get(EmbeddingService)

            # Generate embeddings using service
            result = await embedding_service.generate_embeddings(
                texts=request.input,
                model=request.model,
                input_type=request.input_type,
                user=current_user,
                org=current_org
            )

            # Convert domain result to API response
            embedding_data = []
            for i, embedding in enumerate(result.embeddings):
                embedding_data.append(EmbeddingData(
                    embedding=embedding,
                    index=i
                ))

            response = EmbeddingResponse(
                data=embedding_data,
                model=result.model,
                usage=result.usage_stats
            )

            # Log usage in background
            def log_usage():
                # In a real implementation, this would send analytics events
                pass

            background_tasks.add_task(log_usage)

            return response

        except DomainException as e:
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def list_models(self, current_user: User) -> dict:
        """List available embedding models"""

        try:
            container = get_container()
            embedding_service = container.get(EmbeddingService)

            models = await embedding_service.get_available_models()

            return {"object": "list", "data": models}

        except DomainException as e:
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def compute_similarity(
        self,
        request: SimilarityRequest,
        current_user: User
    ) -> dict:
        """Compute semantic similarity between two texts"""

        try:
            container = get_container()
            embedding_service = container.get(EmbeddingService)

            similarity = await embedding_service.compute_similarity(
                text1=request.text1,
                text2=request.text2,
                model=request.model,
                user=current_user
            )

            return {
                "similarity": similarity,
                "text1": request.text1,
                "text2": request.text2,
                "model": request.model
            }

        except DomainException as e:
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def batch_embeddings(
        self,
        request: BatchEmbeddingRequest,
        current_user: User,
        current_org: Organization
    ) -> EmbeddingResponse:
        """Process large batches of texts for embedding generation"""

        try:
            container = get_container()
            embedding_service = container.get(EmbeddingService)

            result = await embedding_service.batch_embeddings(
                texts=request.texts,
                model=request.model,
                batch_size=request.batch_size,
                input_type=request.input_type,
                user=current_user,
                org=current_org
            )

            # Convert domain result to API response
            embedding_data = []
            for i, embedding in enumerate(result.embeddings):
                embedding_data.append(EmbeddingData(
                    embedding=embedding,
                    index=i
                ))

            return EmbeddingResponse(
                data=embedding_data,
                model=result.model,
                usage=result.usage_stats
            )

        except DomainException as e:
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def health_check(self) -> dict:
        """Check embedding service health"""

        try:
            container = get_container()
            embedding_service = container.get(EmbeddingService)

            health_info = await embedding_service.health_check()

            return health_info

        except DomainException as e:
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )


def get_current_user() -> User:
    """Dependency to get current user - simplified for this example"""
    from ..domain.entities import User
    # In a real implementation, this would extract and validate the user from the JWT token
    return User.create(username="admin", email="admin@xorb.com", roles=["admin"])


def get_current_organization() -> Organization:
    """Dependency to get current organization - simplified for this example"""
    from ..domain.entities import Organization
    # In a real implementation, this would be determined from the user context or request headers
    return Organization.create(name="Default Organization", plan_type="Enterprise")


# Create controller instance and export router
embedding_controller = EmbeddingController()
router = embedding_controller.router
