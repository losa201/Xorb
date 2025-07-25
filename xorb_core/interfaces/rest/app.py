"""
FastAPI Application Factory

Creates and configures the FastAPI application with all routes and middleware.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from ..dependencies import Dependencies, get_dependencies
from .controllers import CampaignController, FindingController, HealthController, KnowledgeController
from .schemas import ErrorResponse

__all__ = ["create_app"]

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan events"""
    
    # Startup
    logger.info("Starting Xorb API service")
    
    try:
        # Initialize dependencies
        deps = await get_dependencies()
        app.state.dependencies = deps
        logger.info("Dependencies initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize dependencies", error=str(e))
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Xorb API service")
        
        if hasattr(app.state, 'dependencies'):
            await app.state.dependencies.cleanup()
            logger.info("Dependencies cleaned up")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Xorb Security Intelligence Platform",
        description="AI-powered security testing and vulnerability discovery platform",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Configure middleware
    configure_middleware(app)
    
    # Configure exception handlers
    configure_exception_handlers(app)
    
    # Configure routes
    configure_routes(app)
    
    return app


def configure_middleware(app: FastAPI) -> None:
    """Configure FastAPI middleware"""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log HTTP requests"""
        
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        logger.info(
            "HTTP request processed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time
        )
        
        return response


def configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                message=exc.detail,
                details={"status_code": exc.status_code}
            ).dict()
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle validation errors"""
        
        logger.warning("Validation error", error=str(exc), path=request.url.path)
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="ValidationError",
                message=str(exc)
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected errors"""
        
        logger.error("Unexpected error", error=str(exc), path=request.url.path)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred"
            ).dict()
        )


def configure_routes(app: FastAPI) -> None:
    """Configure API routes"""
    
    # Health check routes
    app.add_api_route(
        "/health",
        HealthController.health_check,
        methods=["GET"],
        tags=["Health"],
        summary="Health check",
        response_model=HealthResponse
    )
    
    app.add_api_route(
        "/health/ready",
        HealthController.readiness_check,
        methods=["GET"],
        tags=["Health"],
        summary="Readiness probe"
    )
    
    app.add_api_route(
        "/health/live",
        HealthController.liveness_check,
        methods=["GET"],
        tags=["Health"],
        summary="Liveness probe"
    )
    
    # Campaign routes
    app.add_api_route(
        "/api/v1/campaigns",
        CampaignController.create_campaign,
        methods=["POST"],
        tags=["Campaigns"],
        summary="Create a new campaign",
        response_model=CampaignResponse,
        status_code=status.HTTP_201_CREATED
    )
    
    app.add_api_route(
        "/api/v1/campaigns",
        CampaignController.list_campaigns,
        methods=["GET"],
        tags=["Campaigns"],
        summary="List campaigns",
        response_model=List[CampaignResponse]
    )
    
    app.add_api_route(
        "/api/v1/campaigns/{campaign_id}",
        CampaignController.get_campaign,
        methods=["GET"],
        tags=["Campaigns"],
        summary="Get campaign details",
        response_model=CampaignResponse
    )
    
    app.add_api_route(
        "/api/v1/campaigns/{campaign_id}/start",
        CampaignController.start_campaign,
        methods=["POST"],
        tags=["Campaigns"],
        summary="Start a campaign"
    )
    
    app.add_api_route(
        "/api/v1/campaigns/{campaign_id}/pause",
        CampaignController.pause_campaign,
        methods=["POST"],
        tags=["Campaigns"],
        summary="Pause a campaign"
    )
    
    # Finding routes
    app.add_api_route(
        "/api/v1/findings",
        FindingController.list_findings,
        methods=["GET"],
        tags=["Findings"],
        summary="List security findings",
        response_model=List[FindingResponse]
    )
    
    app.add_api_route(
        "/api/v1/findings/{finding_id}",
        FindingController.get_finding,
        methods=["GET"],
        tags=["Findings"],
        summary="Get finding details",
        response_model=FindingResponse
    )
    
    app.add_api_route(
        "/api/v1/findings/{finding_id}/triage",
        FindingController.triage_finding,
        methods=["POST"],
        tags=["Findings"],
        summary="Triage a finding"
    )
    
    app.add_api_route(
        "/api/v1/findings/{finding_id}/similar",
        FindingController.search_similar_findings,
        methods=["GET"],
        tags=["Findings"],
        summary="Find similar findings",
        response_model=List[FindingResponse]
    )
    
    # Knowledge routes
    app.add_api_route(
        "/api/v1/knowledge/embeddings",
        KnowledgeController.generate_embedding,
        methods=["POST"],
        tags=["Knowledge"],
        summary="Generate text embedding"
    )
    
    app.add_api_route(
        "/api/v1/knowledge/atoms/{atom_id}",
        KnowledgeController.get_knowledge_atom,
        methods=["GET"],
        tags=["Knowledge"],
        summary="Get knowledge atom",
        response_model=KnowledgeAtomResponse
    )
    
    app.add_api_route(
        "/api/v1/knowledge/search",
        KnowledgeController.search_knowledge,
        methods=["GET"],
        tags=["Knowledge"],
        summary="Search knowledge base",
        response_model=List[KnowledgeAtomResponse]
    )


# Add missing imports
import time
from .schemas import HealthResponse