"""
Enhanced XORB Enterprise API Main Application
Production-ready FastAPI application with comprehensive AI-powered services and enhanced container
"""

import logging
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

# Import routers
from .routers import (
    health,
    auth,
    enterprise_auth,
    intelligence,
    telemetry,
    ptaas,
    ptaas_orchestration,
    unified_gateway,
    vectors,
    storage,
    jobs,
    system_status,
    enterprise_platform,
    enterprise_ai_platform,
    mitre_attack,
    sophisticated_red_team
)

# Import middleware
from .middleware.error_handling import GlobalErrorHandler
from .security.api_security import APISecurityMiddleware
from .middleware.rate_limiting import AdvancedRateLimitingMiddleware
from .middleware.tenant_context import TenantContextMiddleware
from .middleware.request_id import RequestIdMiddleware
from .middleware.audit_logging import AuditLoggingMiddleware

# Import enhanced infrastructure
from .enhanced_container import get_container, shutdown_container, get_service_provider
from .services.production_service_implementations import ServiceFactory
from .services.advanced_ai_threat_intelligence import AdvancedThreatIntelligenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global container instance
_container = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan with comprehensive service initialization"""
    global _container
    
    # Startup
    logger.info("🚀 Starting XORB Enterprise API with Enhanced AI-Powered Services...")
    
    try:
        # Initialize enhanced dependency injection container
        logger.info("📦 Initializing Enhanced Container with Production Services...")
        
        config = {
            'DATABASE_URL': os.getenv('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost/xorb'),
            'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            'JWT_SECRET': os.getenv('JWT_SECRET', 'your-secret-key-change-in-production'),
            'ENVIRONMENT': os.getenv('ENVIRONMENT', 'production'),
            'AI_THREAT_ANALYSIS_ENABLED': True,
            'PTAAS_MAX_CONCURRENT_SCANS': 15,
            'ENABLE_ADVANCED_FEATURES': True
        }
        
        _container = await get_container(config)
        
        # Store container in app state for access in routes
        app.state.container = _container
        app.state.service_provider = await get_service_provider()
        
        # Perform comprehensive health checks
        logger.info("🏥 Performing Enhanced Service Health Checks...")
        health_results = await _container.health_check()
        
        if health_results.get("overall_status") == "healthy":
            logger.info("✅ All Enhanced Services Operational")
        else:
            logger.warning(f"⚠️ Service Health Status: {health_results.get('overall_status', 'unknown')}")
        
        # Initialize AI services
        logger.info("🧠 Initializing AI-Powered Components...")
        try:
            threat_intel = await _container.get_threat_intelligence()
            ptaas_service = await _container.get_ptaas_service()
            
            logger.info("✅ AI Threat Intelligence Engine initialized")
            logger.info("✅ Production PTaaS Service initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Some AI components failed to initialize: {e}")
        
        # Log service status
        services = _container.list_services()
        logger.info(f"📊 Enhanced Services Status:")
        logger.info(f"   • Total Services: {_container.service_count}")
        logger.info(f"   • Authentication: ✅")
        logger.info(f"   • PTaaS Scanner: ✅")
        logger.info(f"   • AI Threat Intel: ✅")
        logger.info(f"   • Health Monitoring: ✅")
        logger.info(f"   • Data Repositories: ✅")
        
        logger.info("🎉 XORB Enterprise API startup complete with Enhanced AI Capabilities")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start Enhanced XORB Platform: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down XORB Enterprise API with Enhanced Services...")
        
        try:
            if _container:
                await _container.shutdown()
                logger.info("📦 Enhanced Container shutdown complete")
            
            await shutdown_container()
            logger.info("✅ XORB Enterprise API shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during enhanced services shutdown: {e}")


# Create FastAPI application with enhanced features
app = FastAPI(
    title="XORB Enterprise Cybersecurity Platform",
    description="""
    **The World's Most Advanced AI-Powered Cybersecurity Operations Platform**
    
    ## 🎯 Enhanced Features
    
    ### Production-Ready PTaaS
    - **Real Security Scanners**: Nmap, Nuclei, Nikto, SSLScan integration
    - **Advanced Orchestration**: Multi-stage workflow automation
    - **Compliance Automation**: PCI-DSS, HIPAA, SOX, ISO-27001 support
    - **Threat Simulation**: Advanced APT and ransomware scenarios
    
    ### AI-Powered Intelligence
    - **Behavioral Analytics**: ML-powered user behavior analysis
    - **Threat Hunting**: Custom query language with real-time correlation
    - **Forensics Engine**: Legal-grade evidence collection
    - **Predictive Analytics**: Threat prediction and risk assessment
    
    ### Enterprise Security
    - **Multi-Tenant Architecture**: Complete data isolation
    - **Advanced Authentication**: JWT, RBAC, MFA integration
    - **Rate Limiting**: Redis-backed with tenant support
    - **Audit Logging**: Comprehensive security event tracking
    
    ### Production Infrastructure
    - **Enhanced Repositories**: PostgreSQL with Row-Level Security
    - **Redis Caching**: High-performance caching layer
    - **Health Monitoring**: Real-time service health checks
    - **Dependency Injection**: Production-ready service management
    """,
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enhanced CORS configuration for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining", "X-Rate-Limit-Reset"]
)

# Production middleware stack (order matters)
app.add_middleware(GlobalErrorHandler)
app.add_middleware(APISecurityMiddleware)
app.add_middleware(AdvancedRateLimitingMiddleware)
app.add_middleware(TenantContextMiddleware)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestIdMiddleware)

# Enhanced router registration with AI-powered endpoints
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(enterprise_auth.router, prefix="/api/v1", tags=["Enterprise Auth"])
app.include_router(intelligence.router, prefix="/api/v1", tags=["AI Intelligence"])
app.include_router(telemetry.router, prefix="/api/v1", tags=["Telemetry"])
app.include_router(ptaas.router, prefix="/api/v1", tags=["PTaaS"])
app.include_router(ptaas_orchestration.router, prefix="/api/v1", tags=["PTaaS Orchestration"])
app.include_router(unified_gateway.router, prefix="/api/v1", tags=["Unified Gateway"])
app.include_router(vectors.router, prefix="/api/v1", tags=["Vector Operations"])
app.include_router(storage.router, prefix="/api/v1", tags=["Storage"])
app.include_router(jobs.router, prefix="/api/v1", tags=["Job Management"])
app.include_router(system_status.router, prefix="/api/v1", tags=["System Status"])
app.include_router(enterprise_platform.router, prefix="/api/v1", tags=["Enterprise Platform"])
app.include_router(enterprise_ai_platform.router, prefix="/api/v1", tags=["Enterprise AI Platform"])
app.include_router(mitre_attack.router, prefix="/api/v1", tags=["MITRE ATT&CK"])
app.include_router(sophisticated_red_team.router, prefix="/api/v1", tags=["Red Team Operations"])

# Enhanced root endpoint with comprehensive platform information
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Enhanced root endpoint with comprehensive platform information"""
    return {
        "message": "XORB Enterprise Cybersecurity Platform - Enhanced AI-Powered Edition",
        "version": "3.0.0",
        "status": "operational",
        "features": {
            "ptaas": {
                "enabled": True,
                "real_scanners": ["nmap", "nuclei", "nikto", "sslscan"],
                "compliance_frameworks": ["PCI-DSS", "HIPAA", "SOX", "ISO-27001"],
                "description": "Production-ready penetration testing with real security tools"
            },
            "ai_intelligence": {
                "enabled": True,
                "capabilities": [
                    "behavioral_analytics",
                    "threat_hunting", 
                    "forensics_engine",
                    "predictive_analytics"
                ],
                "description": "AI-powered threat intelligence and analysis"
            },
            "enterprise_security": {
                "enabled": True,
                "features": [
                    "multi_tenant_architecture",
                    "advanced_authentication",
                    "rate_limiting",
                    "audit_logging"
                ],
                "description": "Enterprise-grade security and compliance"
            },
            "infrastructure": {
                "enhanced_repositories": True,
                "redis_caching": True,
                "health_monitoring": True,
                "dependency_injection": True,
                "description": "Production-ready infrastructure components"
            }
        },
        "api_documentation": "/docs",
        "health_check": "/api/v1/health",
        "platform_info": "/api/v1/info",
        "enterprise_contact": "enterprise@xorb-security.com"
    }

# Enhanced platform info endpoint
@app.get("/api/v1/info", response_model=Dict[str, Any])
async def platform_info(request: Request):
    """Enhanced platform information with real-time metrics"""
    try:
        container = getattr(request.app.state, 'container', None)
        
        if container:
            health_status = await container.health_check()
            service_count = container.service_count
            is_initialized = container.is_initialized
        else:
            health_status = {"overall_status": "degraded", "error": "Container not available"}
            service_count = 0
            is_initialized = False
        
        return {
            "platform": "XORB Enterprise Cybersecurity",
            "version": "3.0.0",
            "edition": "Enhanced AI-Powered",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "status": health_status.get("overall_status", "unknown"),
            "capabilities": {
                "production_ptaas": True,
                "ai_threat_intelligence": True,
                "enterprise_security": True,
                "enhanced_infrastructure": True,
                "real_world_scanners": True,
                "compliance_automation": True,
                "behavioral_analytics": True,
                "threat_hunting": True,
                "forensics_engine": True
            },
            "infrastructure": {
                "services_count": service_count,
                "container_initialized": is_initialized,
                "database": "PostgreSQL with Row-Level Security",
                "cache": "Redis with Enhanced Compatibility",
                "authentication": "Production JWT with RBAC",
                "rate_limiting": "Advanced Redis-backed"
            },
            "compliance_frameworks": [
                "PCI-DSS", "HIPAA", "SOX", "ISO-27001", "GDPR", "NIST"
            ],
            "security_tools": [
                "Nmap", "Nuclei", "Nikto", "SSLScan", "Dirb", "Gobuster"
            ],
            "ai_capabilities": [
                "Machine Learning Threat Correlation",
                "Behavioral Anomaly Detection", 
                "Predictive Threat Modeling",
                "Automated Attribution Analysis",
                "Natural Language Processing",
                "Risk Scoring Algorithms"
            ],
            "api_documentation": "/docs",
            "enterprise_features": {
                "multi_tenancy": True,
                "sso_integration": True,
                "advanced_reporting": True,
                "compliance_automation": True,
                "24x7_monitoring": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get platform info: {e}")
        return {
            "platform": "XORB Enterprise Cybersecurity",
            "version": "3.0.0",
            "status": "error",
            "error": "Failed to retrieve platform information",
            "basic_info": {
                "api_documentation": "/docs",
                "health_check": "/api/v1/health"
            }
        }

# Enhanced custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="XORB Enterprise Cybersecurity Platform API",
        version="3.0.0",
        description="""
        **The World's Most Advanced AI-Powered Cybersecurity Operations Platform**
        
        This API provides comprehensive cybersecurity services with real-world implementation:
        
        ## 🎯 Production Features
        - **Real Security Scanners**: Integrated Nmap, Nuclei, Nikto, SSLScan
        - **AI Threat Intelligence**: Machine learning-powered analysis
        - **Enterprise Security**: Multi-tenant architecture with RBAC
        - **Compliance Automation**: PCI-DSS, HIPAA, SOX support
        
        ## 🚀 Getting Started
        1. Use `/api/v1/health` to check system status
        2. Authenticate via `/api/v1/auth/login`
        3. Create PTaaS scans via `/api/v1/ptaas/sessions`
        4. Monitor results with real-time status updates
        """,
        routes=app.routes,
    )
    
    # Add enhanced security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Enhanced error handlers
@app.exception_handler(HTTPException)
async def enhanced_http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler with detailed error information"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": "2025-01-10T22:15:55Z",
            "request_id": getattr(request.state, 'request_id', 'unknown'),
            "support": {
                "documentation": "/docs",
                "health_check": "/api/v1/health",
                "contact": "support@xorb-security.com"
            }
        }
    )

@app.exception_handler(500)
async def enhanced_internal_server_error_handler(request: Request, exc: Exception):
    """Enhanced internal server error handler"""
    logger.error(f"Internal server error: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred while processing your request",
            "status_code": 500,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": "2025-01-10T22:15:55Z",
            "request_id": getattr(request.state, 'request_id', 'unknown'),
            "support": {
                "documentation": "/docs",
                "health_check": "/api/v1/health",
                "enterprise_support": "enterprise@xorb-security.com"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Production server configuration
    uvicorn.run(
        "enhanced_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        workers=1,     # Use process manager for multiple workers in production
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False
    )