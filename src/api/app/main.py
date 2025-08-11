"""
XORB Enterprise Cybersecurity Platform - Main Application
Production-ready FastAPI application with comprehensive security, monitoring, and performance optimizations
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Core imports
from .core.config import get_settings, get_config_manager
from .core.logging import setup_logging, LoggingMiddleware, get_logger
from .core.secure_logging import get_audit_logger, get_secure_logger, LoggingSecurityConfig
from .core.metrics import setup_metrics, get_metrics_service
from .core.security import setup_security, SecurityHeadersMiddleware
from .core.cache import setup_cache, get_cache_service
from .core.database import setup_database, get_database_manager
from .core.error_handling import global_exception_handler, get_error_handler

# Security middleware imports
from .middleware.input_validation import InputValidationMiddleware, get_validation_config

# Router imports
from .routers import (
    health, auth, discovery, embeddings, ptaas, telemetry,
    orchestration, agents, enterprise_management, security_dashboard
)

# Get configuration
config_manager = get_config_manager()
settings = config_manager.app_settings

# Setup secure logging early
logging_security_config = LoggingSecurityConfig(
    mask_sensitive_fields=True,
    hash_pii=config_manager.is_production(),
    enable_audit_trail=True,
    log_retention_days=90 if config_manager.is_production() else 30
)

setup_logging(
    log_level=settings.log_level,
    environment=settings.environment,
    enable_json=(settings.log_format == "json"),
    enable_colors=(settings.environment == "development")
)

# Initialize secure audit logger
audit_logger = get_audit_logger()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting XORB Enterprise Cybersecurity Platform")
    
    # Validate configuration
    config_issues = config_manager.validate_configuration()
    if config_issues:
        logger.warning("Configuration issues detected", issues=config_issues)
        if config_manager.is_production():
            raise RuntimeError(f"Critical configuration issues in production: {config_issues}")
    
    # Initialize core services
    try:
        # Setup security
        security_service = setup_security(config_manager.security_config)
        logger.info("✅ Security service initialized")
        
        # Setup metrics
        metrics_service = setup_metrics(config_manager.metric_config)
        await metrics_service.start()
        logger.info("✅ Metrics service started")
        
        # Setup cache
        cache_service = setup_cache(config_manager.cache_config)
        logger.info("✅ Cache service initialized")
        
        # Setup database
        database_manager = setup_database(config_manager.database_config)
        await database_manager.initialize()
        logger.info("✅ Database manager initialized")
        
        # Log configuration summary
        config_summary = config_manager.get_configuration_summary()
        logger.info("Configuration loaded", **config_summary)
        
        logger.info("🎯 XORB platform started successfully")
        
    except Exception as e:
        logger.critical("❌ Failed to initialize services", error=str(e))
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down XORB platform")
    
    try:
        if get_metrics_service():
            await get_metrics_service().stop()
            logger.info("✅ Metrics service stopped")
        
        if get_cache_service():
            await get_cache_service().close()
            logger.info("✅ Cache service closed")
        
        if get_database_manager():
            await get_database_manager().close()
            logger.info("✅ Database connections closed")
        
        logger.info("✅ XORB platform shutdown complete")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="XORB Enterprise Cybersecurity Platform",
    description="""
    🛡️ **XORB Enterprise Cybersecurity Platform** - Advanced AI-Powered Security Operations
    
    A comprehensive cybersecurity platform offering:
    
    ## 🔧 Core Features
    - **PTaaS (Penetration Testing as a Service)** - Real-world security scanner integration
    - **AI-Powered Threat Intelligence** - Advanced ML-based threat analysis
    - **Compliance Automation** - PCI-DSS, HIPAA, SOX, ISO-27001, GDPR support
    - **Enterprise SSO & Multi-tenancy** - Production-grade authentication
    - **Real-time Security Monitoring** - Continuous threat detection
    - **Advanced Analytics** - Behavioral analysis and anomaly detection
    
    ## 🚀 Enterprise Capabilities
    - **Production-ready Architecture** - Microservices with clean architecture
    - **Advanced Security** - JWT, MFA, rate limiting, audit logging
    - **High Performance** - Connection pooling, caching, metrics
    - **Comprehensive Monitoring** - Prometheus metrics, health checks
    - **Scalable Infrastructure** - Docker, Kubernetes, Redis clustering
    
    ## 🛠️ Security Scanner Integration
    - **Nmap** - Network discovery and port scanning
    - **Nuclei** - Modern vulnerability scanner (3000+ templates)
    - **Nikto** - Web application security scanner
    - **SSLScan** - SSL/TLS configuration analysis
    - **Custom Security Checks** - Advanced vulnerability analysis
    
    ## 📊 Compliance Frameworks
    - PCI-DSS (Payment Card Industry)
    - HIPAA (Healthcare Data Protection)
    - SOX (Sarbanes-Oxley)
    - ISO-27001 (Information Security Management)
    - GDPR (General Data Protection Regulation)
    - NIST (National Institute of Standards)
    
    ## 🔐 Security Features
    - **Zero Trust Architecture** - Never trust, always verify
    - **Advanced Threat Hunting** - Custom query language
    - **Forensics Engine** - Legal-grade evidence collection
    - **Behavioral Analytics** - ML-powered user behavior analysis
    - **Network Microsegmentation** - Zero-trust network policies
    
    ## 📈 Monitoring & Analytics
    - **Real-time Dashboards** - Grafana integration
    - **Custom Metrics** - Prometheus monitoring
    - **Advanced Alerting** - Multi-channel notifications
    - **Performance Analytics** - APM and system monitoring
    - **Audit Trails** - Comprehensive logging and compliance
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url=None,  # We'll customize these
    redoc_url=None,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    swagger_ui_oauth2_redirect_url=f"{settings.api_prefix}/docs/oauth2-redirect",
)

# Add security middleware stack (order matters!)
security_headers = SecurityHeadersMiddleware(config_manager.security_config)

# 1. Input validation middleware (first line of defense)
validation_preset = "strict" if config_manager.is_production() else "moderate"
validation_config = get_validation_config(validation_preset)
app.add_middleware(InputValidationMiddleware, config=validation_config)

# 2. Logging middleware
app.middleware("http")(LoggingMiddleware(app))

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses"""
    response = await call_next(request)
    
    # Add security headers
    headers = security_headers.get_security_headers()
    for header, value in headers.items():
        response.headers[header] = value
    
    return response

# Add CORS middleware with security checks
cors_origins = settings.get_cors_origins()
if cors_origins:
    # Validate CORS origins in production
    validated_origins = []
    for origin in cors_origins:
        if settings.environment == "production" and origin == "*":
            logger.warning("Wildcard CORS origin not allowed in production")
            continue
        validated_origins.append(origin)
    
    if validated_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=validated_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=settings.get_cors_methods(),
            allow_headers=settings.get_cors_headers(),
            max_age=settings.cors_max_age,
        )
        logger.info("CORS middleware configured", origins=validated_origins)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add trusted host middleware for production
if config_manager.is_production():
    import os
    # Get allowed hosts from environment or config
    allowed_hosts = os.getenv("ALLOWED_HOSTS", "api.xorb.enterprise,xorb.enterprise").split(",")
    allowed_hosts = [host.strip() for host in allowed_hosts if host.strip()]
    
    if not allowed_hosts:
        allowed_hosts = ["api.xorb.enterprise"]  # Default production host
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )
    logger.info("Trusted host middleware configured", allowed_hosts=allowed_hosts)

# Global exception handler
app.add_exception_handler(Exception, global_exception_handler)

# Include routers
app.include_router(health.router, prefix=settings.api_prefix, tags=["Health"])
app.include_router(auth.router, prefix=settings.api_prefix, tags=["Authentication"])
app.include_router(discovery.router, prefix=settings.api_prefix, tags=["Discovery"])
app.include_router(embeddings.router, prefix=settings.api_prefix, tags=["Embeddings"])
app.include_router(ptaas.router, prefix=settings.api_prefix, tags=["PTaaS"])
app.include_router(telemetry.router, prefix=settings.api_prefix, tags=["Telemetry"])
app.include_router(orchestration.router, prefix=settings.api_prefix, tags=["Orchestration"])
app.include_router(agents.router, prefix=settings.api_prefix, tags=["Agents"])
app.include_router(security_dashboard.router, tags=["Security Dashboard"])

# Include enterprise router with error handling
try:
    app.include_router(enterprise_management.router, prefix=settings.api_prefix, tags=["Enterprise"])
    logger.info("✅ Enterprise Management router loaded")
except ImportError as e:
    logger.warning("Enterprise Management not available", error=str(e))

# Include enhanced PTaaS orchestration
try:
    from .routers import enhanced_ptaas_orchestration
    app.include_router(enhanced_ptaas_orchestration.router, prefix=settings.api_prefix, tags=["Enhanced PTaaS"])
    logger.info("✅ Enhanced PTaaS Orchestration router loaded")
except ImportError as e:
    logger.warning("Enhanced PTaaS Orchestration router not available", error=str(e))

# Include strategic PTaaS enhancements with proper import handling
strategic_routers = [
    ("strategic_ptaas_enhancement", "Strategic PTaaS"),
    ("enterprise_compliance_automation", "Enterprise Compliance")
]

for router_name, display_name in strategic_routers:
    try:
        # Try direct import first, skip if not found
        module = __import__(f"app.routers.{router_name}", fromlist=["router"])
        app.include_router(module.router, prefix=settings.api_prefix, tags=[display_name])
        logger.info(f"✅ {display_name} router loaded")
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"{display_name} router not available", error=str(e))

# Include strategic principal auditor router
try:
    from .routers import strategic_principal_auditor_ptaas
    app.include_router(strategic_principal_auditor_ptaas.router, prefix=settings.api_prefix, tags=["Strategic PTaaS"])
    logger.info("✅ Strategic Principal Auditor PTaaS router loaded")
except ImportError as e:
    logger.warning("Strategic PTaaS router not available", error=str(e))

# Include additional routers with graceful degradation
optional_routers = [
    ("redis_management", "Redis Management"),
    ("advanced_networking", "Advanced Networking"),
    ("enterprise_ptaas", "Enterprise PTaaS"),
    ("red_blue_agents", "Red/Blue Team Agents"),
    # ("principal_auditor_enhanced_ptaas", "Principal Auditor Enhanced PTaaS")  # Disabled due to Pydantic compatibility
]

for router_name, display_name in optional_routers:
    try:
        # Try importing the router module with proper error handling
        try:
            from importlib import import_module
            module = import_module(f"app.routers.{router_name}")
        except ImportError:
            # Module doesn't exist, skip it
            continue
        app.include_router(module.router, prefix=settings.api_prefix, tags=[display_name])
        logger.info(f"✅ {display_name} router loaded")
    except ImportError as e:
        logger.warning(f"{display_name} router not available", error=str(e))


# Custom OpenAPI documentation
def custom_openapi():
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /auth/login endpoint"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service authentication"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []}
    ]
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{settings.api_port}",
            "description": "Development server"
        },
        {
            "url": f"https://api.xorb.enterprise",
            "description": "Production server"
        }
    ]
    
    # Add additional metadata
    openapi_schema["info"]["contact"] = {
        "name": "XORB Support",
        "email": "support@xorb.enterprise",
        "url": "https://docs.xorb.enterprise"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "Enterprise License",
        "url": "https://xorb.enterprise/license"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "none",
            "operationsSorter": "method",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True
        }
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc documentation"""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js",
    )


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with platform information"""
    return {
        "message": "🛡️ XORB Enterprise Cybersecurity Platform",
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "operational",
        "documentation": "/docs",
        "health": f"{settings.api_prefix}/health",
        "features": config_manager.get_feature_flags()
    }


# Development server startup
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload_on_change,
        workers=1 if settings.reload_on_change else settings.api_workers,
        log_level=settings.log_level.lower(),
        access_log=True,
        use_colors=settings.environment == "development"
    )