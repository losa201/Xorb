from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .container import get_container
from fastapi import Depends
try:
    from .dependencies import require_user, require_reader
except ImportError:
    # Fallback dependencies for standalone operation
    def require_user():
        return None
    def require_reader():
        return None

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Instrumentator = None

# Try multiple import paths for compatibility
try:
    from packages.xorb_core.xorb_core.logging import configure_logging, get_logger
except ImportError:
    try:
        from xorb_core.logging import configure_logging, get_logger
    except ImportError:
        # Fallback logging configuration
        import logging
        logging.basicConfig(level=logging.INFO)
        def configure_logging(level="INFO", service_name="xorb-api"):
            pass
        def get_logger(name):
            return logging.getLogger(name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    container = get_container()
    await container.initialize()
    
    log.info("Xorb API starting up with clean architecture")
    
    yield
    
    # Shutdown
    log.info("Xorb API shutting down")


# Configure structured logging for API service
configure_logging(level="INFO", service_name="xorb-api")
log = get_logger(__name__)

app = FastAPI(
    title="Xorb API", 
    version="3.0.0",
    description="Xorb API with Clean Architecture",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[".*admin.*", "/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="xorb_api_requests_inprogress",
        inprogress_labels=True
    )
    instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)
else:
    log.warning("Prometheus instrumentation not available")

# Import and include routers
try:
    from .routers import auth
    app.include_router(auth.router, tags=["Authentication"])
    log.info("Auth router included")
except ImportError as e:
    log.warning(f"Could not import auth router: {e}")

try:
    from .routers import discovery
    app.include_router(
        discovery.router,
        prefix="/v1",
        tags=["Discovery"]
    )
    log.info("Discovery router included")
except ImportError as e:
    log.warning(f"Could not import discovery router: {e}")

try:
    from .routers import embeddings
    app.include_router(
        embeddings.router,
        prefix="/v1",
        tags=["Embeddings"]
    )
    log.info("Embeddings router included")
except ImportError as e:
    log.warning(f"Could not import embeddings router: {e}")

# Include new XORB API routers
try:
    from .routers import agents
    app.include_router(
        agents.router,
        prefix="/v1",
        tags=["Agent Management"]
    )
    log.info("Agents router included")
except ImportError as e:
    log.warning(f"Could not import agents router: {e}")

try:
    from .routers import orchestration
    app.include_router(
        orchestration.router,
        prefix="/v1",
        tags=["Task Orchestration"]
    )
    log.info("Orchestration router included")
except ImportError as e:
    log.warning(f"Could not import orchestration router: {e}")

try:
    from .routers import security_ops
    app.include_router(
        security_ops.router,
        prefix="/v1",
        tags=["Security Operations"]
    )
    log.info("Security Operations router included")
except ImportError as e:
    log.warning(f"Could not import security_ops router: {e}")

try:
    from .routers import intelligence
    app.include_router(
        intelligence.router,
        prefix="/v1",
        tags=["Intelligence Integration"]
    )
    log.info("Intelligence router included")
except ImportError as e:
    log.warning(f"Could not import intelligence router: {e}")

try:
    from .routers import telemetry
    app.include_router(
        telemetry.router,
        prefix="/v1",
        tags=["Telemetry & Monitoring"]
    )
    log.info("Telemetry router included")
except ImportError as e:
    log.warning(f"Could not import telemetry router: {e}")

# Include legacy routers with error handling (for compatibility)
# These routers need additional refactoring to work with the new architecture
log.info("Legacy routers (compliance, gamification, knowledge, swarm) are available but may need updates")



@app.get("/health", tags=["Health"])
async def health_check():
    log.info("Health check requested")
    return {"status": "ok"}
