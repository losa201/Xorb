from fastapi import FastAPI, Depends
from prometheus_fastapi_instrumentator import Instrumentator

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

# Import routers with error handling
routers_to_import = []
try:
    from .routers import auth
    routers_to_import.append(("auth", auth))
except ImportError as e:
    print(f"Warning: Could not import auth router: {e}")

try:
    from .routers import discovery
    routers_to_import.append(("discovery", discovery))
except ImportError as e:
    print(f"Warning: Could not import discovery router: {e}")

try:
    from .routers import embeddings
    routers_to_import.append(("embeddings", embeddings))
except ImportError as e:
    print(f"Warning: Could not import embeddings router: {e}")

# Skip knowledge router for now due to parameter issues
# try:
#     from .routers import knowledge
#     routers_to_import.append(("knowledge", knowledge))
# except ImportError as e:
#     print(f"Warning: Could not import knowledge router: {e}")
from .deps import has_role

# Configure structured logging for API service
configure_logging(level="INFO", service_name="xorb-api")
log = get_logger(__name__)

app = FastAPI(title="Xorb API", version="2.0.0")

# Configure Prometheus metrics
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

# Include available routers
for router_name, router_module in routers_to_import:
    if router_name == "auth":
        app.include_router(router_module.router, tags=["Authentication"])
    elif router_name == "discovery":
        app.include_router(
            router_module.router,
            prefix="/v1",
            tags=["Discovery"],
            dependencies=[Depends(has_role("reader"))],
        )
    elif router_name == "embeddings":
        app.include_router(
            router_module.router,
            prefix="/v1",
            tags=["Embeddings"],
            dependencies=[Depends(has_role("user"))],
        )
    elif router_name == "knowledge":
        app.include_router(
            router_module.router,
            prefix="/v1/knowledge",
            tags=["Knowledge Fabric"],
            dependencies=[Depends(has_role("user"))],
        )

@app.on_event("startup")
async def startup_event():
    log.info("Xorb API starting up")

@app.get("/health", tags=["Health"])
async def health_check():
    log.info("Health check requested")
    return {"status": "ok"}
