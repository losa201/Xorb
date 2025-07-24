from fastapi import FastAPI, Depends
from prometheus_fastapi_instrumentator import Instrumentator
from xorb_core.logging import configure_logging, get_logger
from .routers import auth, discovery, embeddings, knowledge
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

app.include_router(auth.router, tags=["Authentication"])
app.include_router(
    discovery.router,
    prefix="/v1",
    tags=["Discovery"],
    dependencies=[Depends(has_role("reader"))],
)
app.include_router(
    embeddings.router,
    prefix="/v1",
    tags=["Embeddings"],
    dependencies=[Depends(has_role("user"))],
)
app.include_router(
    knowledge.router,
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
