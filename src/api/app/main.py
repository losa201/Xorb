from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import os
import redis.asyncio as redis
from .middleware.advanced_rate_limiter import RateLimitingMiddleware
from .middleware.audit_logging import AuditLoggingMiddleware
import asyncio
from urllib.parse import urlparse

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
# Security middleware (Redis-backed)
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.from_url(redis_url)
app.add_middleware(RateLimitingMiddleware, redis_client=redis_client)
app.add_middleware(AuditLoggingMiddleware, redis_client=redis_client)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS (restrict in production; allow local PTaaS and Temporal UI by default)
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8081",
]
cors_origins = os.getenv("CORS_ALLOW_ORIGINS", ",".join(default_origins)).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
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
    # Basic service health; extend to check Redis/Temporal/DB
    health = {"status": "healthy", "version": app.version}
    try:
        pong = await redis_client.ping()
        health["redis"] = "ok" if pong else "error"
    except Exception:
        health["redis"] = "error"
        health["status"] = "degraded"
    return health


async def _tcp_check(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.close()
        if hasattr(writer, 'wait_closed'):
            await writer.wait_closed()
        return True
    except Exception:
        return False


def _parse_db_host_port(db_url: str) -> tuple[str, int]:
    parsed = urlparse(db_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    return host, port


async def _db_query_ok(dsn: str, timeout: float = 1.5) -> bool:
    try:
        import asyncpg
        conn = await asyncio.wait_for(asyncpg.connect(dsn=dsn), timeout)
        try:
            val = await asyncio.wait_for(conn.fetchval('SELECT 1'), timeout)
        finally:
            await conn.close()
        return val == 1
    except Exception:
        return False


@app.get("/readiness", tags=["Health"]) 
async def readiness_check():
    """Readiness probe: checks Redis and TCP-level connectivity to Postgres and Temporal."""
    status = "ready"
    details = {}

    # Redis
    try:
        pong = await redis_client.ping()
        details["redis"] = "ok" if pong else "error"
        if not pong:
            status = "not_ready"
    except Exception:
        details["redis"] = "error"
        status = "not_ready"

    # Postgres TCP
    db_url = os.getenv("DATABASE_URL", "postgresql://temporal:temporal@postgres:5432/temporal")
    db_host, db_port = _parse_db_host_port(db_url)
    if not await _tcp_check(db_host, db_port):
        details["postgres"] = "error"
        status = "not_ready"
    else:
        # Optional query-level check
        if await _db_query_ok(db_url):
            details["postgres"] = "ok"
        else:
            details["postgres"] = "degraded"
            status = "not_ready"

    # Temporal TCP
    temporal = os.getenv("TEMPORAL_HOST", "temporal:7233")
    if ":" in temporal:
        thost, tport = temporal.split(":", 1)
        try:
            tport = int(tport)
        except ValueError:
            thost, tport = temporal, 7233
    else:
        thost, tport = temporal, 7233

    if not await _tcp_check(thost, tport):
        details["temporal"] = "error"
        status = "not_ready"
    else:
        details["temporal"] = "ok"

    code = 200 if status == "ready" else 503
    return {"status": status, **details}, code
