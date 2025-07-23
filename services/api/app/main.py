from fastapi import FastAPI
from starlette_prometheus import metrics, PrometheusMiddleware
from .routers import discovery

app = FastAPI(title="Xorb API", version="2.0.0")

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

app.include_router(discovery.router, prefix="/v1", tags=["Discovery"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
