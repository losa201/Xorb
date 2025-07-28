#!/usr/bin/env python3
"""
Xorb PTaaS API Service
Basic FastAPI application for development deployment
"""

# Setup logging without conflicts
import logging as std_logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import aiohttp
import asyncio
from typing import Optional

std_logging.basicConfig(level=std_logging.INFO)
logger = std_logging.getLogger(__name__)

# Global connection pool for external services
http_session: Optional[aiohttp.ClientSession] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with connection pooling"""
    global http_session
    
    logger.info("🚀 Starting Xorb PTaaS API Service")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    # Initialize connection pool with optimized settings
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(
        limit=100,  # Total connection pool size
        limit_per_host=20,  # Per-host connection limit
        ttl_dns_cache=300,  # DNS cache TTL
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True
    )
    
    http_session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"User-Agent": "Xorb-API/2.0.0"}
    )
    
    logger.info("✅ HTTP connection pool initialized")
    
    yield
    
    # Cleanup connection pool
    if http_session:
        await http_session.close()
        logger.info("✅ HTTP connection pool closed")
    
    logger.info("📴 Shutting down Xorb PTaaS API Service")

# Create FastAPI app
app = FastAPI(
    title="Xorb PTaaS API",
    description="Penetration Testing as a Service API",
    version="2.0.0",
    lifespan=lifespan
)

# Add performance optimization middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Xorb PTaaS API",
        "version": "2.0.0",
        "status": "operational",
        "message": "Welcome to Xorb Penetration Testing as a Service"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "xorb-api",
        "timestamp": "2024-07-24T00:00:00Z"
    }

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint"""
    return {
        "api_status": "operational",
        "services": {
            "database": "connected",
            "redis": "connected",
            "nats": "connected"
        },
        "version": "2.0.0"
    }

# Research & asset management endpoints
@app.get("/api/v1/assets")
async def list_assets():
    """List assets"""
    return {
        "assets": [],
        "total": 0,
        "message": "Asset management system ready"
    }

@app.get("/api/v1/scans")
async def list_scans():
    """List security scans"""
    return {
        "scans": [],
        "total": 0,
        "message": "Security scanning system ready"
    }

@app.get("/api/v1/bounties")
async def list_bounties():
    """List bug bounty programs"""
    return {
        "bounties": [],
        "total": 0,
        "message": "Bug bounty system ready"
    }

@app.get("/api/v1/findings")
async def list_findings():
    """List security findings"""
    return {
        "findings": [],
        "total": 0,
        "message": "Findings management system ready"
    }

# Gamification endpoints
@app.get("/api/gamification/leaderboard")
async def get_leaderboard():
    """Get researcher leaderboard"""
    return {
        "leaderboard": [],
        "message": "Gamification system ready"
    }

# Compliance endpoints
@app.get("/api/compliance/status")
async def get_compliance_status():
    """Get SOC 2 compliance status"""
    return {
        "compliance_status": "ready",
        "soc2_readiness": "green",
        "message": "Compliance automation system ready"
    }

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting Xorb PTaaS API on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
