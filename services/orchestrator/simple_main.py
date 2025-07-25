#!/usr/bin/env python3
"""
Xorb PTaaS Orchestrator Service - Simple Version
Minimal orchestrator without conflicting imports
"""

import os
import sys
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Simple app without complex logging
app = FastAPI(
    title="Xorb PTaaS Orchestrator",
    description="Campaign orchestration and agent management",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Xorb PTaaS Orchestrator",
        "version": "2.0.0",
        "status": "operational",
        "message": "Campaign orchestration service ready",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "xorb-orchestrator",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/campaigns")
async def list_campaigns():
    """List active campaigns"""
    return {
        "campaigns": [],
        "total": 0,
        "message": "Campaign management system ready"
    }

@app.get("/api/v1/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": [],
        "total": 0,
        "message": "Agent discovery system ready"
    }

@app.get("/api/v1/orchestrator/status")
async def orchestrator_status():
    """Orchestrator status"""
    return {
        "orchestrator_status": "operational",
        "active_campaigns": 0,
        "active_agents": 0,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    
    print(f"🚀 Starting Xorb PTaaS Orchestrator on {host}:{port}")
    
    uvicorn.run(
        "simple_main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )