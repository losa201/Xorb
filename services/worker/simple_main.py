#!/usr/bin/env python3
"""
Xorb PTaaS Worker Service - Simple Version
Minimal worker with HTTP health endpoint
"""

import os
import sys
import asyncio
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Simple app for health checks
app = FastAPI(
    title="Xorb PTaaS Worker",
    description="Background task processing worker",
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

# Global worker state
worker_state = {
    "running": False,
    "tasks_processed": 0,
    "started_at": None
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Xorb PTaaS Worker",
        "version": "2.0.0",
        "status": "operational",
        "message": "Background task processing worker",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "xorb-worker",
        "running": worker_state["running"],
        "tasks_processed": worker_state["tasks_processed"],
        "started_at": worker_state["started_at"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/worker/status")
async def worker_status():
    """Worker status endpoint"""
    return {
        "worker_status": "operational",
        "tasks_processed": worker_state["tasks_processed"],
        "running": worker_state["running"],
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

class SimpleWorker:
    """Simple worker implementation"""
    
    def __init__(self):
        self.running = False
        
    async def start(self):
        """Start the worker"""
        self.running = True
        worker_state["running"] = True
        worker_state["started_at"] = datetime.now().isoformat()
        
        print(f"🚀 Starting Xorb PTaaS Worker - {datetime.now().isoformat()}")
        print("📋 Worker ready to process tasks")
        
        try:
            while self.running:
                # Simulate task processing
                await asyncio.sleep(30)
                worker_state["tasks_processed"] += 1
                print(f"💼 Worker heartbeat - Tasks processed: {worker_state['tasks_processed']}")
                
        except Exception as e:
            print(f"❌ Worker error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the worker"""
        self.running = False
        worker_state["running"] = False
        print("📴 Worker shutting down gracefully")

# Global worker instance
worker = SimpleWorker()

@app.on_event("startup")
async def startup_event():
    """Start the worker when the app starts"""
    asyncio.create_task(worker.start())

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9090"))
    
    print(f"🚀 Starting Xorb PTaaS Worker on {host}:{port}")
    
    uvicorn.run(
        "simple_main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )