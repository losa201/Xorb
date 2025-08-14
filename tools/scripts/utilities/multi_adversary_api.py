#!/usr/bin/env python3
"""
Multi-Adversary Simulation Framework API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import sys
import asyncio
sys.path.insert(0, '/root/Xorb')

from xorb_core.simulation import (
    SyntheticAdversaryProfileManager,
    MultiActorSimulationEngine,
    PredictiveThreatIntelligenceSynthesizer,
    CampaignGoalOptimizer,
    AdversaryType,
    SimulationMode,
    OptimizationStrategy
)

app = FastAPI(title="XORB Multi-Adversary Simulation API", version="2.0.0")

# Initialize framework components (would be done properly with dependency injection)
profile_manager = SyntheticAdversaryProfileManager()
simulation_engine = MultiActorSimulationEngine(profile_manager)
threat_synthesizer = PredictiveThreatIntelligenceSynthesizer()
goal_optimizer = CampaignGoalOptimizer()

@app.get("/")
async def root():
    return {"message": "XORB Multi-Adversary Simulation Framework API", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "framework": "operational"}

@app.get("/adversary-profiles")
async def list_adversary_profiles():
    try:
        profiles = await profile_manager.list_profiles()
        return {"profiles": [{"id": p.profile_id, "name": p.name, "type": p.adversary_type.value} for p in profiles]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulations")
async def list_simulations():
    try:
        simulations = list(simulation_engine.active_simulations.keys())
        return {"active_simulations": simulations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/framework-status")
async def framework_status():
    return {
        "profile_manager": "operational",
        "simulation_engine": "operational",
        "threat_synthesizer": "operational",
        "goal_optimizer": "operational",
        "integration": "complete"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
