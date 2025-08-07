#!/usr/bin/env python3
"""
XORB Evolution Accelerator Service Deployment
Integration with the autonomous AI orchestration platform
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from xorb_autonomous_evolution_accelerator import XORBEvolutionAccelerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="XORB Evolution Accelerator Service",
    description="Advanced autonomous AI evolution and emergent behavior detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global evolution accelerator instance
evolution_accelerator: Optional[XORBEvolutionAccelerator] = None

# Pydantic models for API
class AgentEvolutionRequest(BaseModel):
    agent_id: str
    success_rate: Optional[float] = 0.5
    learning_efficiency: Optional[float] = 0.5
    collaboration_effectiveness: Optional[float] = 0.5
    neural_layers: Optional[int] = 4
    hidden_units: Optional[list] = [128, 64, 32]
    learning_rate: Optional[float] = 0.001
    exploration_rate: Optional[float] = 0.1
    adaptation_speed: Optional[float] = 0.1
    innovation_score: Optional[float] = 0.5
    resource_efficiency: Optional[float] = 0.5

class BehaviorDetectionRequest(BaseModel):
    agent_id: str
    behavior_type: Optional[str] = "unknown"
    description: Optional[str] = "Behavior observation"
    sequence: list
    context: Optional[dict] = {}
    efficiency: Optional[float] = 0.5
    innovation_score: Optional[float] = 0.5

class MetaLearningRequest(BaseModel):
    learning_experiences: list
    agent_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the evolution accelerator on startup"""
    global evolution_accelerator
    try:
        evolution_accelerator = XORBEvolutionAccelerator({
            'mutation_rate': 0.12,
            'crossover_rate': 0.35,
            'selection_pressure': 0.8,
            'novelty_threshold': 0.82,
            'meta_learning_window': 75
        })
        logger.info("XORB Evolution Accelerator Service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize evolution accelerator: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = await evolution_accelerator.get_evolution_status() if evolution_accelerator else {}
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "evolution_accelerator",
            "evolution_stats": {
                "total_genomes": status.get("total_genomes", 0),
                "emergent_behaviors": status.get("emergent_behaviors", 0),
                "meta_insights": status.get("meta_learning_insights", 0),
                "active_agents": status.get("active_agents", 0)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evolve/agent")
async def evolve_agent(request: AgentEvolutionRequest, background_tasks: BackgroundTasks):
    """Evolve a specific agent"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        agent_data = request.dict()
        
        # Run evolution in background for performance
        evolution_result = await evolution_accelerator.accelerate_agent_evolution(agent_data)
        
        if evolution_result.get('success'):
            logger.info(f"Agent evolution successful: {request.agent_id}")
            return {
                "success": True,
                "agent_id": request.agent_id,
                "evolved_genome_id": evolution_result.get("evolved_genome_id"),
                "evolution_methods": evolution_result.get("evolution_methods", []),
                "fitness_improvement": evolution_result.get("fitness_improvement", 0.0),
                "generation": evolution_result.get("generation", 1),
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.warning(f"Agent evolution failed: {request.agent_id} - {evolution_result.get('reason')}")
            return {
                "success": False,
                "agent_id": request.agent_id,
                "reason": evolution_result.get("reason", "Unknown failure"),
                "error": evolution_result.get("error")
            }
            
    except Exception as e:
        logger.error(f"Agent evolution endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/behavior")
async def detect_emergent_behavior(request: BehaviorDetectionRequest):
    """Detect emergent behaviors"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        behavior_data = request.dict()
        
        detection_result = await evolution_accelerator.detect_emergent_behaviors(behavior_data)
        
        if detection_result.get('behavior_detected'):
            logger.info(f"Emergent behavior detected: {request.agent_id}")
            return {
                "behavior_detected": True,
                "agent_id": request.agent_id,
                "behavior_id": detection_result.get("behavior_id"),
                "novelty_score": detection_result.get("novelty_score", 0.0),
                "effectiveness_score": detection_result.get("effectiveness_score", 0.0),
                "impact_assessment": detection_result.get("impact_assessment", {}),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "behavior_detected": False,
                "agent_id": request.agent_id,
                "novelty_score": detection_result.get("novelty_score", 0.0),
                "reason": "Behavior not sufficiently novel"
            }
            
    except Exception as e:
        logger.error(f"Behavior detection endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/insights")
async def generate_meta_insights(request: MetaLearningRequest):
    """Generate meta-learning insights"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        insight_result = await evolution_accelerator.generate_meta_learning_insights(
            request.learning_experiences
        )
        
        if insight_result.get('insight_generated'):
            logger.info(f"Meta-learning insights generated: {insight_result.get('insights_created', 0)}")
            return {
                "insight_generated": True,
                "insights_created": insight_result.get("insights_created", 0),
                "insight_ids": insight_result.get("insight_ids", []),
                "agent_id": request.agent_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "insight_generated": False,
                "reason": insight_result.get("reason", "Insufficient data"),
                "agent_id": request.agent_id
            }
            
    except Exception as e:
        logger.error(f"Meta-learning insights endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/evolution")
async def get_evolution_status():
    """Get comprehensive evolution system status"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        status = await evolution_accelerator.get_evolution_status()
        return status
        
    except Exception as e:
        logger.error(f"Evolution status endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/genomes/list")
async def list_genomes():
    """List all evolution genomes"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        genomes = []
        for genome_id, genome in evolution_accelerator.evolution_genomes.items():
            genomes.append({
                "genome_id": genome.genome_id,
                "agent_id": genome.agent_id,
                "fitness_score": genome.fitness_score,
                "generation": genome.generation,
                "created_at": genome.created_at.isoformat(),
                "parent_genomes": genome.parent_genomes,
                "mutation_count": len(genome.mutation_history)
            })
        
        return {
            "total_genomes": len(genomes),
            "genomes": genomes
        }
        
    except Exception as e:
        logger.error(f"Genome listing endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/behaviors/emergent")
async def list_emergent_behaviors():
    """List all detected emergent behaviors"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        behaviors = []
        for behavior_id, behavior in evolution_accelerator.emergent_behaviors.items():
            behaviors.append({
                "behavior_id": behavior.behavior_id,
                "agent_id": behavior.agent_id,
                "behavior_type": behavior.behavior_type,
                "description": behavior.description,
                "novelty_score": behavior.novelty_score,
                "effectiveness_score": behavior.effectiveness_score,
                "reproducibility_score": behavior.reproducibility_score,
                "first_observed": behavior.first_observed.isoformat(),
                "observation_count": behavior.observation_count,
                "impact_assessment": behavior.impact_assessment
            })
        
        return {
            "total_behaviors": len(behaviors),
            "emergent_behaviors": behaviors
        }
        
    except Exception as e:
        logger.error(f"Emergent behaviors endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insights/meta")
async def list_meta_insights():
    """List all meta-learning insights"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        insights = []
        for insight_id, insight in evolution_accelerator.meta_learning_insights.items():
            insights.append({
                "insight_id": insight.insight_id,
                "insight_type": insight.insight_type,
                "generalization_potential": insight.generalization_potential,
                "transfer_probability": insight.transfer_probability,
                "validation_confidence": insight.validation_confidence,
                "actionable_recommendations": insight.actionable_recommendations,
                "timestamp": insight.timestamp.isoformat(),
                "source_experiences_count": len(insight.source_experiences)
            })
        
        return {
            "total_insights": len(insights),
            "meta_learning_insights": insights
        }
        
    except Exception as e:
        logger.error(f"Meta-learning insights endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class BatchEvolutionRequest(BaseModel):
    agents: list

@app.post("/evolution/batch")
async def batch_evolution(request: BatchEvolutionRequest):
    """Perform batch evolution for multiple agents"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        results = []
        
        for agent_data in request.agents:
            try:
                evolution_result = await evolution_accelerator.accelerate_agent_evolution(agent_data)
                results.append({
                    "agent_id": agent_data.get("agent_id"),
                    "success": evolution_result.get("success", False),
                    "evolution_result": evolution_result
                })
            except Exception as e:
                results.append({
                    "agent_id": agent_data.get("agent_id"),
                    "success": False,
                    "error": str(e)
                })
        
        successful_evolutions = len([r for r in results if r["success"]])
        
        return {
            "batch_success": True,
            "total_agents": len(request.agents),
            "successful_evolutions": successful_evolutions,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch evolution endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get evolution system performance metrics"""
    try:
        if not evolution_accelerator:
            raise HTTPException(status_code=503, detail="Evolution accelerator not initialized")
        
        status = await evolution_accelerator.get_evolution_status()
        
        # Calculate additional performance metrics
        recent_behaviors = len([b for b in evolution_accelerator.emergent_behaviors.values() 
                              if (datetime.now() - b.first_observed).total_seconds() < 3600])
        
        recent_evolutions = len([e for e in evolution_accelerator.evolution_history 
                               if (datetime.now() - datetime.fromisoformat(e.get('timestamp', '2023-01-01T00:00:00'))).total_seconds() < 3600])
        
        return {
            "performance_metrics": {
                "total_genomes": status.get("total_genomes", 0),
                "emergent_behaviors": status.get("emergent_behaviors", 0),
                "meta_learning_insights": status.get("meta_learning_insights", 0),
                "average_fitness": status.get("average_fitness", 0.0),
                "highest_generation": status.get("highest_generation", 0),
                "recent_behaviors_1h": recent_behaviors,
                "recent_evolutions_1h": recent_evolutions,
                "evolution_rate": recent_evolutions / 3600.0,  # Evolutions per second
                "novelty_detection_rate": recent_behaviors / 3600.0  # Behaviors per second
            },
            "system_health": {
                "accelerator_active": evolution_accelerator is not None,
                "memory_usage": len(evolution_accelerator.evolution_history),
                "behavior_patterns_stored": sum(len(patterns) for patterns in evolution_accelerator.behavior_patterns.values())
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the service
if __name__ == "__main__":
    print("🧬 Starting XORB Evolution Accelerator Service...")
    print("🚀 Service will be available at: http://localhost:8008")
    print("📊 API Documentation: http://localhost:8008/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8008,
        log_level="info",
        access_log=True
    )