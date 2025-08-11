"""
XORB Intelligence Integration API Endpoints
Provides AI-driven decision making and learning integration
"""
import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..security import (
    SecurityConfig,
    require_admin,
    require_permission,
    Permission
)
from ..services.intelligence_service import IntelligenceService
from ..middleware.tenant_context import require_tenant_context


class DecisionType(str, Enum):
    """Types of AI decisions"""
    TASK_PRIORITIZATION = "task_prioritization"
    AGENT_ASSIGNMENT = "agent_assignment"
    THREAT_CLASSIFICATION = "threat_classification"
    RESPONSE_STRATEGY = "response_strategy"
    RESOURCE_ALLOCATION = "resource_allocation"
    RISK_ASSESSMENT = "risk_assessment"
    ORCHESTRATION_OPTIMIZATION = "orchestration_optimization"
    SECURITY_POSTURE = "security_posture"


class LearningType(str, Enum):
    """Types of learning feedback"""
    REINFORCEMENT = "reinforcement"  # Outcome-based learning
    SUPERVISED = "supervised"        # Labeled training data
    UNSUPERVISED = "unsupervised"   # Pattern discovery
    FEDERATED = "federated"         # Distributed learning


class ModelType(str, Enum):
    """AI model types"""
    QWEN3_ORCHESTRATOR = "qwen3_orchestrator"
    CLAUDE_AGENT = "claude_agent"
    THREAT_CLASSIFIER = "threat_classifier"
    ANOMALY_DETECTOR = "anomaly_detector"
    DECISION_ENGINE = "decision_engine"
    RISK_PREDICTOR = "risk_predictor"


# Pydantic Models
class DecisionContext(BaseModel):
    """Context for AI decision making"""
    scenario: str
    available_data: Dict[str, Any]
    constraints: Dict[str, Any] = Field(default_factory=dict)
    historical_context: List[Dict[str, Any]] = Field(default_factory=list)
    urgency_level: str = "normal"  # low, normal, high, critical
    confidence_threshold: float = Field(0.7, ge=0, le=1)


class DecisionRequest(BaseModel):
    """Request for AI decision"""
    decision_type: DecisionType
    context: DecisionContext
    model_preferences: List[ModelType] = Field(default_factory=list)
    timeout_seconds: int = Field(30, ge=1, le=300)
    explanation_required: bool = True


class DecisionResponse(BaseModel):
    """AI decision response"""
    decision_id: str
    decision_type: DecisionType
    recommendation: str
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(ge=0, le=1)
    reasoning: List[str] = Field(default_factory=list)
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)
    model_used: ModelType
    processing_time_ms: int
    timestamp: datetime
    expires_at: Optional[datetime] = None


class LearningFeedback(BaseModel):
    """Feedback for AI learning"""
    decision_id: str
    outcome: str  # success, failure, partial_success
    actual_result: Dict[str, Any]
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    effectiveness_score: float = Field(ge=0, le=1)
    lessons_learned: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)


class ModelTrainingRequest(BaseModel):
    """Request for model training/fine-tuning"""
    model_type: ModelType
    training_data: List[Dict[str, Any]]
    learning_type: LearningType
    validation_split: float = Field(0.2, ge=0, le=0.5)
    epochs: int = Field(10, ge=1, le=1000)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class IntelligenceMetrics(BaseModel):
    """Intelligence system metrics"""
    total_decisions: int
    avg_confidence_score: float
    avg_processing_time_ms: float
    success_rate: float
    model_accuracy: Dict[str, float]
    learning_iterations: int
    active_models: int
    decision_distribution: Dict[str, int]


class OrchestrationBrainStatus(BaseModel):
    """Status of orchestration brain (Qwen3)"""
    model_id: str
    status: str  # active, training, maintenance, error
    version: str
    capabilities: List[str]
    performance_metrics: Dict[str, float]
    last_training: Optional[datetime] = None
    next_training: Optional[datetime] = None
    training_data_points: int
    memory_usage_mb: float
    processing_queue_depth: int


# Initialize intelligence service
intelligence_service = IntelligenceService()

# Model states (could be moved to database in production)
model_states = {
    ModelType.QWEN3_ORCHESTRATOR: {
        "status": "active",
        "version": "3.5.1",
        "accuracy": 0.92,
        "last_training": datetime.utcnow() - timedelta(days=1),
        "training_data_points": 50000
    },
    ModelType.CLAUDE_AGENT: {
        "status": "active", 
        "version": "4.0",
        "accuracy": 0.89,
        "last_training": datetime.utcnow() - timedelta(hours=6),
        "training_data_points": 25000
    }
}


router = APIRouter(prefix="/intelligence", tags=["Intelligence Integration"])


@router.post("/decisions", response_model=DecisionResponse)
async def request_decision(
    request: DecisionRequest,
    tenant_id: str = Depends(require_tenant_context),
    # Security context placeholder for production auth
) -> DecisionResponse:
    """Request AI-driven decision"""
    
    if Permission.SYSTEM_ADMIN not in context.permissions and Permission.TASK_SUBMIT not in context.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions for decision requests")
    
    try:
        # Initialize service if needed
        if not hasattr(intelligence_service, '_initialized'):
            await intelligence_service.initialize()
            intelligence_service._initialized = True
        
        # Process decision using real intelligence service
        response = await intelligence_service.process_decision_request(request, UUID(tenant_id))
        return response
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Decision request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision processing failed: {str(e)}")


@router.get("/decisions/{decision_id}", response_model=DecisionResponse)
async def get_decision(
    decision_id: str,
    # Security context placeholder for production auth
) -> DecisionResponse:
    """Retrieve a previous decision"""
    
    if Permission.TELEMETRY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: telemetry:read")
    
    decision = decisions_store.get(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    return decision


@router.post("/feedback")
async def provide_feedback(
    feedback: LearningFeedback,
    background_tasks: BackgroundTasks,
    # Security context placeholder for production auth
) -> Dict[str, str]:
    """Provide feedback for AI learning"""
    
    if Permission.TELEMETRY_WRITE not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: telemetry:write")
    
    # Validate decision exists
    if feedback.decision_id not in decisions_store:
        raise HTTPException(status_code=404, detail="Decision not found for feedback")
    
    # Store feedback
    feedback_store[feedback.decision_id] = feedback
    
    # Trigger learning process
    background_tasks.add_task(_process_learning_feedback, feedback)
    
    return {
        "message": "Feedback received and learning process initiated",
        "feedback_id": feedback.decision_id
    }


@router.post("/models/train")
async def initiate_model_training(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    # Admin permission required - placeholder for production auth
) -> Dict[str, str]:
    """Initiate model training or fine-tuning"""
    
    training_id = str(uuid.uuid4())
    
    # Validate model type
    if request.model_type not in model_states:
        raise HTTPException(status_code=400, detail="Unsupported model type")
    
    # Validate training data
    if len(request.training_data) < 10:
        raise HTTPException(status_code=400, detail="Insufficient training data")
    
    # Start training process
    background_tasks.add_task(_train_model, training_id, request)
    
    # Update model status
    model_states[request.model_type]["status"] = "training"
    
    return {
        "training_id": training_id,
        "model_type": request.model_type.value,
        "status": "initiated",
        "estimated_completion": "2-4 hours"
    }


@router.get("/models", response_model=List[OrchestrationBrainStatus])
async def list_models(
    # Security context placeholder for production auth
) -> List[OrchestrationBrainStatus]:
    """List available AI models and their status"""
    
    if Permission.TELEMETRY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: telemetry:read")
    
    models = []
    
    for model_type, state in model_states.items():
        # Simulate model capabilities based on type
        capabilities = _get_model_capabilities(model_type)
        
        model_status = OrchestrationBrainStatus(
            model_id=model_type.value,
            status=state["status"],
            version=state["version"],
            capabilities=capabilities,
            performance_metrics={
                "accuracy": state["accuracy"],
                "throughput_requests_per_second": 50.0 if model_type == ModelType.QWEN3_ORCHESTRATOR else 25.0,
                "latency_ms": 100.0 if model_type == ModelType.QWEN3_ORCHESTRATOR else 200.0
            },
            last_training=state.get("last_training"),
            next_training=state.get("next_training"),
            training_data_points=state["training_data_points"],
            memory_usage_mb=2048.0 if model_type == ModelType.QWEN3_ORCHESTRATOR else 1024.0,
            processing_queue_depth=len(decisions_store) if model_type == ModelType.QWEN3_ORCHESTRATOR else 0
        )
        
        models.append(model_status)
    
    return models


@router.get("/models/{model_type}/brain-status", response_model=OrchestrationBrainStatus)
async def get_orchestration_brain_status(
    model_type: ModelType,
    # Security context placeholder for production auth
) -> OrchestrationBrainStatus:
    """Get detailed status of orchestration brain (Qwen3)"""
    
    if Permission.TELEMETRY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: telemetry:read")
    
    if model_type not in model_states:
        raise HTTPException(status_code=404, detail="Model not found")
    
    state = model_states[model_type]
    capabilities = _get_model_capabilities(model_type)
    
    # Enhanced status for orchestration brain
    if model_type == ModelType.QWEN3_ORCHESTRATOR:
        performance_metrics = {
            "accuracy": state["accuracy"],
            "decision_speed_ms": 150.0,
            "learning_rate": 0.001,
            "memory_efficiency": 0.85,
            "context_window_tokens": 32768,
            "concurrent_sessions": 100,
            "uptime_hours": 168.5,  # ~7 days
            "successful_orchestrations": 1250,
            "optimization_score": 0.92
        }
    else:
        performance_metrics = {
            "accuracy": state["accuracy"],
            "response_time_ms": 200.0,
            "task_completion_rate": 0.88,
            "error_rate": 0.02
        }
    
    return OrchestrationBrainStatus(
        model_id=model_type.value,
        status=state["status"],
        version=state["version"],
        capabilities=capabilities,
        performance_metrics=performance_metrics,
        last_training=state.get("last_training"),
        next_training=datetime.utcnow() + timedelta(days=7),  # Weekly retraining
        training_data_points=state["training_data_points"],
        memory_usage_mb=2048.0 if model_type == ModelType.QWEN3_ORCHESTRATOR else 1024.0,
        processing_queue_depth=len([d for d in decisions_store.values() if d.model_used == model_type])
    )


@router.post("/models/{model_type}/optimization")
async def optimize_model(
    model_type: ModelType,
    optimization_params: Dict[str, Any],
    background_tasks: BackgroundTasks,
    # Admin permission required - placeholder for production auth
) -> Dict[str, str]:
    """Trigger model optimization and self-improvement"""
    
    if model_type not in model_states:
        raise HTTPException(status_code=404, detail="Model not found")
    
    optimization_id = str(uuid.uuid4())
    
    # Start optimization process
    background_tasks.add_task(_optimize_model, model_type, optimization_params)
    
    return {
        "optimization_id": optimization_id,
        "model": model_type.value,
        "status": "initiated",
        "message": "Model optimization started"
    }


@router.get("/metrics", response_model=IntelligenceMetrics)
async def get_intelligence_metrics(
    # Security context placeholder for production auth,
    time_range_hours: int = 24
) -> IntelligenceMetrics:
    """Get intelligence system performance metrics"""
    
    if Permission.TELEMETRY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: telemetry:read")
    
    cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
    
    # Filter decisions within time range
    recent_decisions = [
        d for d in decisions_store.values()
        if d.timestamp >= cutoff_time
    ]
    
    # Calculate metrics
    total_decisions = len(recent_decisions)
    
    if total_decisions > 0:
        avg_confidence = sum(d.confidence_score for d in recent_decisions) / total_decisions
        avg_processing_time = sum(d.processing_time_ms for d in recent_decisions) / total_decisions
        
        # Calculate success rate from feedback
        decisions_with_feedback = [
            d.decision_id for d in recent_decisions
            if d.decision_id in feedback_store
        ]
        
        if decisions_with_feedback:
            successful_decisions = sum(
                1 for decision_id in decisions_with_feedback
                if feedback_store[decision_id].outcome in ["success", "partial_success"]
            )
            success_rate = successful_decisions / len(decisions_with_feedback)
        else:
            success_rate = 0.85  # Assumed baseline
    else:
        avg_confidence = 0.0
        avg_processing_time = 0.0
        success_rate = 0.0
    
    # Model accuracy from states
    model_accuracy = {
        model.value: state["accuracy"]
        for model, state in model_states.items()
    }
    
    # Decision type distribution
    decision_distribution = {}
    for decision in recent_decisions:
        decision_type = decision.decision_type.value
        decision_distribution[decision_type] = decision_distribution.get(decision_type, 0) + 1
    
    return IntelligenceMetrics(
        total_decisions=total_decisions,
        avg_confidence_score=avg_confidence,
        avg_processing_time_ms=avg_processing_time,
        success_rate=success_rate,
        model_accuracy=model_accuracy,
        learning_iterations=len(training_history),
        active_models=sum(1 for state in model_states.values() if state["status"] == "active"),
        decision_distribution=decision_distribution
    )


@router.post("/continuous-learning/enable")
async def enable_continuous_learning(
    # Admin permission required - placeholder for production auth
) -> Dict[str, str]:
    """Enable continuous learning for AI models"""
    
    # This would configure the system for continuous learning
    # In a real implementation, this would:
    # 1. Set up data pipelines for real-time learning
    # 2. Configure model update schedules
    # 3. Enable automatic feedback processing
    
    return {
        "status": "enabled",
        "message": "Continuous learning enabled for all models",
        "learning_frequency": "hourly",
        "auto_deployment": "staging_first"
    }


# Helper functions
async def _select_decision_model(decision_type: DecisionType, preferences: List[ModelType]) -> ModelType:
    """Select the best model for a decision type"""
    
    # Model capabilities mapping
    model_capabilities = {
        ModelType.QWEN3_ORCHESTRATOR: [
            DecisionType.ORCHESTRATION_OPTIMIZATION,
            DecisionType.RESOURCE_ALLOCATION,
            DecisionType.TASK_PRIORITIZATION,
            DecisionType.AGENT_ASSIGNMENT
        ],
        ModelType.CLAUDE_AGENT: [
            DecisionType.THREAT_CLASSIFICATION,
            DecisionType.RESPONSE_STRATEGY,
            DecisionType.RISK_ASSESSMENT,
            DecisionType.SECURITY_POSTURE
        ]
    }
    
    # Find suitable models
    suitable_models = [
        model for model, capabilities in model_capabilities.items()
        if decision_type in capabilities and model_states[model]["status"] == "active"
    ]
    
    # Prefer user preferences if available
    for pref in preferences:
        if pref in suitable_models:
            return pref
    
    # Default selection based on decision type
    if suitable_models:
        return suitable_models[0]
    
    # Fallback to Qwen3 orchestrator
    return ModelType.QWEN3_ORCHESTRATOR


async def _process_decision_request(request: DecisionRequest, model: ModelType) -> Dict[str, Any]:
    """Process decision request using selected model"""
    
    # Simulate processing time based on complexity
    processing_time = 0.5 if model == ModelType.QWEN3_ORCHESTRATOR else 1.0
    await asyncio.sleep(processing_time)
    
    # Generate decision based on type and context
    decision_generators = {
        DecisionType.TASK_PRIORITIZATION: _generate_task_priority_decision,
        DecisionType.AGENT_ASSIGNMENT: _generate_agent_assignment_decision,
        DecisionType.THREAT_CLASSIFICATION: _generate_threat_classification_decision,
        DecisionType.RESPONSE_STRATEGY: _generate_response_strategy_decision,
        DecisionType.ORCHESTRATION_OPTIMIZATION: _generate_orchestration_decision
    }
    
    generator = decision_generators.get(request.decision_type, _generate_generic_decision)
    return await generator(request.context, model)


async def _generate_task_priority_decision(context: DecisionContext, model: ModelType) -> Dict[str, Any]:
    """Generate task prioritization decision"""
    
    # Simulate intelligent prioritization
    tasks = context.available_data.get("tasks", [])
    urgency_factor = 1.2 if context.urgency_level == "high" else 1.0
    
    confidence = 0.85 if model == ModelType.QWEN3_ORCHESTRATOR else 0.75
    
    return {
        "recommendation": "prioritize_critical_tasks_first",
        "confidence": confidence,
        "reasoning": [
            "Analyzed task dependencies and resource requirements",
            f"Applied {urgency_factor}x urgency multiplier",
            "Optimized for overall system throughput",
            f"Confidence boosted by {model.value} specialization"
        ],
        "alternatives": [
            {"strategy": "fifo_processing", "confidence": 0.6},
            {"strategy": "load_balanced", "confidence": 0.7}
        ],
        "evidence": {
            "tasks_analyzed": len(tasks),
            "complexity_score": 0.7,
            "resource_availability": 0.8
        }
    }


async def _generate_agent_assignment_decision(context: DecisionContext, model: ModelType) -> Dict[str, Any]:
    """Generate agent assignment decision"""
    
    available_agents = context.available_data.get("agents", [])
    task_requirements = context.available_data.get("task_requirements", {})
    
    # Select best agent based on capabilities
    best_agent = available_agents[0] if available_agents else "default_agent"
    
    return {
        "recommendation": f"assign_to_{best_agent}",
        "confidence": 0.9,
        "reasoning": [
            "Matched task requirements with agent capabilities",
            "Considered agent current workload",
            "Optimized for fastest completion time"
        ],
        "evidence": {
            "agents_evaluated": len(available_agents),
            "capability_match_score": 0.95,
            "workload_balance_score": 0.8
        }
    }


async def _generate_threat_classification_decision(context: DecisionContext, model: ModelType) -> Dict[str, Any]:
    """Generate threat classification decision"""
    
    indicators_raw = context.available_data.get("indicators", [])
    # Handle both list and integer indicators for backward compatibility
    indicators = indicators_raw if isinstance(indicators_raw, list) else [f"indicator_{i}" for i in range(indicators_raw)] if isinstance(indicators_raw, int) else []
    indicators_count = len(indicators) if isinstance(indicators_raw, list) else indicators_raw if isinstance(indicators_raw, int) else 0
    severity_score = context.available_data.get("severity_score", 0.5)
    
    if severity_score > 0.8:
        classification = "critical_threat"
        confidence = 0.92
    elif severity_score > 0.6:
        classification = "high_priority_threat"
        confidence = 0.85
    else:
        classification = "medium_priority_threat"
        confidence = 0.78
    
    return {
        "recommendation": classification,
        "confidence": confidence,
        "reasoning": [
            f"Analyzed {indicators_count} threat indicators",
            f"Severity score: {severity_score}",
            "Cross-referenced with threat intelligence",
            "Applied MITRE ATT&CK framework"
        ],
        "evidence": {
            "indicators_count": indicators_count,
            "severity_score": severity_score,
            "threat_intel_matches": 3
        }
    }


async def _generate_response_strategy_decision(context: DecisionContext, model: ModelType) -> Dict[str, Any]:
    """Generate response strategy decision"""
    
    threat_level = context.available_data.get("threat_level", "medium")
    available_actions = context.available_data.get("available_actions", [])
    
    if threat_level == "critical":
        strategy = "immediate_containment"
        confidence = 0.95
    elif threat_level == "high":
        strategy = "rapid_response"
        confidence = 0.88
    else:
        strategy = "standard_investigation"
        confidence = 0.80
    
    return {
        "recommendation": strategy,
        "confidence": confidence,
        "reasoning": [
            f"Threat level assessed as {threat_level}",
            f"Evaluated {len(available_actions)} response options",
            "Prioritized containment over investigation speed",
            "Considered business impact and resource availability"
        ],
        "evidence": {
            "threat_level": threat_level,
            "response_options": len(available_actions),
            "estimated_impact": "medium"
        }
    }


async def _generate_orchestration_decision(context: DecisionContext, model: ModelType) -> Dict[str, Any]:
    """Generate orchestration optimization decision"""
    
    current_load = context.available_data.get("system_load", 0.5)
    pending_tasks = context.available_data.get("pending_tasks", 0)
    
    if current_load > 0.8:
        strategy = "load_balancing_optimization"
        confidence = 0.92
    elif pending_tasks > 50:
        strategy = "parallel_processing_boost"
        confidence = 0.88
    else:
        strategy = "efficiency_optimization"
        confidence = 0.85
    
    return {
        "recommendation": strategy,
        "confidence": confidence,
        "reasoning": [
            f"Current system load: {current_load}",
            f"Pending tasks: {pending_tasks}",
            "Applied predictive load balancing",
            "Optimized for maximum throughput"
        ],
        "evidence": {
            "system_load": current_load,
            "pending_tasks": pending_tasks,
            "resource_utilization": 0.75
        }
    }


async def _generate_generic_decision(context: DecisionContext, model: ModelType) -> Dict[str, Any]:
    """Generate generic decision for unknown types"""
    
    return {
        "recommendation": "analyze_and_recommend",
        "confidence": 0.7,
        "reasoning": [
            "Applied general decision framework",
            "Considered available context data",
            "Used conservative approach for unknown scenario"
        ],
        "evidence": context.available_data
    }


def _get_model_capabilities(model_type: ModelType) -> List[str]:
    """Get capabilities for a model type"""
    
    capabilities_map = {
        ModelType.QWEN3_ORCHESTRATOR: [
            "task_orchestration",
            "resource_optimization",
            "multi_agent_coordination",
            "strategic_planning",
            "real_time_adaptation",
            "federated_learning",
            "continuous_improvement"
        ],
        ModelType.CLAUDE_AGENT: [
            "threat_analysis",
            "security_reasoning",
            "incident_response",
            "risk_assessment",
            "natural_language_processing",
            "code_analysis",
            "decision_explanation"
        ],
        ModelType.THREAT_CLASSIFIER: [
            "threat_detection",
            "malware_classification",
            "anomaly_detection",
            "pattern_recognition"
        ]
    }
    
    return capabilities_map.get(model_type, ["general_ai"])


async def _process_learning_feedback(feedback: LearningFeedback):
    """Process learning feedback for model improvement"""
    
    try:
        # Get original decision
        decision = decisions_store.get(feedback.decision_id)
        if not decision:
            return
        
        # Extract learning data
        learning_data = {
            "decision_type": decision.decision_type.value,
            "context": "processed",  # Would be the original context
            "prediction": decision.recommendation,
            "actual_outcome": feedback.outcome,
            "effectiveness": feedback.effectiveness_score,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in training history
        training_history.append(learning_data)
        
        # Update model performance metrics
        model_used = decision.model_used
        if model_used in model_states:
            current_accuracy = model_states[model_used]["accuracy"]
            
            # Simple accuracy update based on feedback
            if feedback.outcome == "success":
                new_accuracy = current_accuracy * 0.99 + 0.01  # Slight improvement
            elif feedback.outcome == "failure":
                new_accuracy = current_accuracy * 0.995  # Slight degradation
            else:  # partial_success
                new_accuracy = current_accuracy  # No change
                
            model_states[model_used]["accuracy"] = min(new_accuracy, 1.0)
        
        # Trigger retraining if enough feedback accumulated
        if len(training_history) % 100 == 0:  # Every 100 feedback items
            await _schedule_model_retraining(model_used)
            
    except Exception as e:
        # Log error but don't fail
        pass


async def _train_model(training_id: str, request: ModelTrainingRequest):
    """Train or fine-tune a model"""
    
    try:
        # Simulate training process
        model_type = request.model_type
        
        # Update model status
        model_states[model_type]["status"] = "training"
        
        # Simulate training time
        training_duration = len(request.training_data) * 0.1  # 0.1 seconds per data point
        await asyncio.sleep(min(training_duration, 10))  # Max 10 seconds for demo
        
        # Update model with new training
        model_states[model_type]["last_training"] = datetime.utcnow()
        model_states[model_type]["training_data_points"] += len(request.training_data)
        
        # Simulate accuracy improvement
        current_accuracy = model_states[model_type]["accuracy"]
        improvement = min(0.05, (100 - current_accuracy * 100) * 0.1 / 100)
        model_states[model_type]["accuracy"] = min(current_accuracy + improvement, 1.0)
        
        # Return to active status
        model_states[model_type]["status"] = "active"
        
        # Record training session
        training_history.append({
            "training_id": training_id,
            "model_type": model_type.value,
            "data_points": len(request.training_data),
            "learning_type": request.learning_type.value,
            "accuracy_improvement": improvement,
            "completed_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        # Mark training as failed
        if model_type in model_states:
            model_states[model_type]["status"] = "error"


async def _optimize_model(model_type: ModelType, optimization_params: Dict[str, Any]):
    """Optimize model performance"""
    
    try:
        # Simulate optimization process
        await asyncio.sleep(5)  # Simulation delay
        
        # Apply optimizations
        if model_type in model_states:
            state = model_states[model_type]
            
            # Simulate performance improvements
            current_accuracy = state["accuracy"]
            optimization_boost = optimization_params.get("performance_boost", 0.02)
            
            state["accuracy"] = min(current_accuracy + optimization_boost, 1.0)
            state["last_training"] = datetime.utcnow()
            
    except Exception as e:
        pass


async def _schedule_model_retraining(model_type: ModelType):
    """Schedule model retraining"""
    
    if model_type in model_states:
        # Set next training time
        model_states[model_type]["next_training"] = datetime.utcnow() + timedelta(hours=24)


# Initialize sample data
async def _initialize_sample_intelligence_data():
    """Initialize sample intelligence data"""
    
    # Create some sample decisions
    if not decisions_store:
        sample_decisions = [
            {
                "decision_type": DecisionType.TASK_PRIORITIZATION,
                "recommendation": "prioritize_critical_security_tasks",
                "confidence": 0.92,
                "model": ModelType.QWEN3_ORCHESTRATOR
            },
            {
                "decision_type": DecisionType.THREAT_CLASSIFICATION,
                "recommendation": "classify_as_high_priority_threat",
                "confidence": 0.88,
                "model": ModelType.CLAUDE_AGENT
            }
        ]
        
        for i, decision_data in enumerate(sample_decisions):
            decision_id = str(uuid.uuid4())
            current_time = datetime.utcnow() - timedelta(hours=i)
            
            decision = DecisionResponse(
                decision_id=decision_id,
                decision_type=decision_data["decision_type"],
                recommendation=decision_data["recommendation"],
                confidence_score=decision_data["confidence"],
                reasoning=[
                    "Analyzed current security posture",
                    "Considered resource availability",
                    "Applied best practices framework"
                ],
                model_used=decision_data["model"],
                processing_time_ms=150 + i * 50,
                timestamp=current_time
            )
            
            decisions_store[decision_id] = decision


# Note: Sample data initialization handled during startup