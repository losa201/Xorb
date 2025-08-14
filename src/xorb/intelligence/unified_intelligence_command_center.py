#!/usr/bin/env python3
"""
Unified Intelligence Command Center - Strategic Enhancement
Principal Auditor Implementation: Next-Generation Autonomous Cybersecurity Orchestration

This module implements the central intelligence orchestration layer that coordinates
all autonomous capabilities into a unified, enterprise-grade cybersecurity platform.

Key Features:
- Unified AI command and control center
- Real-time intelligence fusion and correlation
- Autonomous decision orchestration across all platform components
- Enterprise-grade mission planning and execution
- Advanced threat scenario modeling and simulation
- Quantum-safe intelligence operations
- Global threat intelligence integration
"""

import asyncio
import logging
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import secrets
from concurrent.futures import ThreadPoolExecutor
import structlog

# Advanced ML and AI imports
try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

# Internal XORB component imports
from .advanced_threat_prediction_engine import get_threat_prediction_engine, ThreatPredictionType, PredictionHorizon
from ..security.autonomous_red_team_engine import get_autonomous_red_team_engine, OperationObjective, SecurityConstraints
from ..exploitation.advanced_payload_engine import get_payload_engine, PayloadConfiguration, PayloadType
from ..common.security_framework import SecurityFramework, SecurityLevel
from ..common.audit_logger import AuditLogger, AuditEvent

# Configure structured logging
logger = structlog.get_logger(__name__)


class MissionPriority(Enum):
    """Mission priority levels for intelligence operations"""
    CRITICAL = "critical"       # Immediate threat response
    HIGH = "high"              # Important security operations
    MEDIUM = "medium"          # Standard security activities
    LOW = "low"                # Background monitoring
    MAINTENANCE = "maintenance" # System maintenance tasks


class IntelligenceSource(Enum):
    """Sources of intelligence for fusion"""
    THREAT_PREDICTION = "threat_prediction"
    AUTONOMOUS_RED_TEAM = "autonomous_red_team"
    PAYLOAD_INTELLIGENCE = "payload_intelligence"
    PTAAS_SCANNING = "ptaas_scanning"
    EXTERNAL_FEEDS = "external_feeds"
    BEHAVIORAL_ANALYTICS = "behavioral_analytics"
    NETWORK_MONITORING = "network_monitoring"
    GLOBAL_INTELLIGENCE = "global_intelligence"


class OperationStatus(Enum):
    """Status of intelligence operations"""
    PLANNED = "planned"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


@dataclass
class IntelligenceAsset:
    """Unified intelligence asset representation"""
    asset_id: str
    asset_type: str
    source: IntelligenceSource
    data: Dict[str, Any]
    confidence_score: float
    timestamp: datetime
    expires_at: Optional[datetime] = None
    correlated_assets: List[str] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedMission:
    """Unified mission encompassing multiple autonomous capabilities"""
    mission_id: str
    name: str
    description: str
    priority: MissionPriority
    objectives: List[str]
    
    # Component assignments
    threat_prediction_tasks: List[Dict[str, Any]] = field(default_factory=list)
    red_team_operations: List[Dict[str, Any]] = field(default_factory=list)
    payload_requirements: List[Dict[str, Any]] = field(default_factory=list)
    ptaas_scans: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution metadata
    status: OperationStatus = OperationStatus.PLANNED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    # Intelligence coordination
    required_assets: List[str] = field(default_factory=list)
    generated_assets: List[str] = field(default_factory=list)
    correlation_requirements: List[str] = field(default_factory=list)
    
    # Success criteria and metrics
    success_criteria: List[str] = field(default_factory=list)
    progress_metrics: Dict[str, float] = field(default_factory=dict)
    performance_data: Dict[str, Any] = field(default_factory=dict)
    
    # Safety and compliance
    safety_level: str = "high"
    compliance_requirements: List[str] = field(default_factory=list)
    human_oversight_required: bool = True
    
    # Results and lessons learned
    results: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class IntelligenceFusionResult:
    """Result of intelligence fusion and correlation"""
    fusion_id: str
    source_assets: List[str]
    fused_intelligence: Dict[str, Any]
    confidence_score: float
    correlation_strength: float
    threat_level: str
    recommended_actions: List[str]
    fusion_timestamp: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedIntelligenceFusionEngine:
    """Advanced intelligence fusion and correlation engine"""
    
    def __init__(self):
        self.fusion_algorithms = {
            "weighted_consensus": self._weighted_consensus_fusion,
            "bayesian_fusion": self._bayesian_fusion,
            "deep_correlation": self._deep_correlation_fusion,
            "temporal_fusion": self._temporal_fusion,
            "multi_modal_fusion": self._multi_modal_fusion
        }
        
        self.correlation_matrix = np.eye(10)  # Initialize correlation matrix
        self.fusion_history: List[IntelligenceFusionResult] = []
        
        # ML-based fusion models
        if ADVANCED_ML_AVAILABLE:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.correlation_model = self._initialize_correlation_model()
    
    def _initialize_correlation_model(self) -> Optional[nn.Module]:
        """Initialize neural network for intelligence correlation"""
        if not ADVANCED_ML_AVAILABLE:
            return None
        
        try:
            class IntelligenceCorrelationNetwork(nn.Module):
                def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, 32)
                    )
                    
                    self.correlation_head = nn.Sequential(
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    correlation = self.correlation_head(encoded)
                    return encoded, correlation
            
            return IntelligenceCorrelationNetwork()
        except Exception as e:
            logger.error(f"Failed to initialize correlation model: {e}")
            return None
    
    async def fuse_intelligence(self, assets: List[IntelligenceAsset], 
                              fusion_method: str = "weighted_consensus") -> IntelligenceFusionResult:
        """Fuse multiple intelligence assets into unified intelligence"""
        try:
            fusion_id = str(uuid.uuid4())
            
            # Select fusion algorithm
            fusion_algorithm = self.fusion_algorithms.get(fusion_method, self._weighted_consensus_fusion)
            
            # Perform fusion
            fused_data, confidence = await fusion_algorithm(assets)
            
            # Calculate correlation strength
            correlation_strength = await self._calculate_correlation_strength(assets)
            
            # Assess threat level
            threat_level = await self._assess_threat_level(fused_data, confidence)
            
            # Generate recommendations
            recommendations = await self._generate_fusion_recommendations(
                fused_data, threat_level, correlation_strength
            )
            
            # Create fusion result
            fusion_result = IntelligenceFusionResult(
                fusion_id=fusion_id,
                source_assets=[asset.asset_id for asset in assets],
                fused_intelligence=fused_data,
                confidence_score=confidence,
                correlation_strength=correlation_strength,
                threat_level=threat_level,
                recommended_actions=recommendations,
                fusion_timestamp=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            
            # Store fusion history
            self.fusion_history.append(fusion_result)
            
            logger.info(f"Intelligence fusion completed",
                       fusion_id=fusion_id,
                       assets_count=len(assets),
                       confidence=confidence,
                       threat_level=threat_level)
            
            return fusion_result
            
        except Exception as e:
            logger.error(f"Intelligence fusion failed: {e}")
            raise
    
    async def _weighted_consensus_fusion(self, assets: List[IntelligenceAsset]) -> Tuple[Dict[str, Any], float]:
        """Weighted consensus fusion algorithm"""
        try:
            if not assets:
                return {}, 0.0
            
            # Calculate weights based on confidence and source reliability
            weights = []
            total_weight = 0.0
            
            for asset in assets:
                # Base weight on confidence score
                weight = asset.confidence_score
                
                # Adjust weight based on source type
                source_weights = {
                    IntelligenceSource.THREAT_PREDICTION: 1.2,
                    IntelligenceSource.AUTONOMOUS_RED_TEAM: 1.1,
                    IntelligenceSource.PTAAS_SCANNING: 1.0,
                    IntelligenceSource.EXTERNAL_FEEDS: 0.8,
                    IntelligenceSource.BEHAVIORAL_ANALYTICS: 0.9
                }
                
                weight *= source_weights.get(asset.source, 1.0)
                
                # Adjust for recency
                age_hours = (datetime.utcnow() - asset.timestamp).total_seconds() / 3600
                recency_factor = max(0.5, 1.0 - (age_hours * 0.1))  # Decay over time
                weight *= recency_factor
                
                weights.append(weight)
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
            else:
                normalized_weights = [1.0 / len(assets)] * len(assets)
            
            # Fuse data using weighted average
            fused_data = {}
            overall_confidence = 0.0
            
            # Collect all data keys
            all_keys = set()
            for asset in assets:
                all_keys.update(asset.data.keys())
            
            # Fuse each data field
            for key in all_keys:
                values = []
                key_weights = []
                
                for i, asset in enumerate(assets):
                    if key in asset.data:
                        values.append(asset.data[key])
                        key_weights.append(normalized_weights[i])
                
                if values:
                    # Handle different data types
                    if all(isinstance(v, (int, float)) for v in values):
                        # Numeric weighted average
                        weighted_sum = sum(v * w for v, w in zip(values, key_weights))
                        weight_sum = sum(key_weights)
                        fused_data[key] = weighted_sum / weight_sum if weight_sum > 0 else 0
                    elif all(isinstance(v, bool) for v in values):
                        # Boolean majority vote with weights
                        true_weight = sum(w for v, w in zip(values, key_weights) if v)
                        total_weight = sum(key_weights)
                        fused_data[key] = true_weight > (total_weight / 2)
                    else:
                        # String/object: most confident source
                        max_idx = key_weights.index(max(key_weights))
                        fused_data[key] = values[max_idx]
            
            # Calculate overall confidence
            overall_confidence = sum(conf * w for conf, w in zip([asset.confidence_score for asset in assets], normalized_weights))
            
            return fused_data, overall_confidence
            
        except Exception as e:
            logger.error(f"Weighted consensus fusion failed: {e}")
            return {}, 0.0
    
    async def _calculate_correlation_strength(self, assets: List[IntelligenceAsset]) -> float:
        """Calculate correlation strength between intelligence assets"""
        try:
            if len(assets) < 2:
                return 1.0
            
            correlations = []
            
            # Calculate pairwise correlations
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    
                    # Time correlation
                    time_diff = abs((asset1.timestamp - asset2.timestamp).total_seconds())
                    time_correlation = max(0, 1.0 - (time_diff / 3600))  # Decay over 1 hour
                    
                    # Source correlation
                    source_correlation = 0.8 if asset1.source == asset2.source else 0.6
                    
                    # Data correlation (simplified)
                    common_keys = set(asset1.data.keys()) & set(asset2.data.keys())
                    data_correlation = len(common_keys) / max(len(asset1.data), len(asset2.data), 1)
                    
                    # Combined correlation
                    combined = (time_correlation + source_correlation + data_correlation) / 3
                    correlations.append(combined)
            
            return np.mean(correlations) if correlations else 0.5
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return 0.5


class UnifiedIntelligenceCommandCenter:
    """
    Unified Intelligence Command Center
    
    Central orchestration layer that coordinates all autonomous capabilities
    into a unified, enterprise-grade cybersecurity platform.
    
    Features:
    - Unified mission planning and execution
    - Real-time intelligence fusion and correlation
    - Autonomous decision orchestration
    - Advanced threat scenario modeling
    - Enterprise-grade reporting and analytics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.command_center_id = str(uuid.uuid4())
        
        # Core component interfaces
        self.threat_prediction_engine = None
        self.autonomous_red_team_engine = None
        self.payload_engine = None
        self.ptaas_orchestrator = None
        
        # Intelligence management
        self.fusion_engine = AdvancedIntelligenceFusionEngine()
        self.intelligence_assets: Dict[str, IntelligenceAsset] = {}
        self.correlation_database: Dict[str, List[str]] = {}
        
        # Mission management
        self.active_missions: Dict[str, UnifiedMission] = {}
        self.mission_history: List[UnifiedMission] = []
        self.mission_queue: List[str] = []
        
        # Coordination and orchestration
        self.decision_pipeline: List[Callable] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.coordination_rules: Dict[str, Dict[str, Any]] = {}
        
        # Performance and analytics
        self.performance_metrics = {
            "missions_completed": 0,
            "successful_missions": 0,
            "intelligence_assets_processed": 0,
            "fusion_operations": 0,
            "autonomous_decisions": 0,
            "coordination_events": 0
        }
        
        # Safety and compliance
        self.security_framework = SecurityFramework()
        self.audit_logger = AuditLogger()
        self.safety_constraints = {
            "max_concurrent_missions": 10,
            "human_oversight_required": True,
            "safety_validation_required": True,
            "compliance_enforcement": True
        }
        
        # Executor for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("Unified Intelligence Command Center initialized",
                   command_center_id=self.command_center_id)
    
    async def initialize(self) -> bool:
        """Initialize the command center with all component integrations"""
        try:
            logger.info("Initializing Unified Intelligence Command Center")
            
            # Initialize core security components
            await self.security_framework.initialize()
            await self.audit_logger.initialize()
            
            # Initialize XORB component engines
            await self._initialize_component_engines()
            
            # Setup intelligence coordination
            await self._setup_intelligence_coordination()
            
            # Initialize decision pipeline
            await self._initialize_decision_pipeline()
            
            # Setup event handling
            await self._setup_event_handling()
            
            # Load coordination rules
            await self._load_coordination_rules()
            
            # Start monitoring and metrics collection
            await self._start_monitoring()
            
            logger.info("Unified Intelligence Command Center fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Command center initialization failed: {e}")
            return False
    
    async def _initialize_component_engines(self):
        """Initialize all XORB component engines"""
        try:
            # Initialize threat prediction engine
            self.threat_prediction_engine = await get_threat_prediction_engine(
                self.config.get("threat_prediction", {})
            )
            
            # Initialize autonomous red team engine
            self.autonomous_red_team_engine = await get_autonomous_red_team_engine(
                self.config.get("autonomous_red_team", {})
            )
            
            # Initialize payload engine
            self.payload_engine = await get_payload_engine(
                self.config.get("payload_engine", {})
            )
            
            logger.info("All component engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Component engine initialization failed: {e}")
            raise
    
    async def plan_unified_mission(self, mission_spec: Dict[str, Any]) -> UnifiedMission:
        """
        Plan a unified mission that coordinates multiple autonomous capabilities
        
        Args:
            mission_spec: Mission specification with objectives and requirements
            
        Returns:
            Comprehensive unified mission plan
        """
        mission_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Planning unified mission {mission_id}")
            
            # Parse mission specification
            mission_name = mission_spec.get("name", f"Mission_{mission_id[:8]}")
            mission_description = mission_spec.get("description", "Unified cybersecurity mission")
            priority = MissionPriority(mission_spec.get("priority", "medium"))
            objectives = mission_spec.get("objectives", [])
            
            # Analyze mission requirements
            threat_analysis_required = mission_spec.get("threat_analysis", True)
            red_team_operations = mission_spec.get("red_team_operations", [])
            payload_requirements = mission_spec.get("payload_requirements", [])
            ptaas_scans = mission_spec.get("ptaas_scans", [])
            
            # Create mission plan
            mission = UnifiedMission(
                mission_id=mission_id,
                name=mission_name,
                description=mission_description,
                priority=priority,
                objectives=objectives,
                safety_level=mission_spec.get("safety_level", "high"),
                compliance_requirements=mission_spec.get("compliance_requirements", []),
                human_oversight_required=mission_spec.get("human_oversight_required", True)
            )
            
            # Plan threat prediction tasks
            if threat_analysis_required:
                await self._plan_threat_prediction_tasks(mission, mission_spec)
            
            # Plan red team operations
            if red_team_operations:
                await self._plan_red_team_operations(mission, red_team_operations)
            
            # Plan payload requirements
            if payload_requirements:
                await self._plan_payload_requirements(mission, payload_requirements)
            
            # Plan PTaaS scans
            if ptaas_scans:
                await self._plan_ptaas_scans(mission, ptaas_scans)
            
            # Calculate mission timeline and dependencies
            await self._calculate_mission_timeline(mission)
            await self._analyze_mission_dependencies(mission)
            
            # Validate mission safety and compliance
            await self._validate_mission_safety(mission)
            
            # Store mission
            self.active_missions[mission_id] = mission
            self.mission_queue.append(mission_id)
            
            # Log mission planning
            await self.audit_logger.log_event(AuditEvent(
                event_type="unified_mission_planned",
                component="unified_intelligence_command_center",
                details={
                    "mission_id": mission_id,
                    "mission_name": mission_name,
                    "priority": priority.value,
                    "objectives_count": len(objectives),
                    "components_involved": self._get_involved_components(mission)
                },
                security_level=SecurityLevel.HIGH
            ))
            
            logger.info(f"Unified mission {mission_id} planned successfully")
            
            return mission
            
        except Exception as e:
            logger.error(f"Mission planning failed: {e}")
            raise
    
    async def execute_unified_mission(self, mission_id: str) -> Dict[str, Any]:
        """
        Execute a unified mission with coordinated autonomous capabilities
        
        Args:
            mission_id: Unique identifier for the planned mission
            
        Returns:
            Comprehensive mission execution results
        """
        try:
            if mission_id not in self.active_missions:
                raise ValueError(f"Mission {mission_id} not found")
            
            mission = self.active_missions[mission_id]
            
            logger.info(f"Executing unified mission {mission_id}")
            
            # Update mission status
            mission.status = OperationStatus.EXECUTING
            mission.started_at = datetime.utcnow()
            
            # Initialize execution context
            execution_context = {
                "mission_id": mission_id,
                "start_time": datetime.utcnow(),
                "component_results": {},
                "intelligence_generated": [],
                "decisions_made": [],
                "coordination_events": []
            }
            
            # Execute mission components in coordinated fashion
            try:
                # Phase 1: Threat prediction and intelligence gathering
                if mission.threat_prediction_tasks:
                    await self._execute_threat_prediction_phase(mission, execution_context)
                
                # Phase 2: Red team operations (if authorized)
                if mission.red_team_operations:
                    await self._execute_red_team_phase(mission, execution_context)
                
                # Phase 3: Payload operations (if required)
                if mission.payload_requirements:
                    await self._execute_payload_phase(mission, execution_context)
                
                # Phase 4: PTaaS scanning operations
                if mission.ptaas_scans:
                    await self._execute_ptaas_phase(mission, execution_context)
                
                # Phase 5: Intelligence fusion and analysis
                await self._execute_intelligence_fusion_phase(mission, execution_context)
                
                # Mission completion
                mission.status = OperationStatus.COMPLETED
                mission.completed_at = datetime.utcnow()
                
                # Generate comprehensive results
                mission_results = await self._generate_mission_results(mission, execution_context)
                mission.results = mission_results
                
                # Update metrics
                self.performance_metrics["missions_completed"] += 1
                self.performance_metrics["successful_missions"] += 1
                
                logger.info(f"Unified mission {mission_id} completed successfully")
                
                return mission_results
                
            except Exception as e:
                mission.status = OperationStatus.FAILED
                mission.completed_at = datetime.utcnow()
                logger.error(f"Mission execution failed: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Unified mission execution failed: {e}")
            raise
    
    async def _execute_threat_prediction_phase(self, mission: UnifiedMission, context: Dict[str, Any]):
        """Execute threat prediction phase of unified mission"""
        try:
            logger.info("Executing threat prediction phase")
            
            threat_results = []
            
            for task in mission.threat_prediction_tasks:
                prediction_type = ThreatPredictionType(task["prediction_type"])
                prediction_horizon = PredictionHorizon(task["prediction_horizon"])
                task_context = task.get("context", {})
                
                # Generate threat prediction
                prediction = await self.threat_prediction_engine.predict_threat(
                    prediction_type=prediction_type,
                    prediction_horizon=prediction_horizon,
                    context=task_context
                )
                
                threat_results.append({
                    "task_id": task.get("task_id", str(uuid.uuid4())),
                    "prediction": prediction,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Create intelligence asset
                asset = IntelligenceAsset(
                    asset_id=str(uuid.uuid4()),
                    asset_type="threat_prediction",
                    source=IntelligenceSource.THREAT_PREDICTION,
                    data={
                        "prediction_type": prediction_type.value,
                        "predicted_value": prediction.predicted_value,
                        "confidence_score": prediction.confidence_score,
                        "risk_level": prediction.risk_level,
                        "recommendations": prediction.recommended_actions
                    },
                    confidence_score=prediction.confidence_score,
                    timestamp=datetime.utcnow(),
                    expires_at=prediction.valid_until
                )
                
                self.intelligence_assets[asset.asset_id] = asset
                context["intelligence_generated"].append(asset.asset_id)
            
            context["component_results"]["threat_prediction"] = threat_results
            
        except Exception as e:
            logger.error(f"Threat prediction phase failed: {e}")
            raise
    
    async def get_command_center_metrics(self) -> Dict[str, Any]:
        """Get comprehensive command center metrics and performance data"""
        try:
            # Calculate success rates
            total_missions = self.performance_metrics["missions_completed"]
            success_rate = (self.performance_metrics["successful_missions"] / total_missions 
                          if total_missions > 0 else 0.0)
            
            # Mission status distribution
            mission_status_dist = {}
            for mission in self.active_missions.values():
                status = mission.status.value
                mission_status_dist[status] = mission_status_dist.get(status, 0) + 1
            
            # Intelligence asset metrics
            asset_type_dist = {}
            asset_source_dist = {}
            for asset in self.intelligence_assets.values():
                asset_type_dist[asset.asset_type] = asset_type_dist.get(asset.asset_type, 0) + 1
                source_key = asset.source.value
                asset_source_dist[source_key] = asset_source_dist.get(source_key, 0) + 1
            
            return {
                "command_center_metrics": {
                    "command_center_id": self.command_center_id,
                    "total_missions": total_missions,
                    "successful_missions": self.performance_metrics["successful_missions"],
                    "success_rate": success_rate,
                    "active_missions": len(self.active_missions),
                    "queued_missions": len(self.mission_queue),
                    "intelligence_assets": len(self.intelligence_assets),
                    "fusion_operations": self.performance_metrics["fusion_operations"],
                    "autonomous_decisions": self.performance_metrics["autonomous_decisions"]
                },
                "mission_distribution": {
                    "status_distribution": mission_status_dist,
                    "priority_distribution": self._calculate_priority_distribution()
                },
                "intelligence_metrics": {
                    "asset_type_distribution": asset_type_dist,
                    "asset_source_distribution": asset_source_dist,
                    "fusion_history": len(self.fusion_engine.fusion_history),
                    "correlation_database_size": len(self.correlation_database)
                },
                "component_availability": {
                    "threat_prediction_engine": self.threat_prediction_engine is not None,
                    "autonomous_red_team_engine": self.autonomous_red_team_engine is not None,
                    "payload_engine": self.payload_engine is not None,
                    "advanced_ml_available": ADVANCED_ML_AVAILABLE
                },
                "performance_indicators": {
                    "average_mission_duration": await self._calculate_average_mission_duration(),
                    "intelligence_fusion_rate": await self._calculate_fusion_rate(),
                    "decision_accuracy": await self._calculate_decision_accuracy(),
                    "coordination_efficiency": await self._calculate_coordination_efficiency()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get command center metrics: {e}")
            return {"error": str(e)}


# Global command center instance
_command_center: Optional[UnifiedIntelligenceCommandCenter] = None


async def get_unified_intelligence_command_center(config: Dict[str, Any] = None) -> UnifiedIntelligenceCommandCenter:
    """Get singleton unified intelligence command center instance"""
    global _command_center
    
    if _command_center is None:
        _command_center = UnifiedIntelligenceCommandCenter(config)
        await _command_center.initialize()
    
    return _command_center


# Export main classes
__all__ = [
    "UnifiedIntelligenceCommandCenter",
    "UnifiedMission",
    "IntelligenceAsset",
    "IntelligenceFusionResult",
    "AdvancedIntelligenceFusionEngine",
    "MissionPriority",
    "IntelligenceSource",
    "OperationStatus",
    "get_unified_intelligence_command_center"
]