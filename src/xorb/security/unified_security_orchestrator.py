"""
XORB Unified Security Orchestrator
Master orchestration system integrating red vs blue vs purple team operations
with existing PTaaS, threat intelligence, and behavioral analytics capabilities.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

# Import XORB core components
from .team_orchestration_framework import TeamOrchestrationFramework, get_team_orchestration_framework
from ..intelligence.ml_tactical_coordinator import MLTacticalCoordinator, get_ml_tactical_coordinator
from ..intelligence.threat_intelligence_engine import ThreatIntelligenceEngine
from ...ptaas.behavioral_analytics import get_behavioral_analytics_engine
from ...ptaas.threat_hunting_engine import ThreatHuntingEngine

# Import existing PTaaS services
try:
    from ...api.app.services.ptaas_scanner_service import get_scanner_service
    from ...api.app.services.intelligence_service import get_intelligence_service
    PTAAS_SERVICES_AVAILABLE = True
except ImportError:
    PTAAS_SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)

class OrchestrationMode(Enum):
    """Modes of security orchestration"""
    AUTONOMOUS = "autonomous"
    SEMI_AUTOMATED = "semi_automated"
    MANUAL = "manual"
    HYBRID = "hybrid"

class SecurityOperation(Enum):
    """Types of unified security operations"""
    COMPREHENSIVE_ASSESSMENT = "comprehensive_assessment"
    PURPLE_TEAM_EXERCISE = "purple_team_exercise"
    THREAT_SIMULATION = "threat_simulation"
    INCIDENT_RESPONSE_DRILL = "incident_response_drill"
    CONTINUOUS_MONITORING = "continuous_monitoring"
    ADAPTIVE_DEFENSE = "adaptive_defense"

@dataclass
class UnifiedOperationPlan:
    """Comprehensive unified operation plan"""
    operation_id: str
    operation_type: SecurityOperation
    orchestration_mode: OrchestrationMode
    
    # Team operation components
    team_operation_plan_id: Optional[str]
    
    # PTaaS components
    ptaas_session_ids: List[str]
    scan_profiles: List[str]
    
    # Intelligence components
    threat_hunting_queries: List[str]
    behavioral_monitoring: bool
    
    # ML coordination
    tactical_strategies: List[str]
    adaptive_mechanisms: List[str]
    
    # Orchestration details
    execution_timeline: Dict[str, Any]
    success_criteria: Dict[str, float]
    escalation_procedures: List[str]
    
    # Status tracking
    current_phase: str
    overall_progress: float
    component_status: Dict[str, str]
    
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

class UnifiedSecurityOrchestrator:
    """Master orchestrator for unified security operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.team_framework: Optional[TeamOrchestrationFramework] = None
        self.ml_coordinator: Optional[MLTacticalCoordinator] = None
        self.threat_hunting_engine: Optional[ThreatHuntingEngine] = None
        self.behavioral_analytics = None
        
        # PTaaS components
        self.scanner_service = None
        self.intelligence_service = None
        
        # Operation management
        self.active_operations: Dict[str, UnifiedOperationPlan] = {}
        self.operation_history: List[UnifiedOperationPlan] = []
        self.integration_metrics: Dict[str, Any] = {}
        
        # Real-time coordination
        self.coordination_channels: Dict[str, Any] = {}
        self.cross_component_events: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize the unified security orchestrator"""
        try:
            logger.info("Initializing Unified Security Orchestrator...")
            
            # Initialize core team framework
            self.team_framework = await get_team_orchestration_framework()
            
            # Initialize ML tactical coordinator
            self.ml_coordinator = await get_ml_tactical_coordinator()
            
            # Initialize behavioral analytics
            self.behavioral_analytics = await get_behavioral_analytics_engine()
            
            # Initialize PTaaS services if available
            if PTAAS_SERVICES_AVAILABLE:
                try:
                    self.scanner_service = await get_scanner_service()
                    self.intelligence_service = await get_intelligence_service()
                    logger.info("PTaaS services integrated successfully")
                except Exception as e:
                    logger.warning(f"PTaaS services integration failed: {e}")
            
            # Setup cross-component communication
            await self._setup_cross_component_communication()
            
            # Initialize integration metrics
            await self._initialize_integration_metrics()
            
            logger.info("Unified Security Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Unified Security Orchestrator: {e}")
            raise
    
    async def _setup_cross_component_communication(self):
        """Setup communication channels between components"""
        self.coordination_channels = {
            "team_operations": {
                "type": "bidirectional",
                "component": "team_framework",
                "event_types": ["operation_start", "phase_change", "completion"],
                "handlers": []
            },
            "ptaas_scanning": {
                "type": "event_driven", 
                "component": "scanner_service",
                "event_types": ["scan_complete", "vulnerability_found", "threat_detected"],
                "handlers": []
            },
            "threat_intelligence": {
                "type": "continuous",
                "component": "intelligence_service",
                "event_types": ["threat_update", "ioc_detected", "risk_change"],
                "handlers": []
            },
            "behavioral_analytics": {
                "type": "continuous",
                "component": "behavioral_analytics",
                "event_types": ["anomaly_detected", "pattern_learned", "risk_escalation"],
                "handlers": []
            },
            "ml_coordination": {
                "type": "adaptive",
                "component": "ml_coordinator",
                "event_types": ["strategy_update", "tactical_adjustment", "prediction_update"],
                "handlers": []
            }
        }
        
        logger.info("Cross-component communication channels established")
    
    async def _initialize_integration_metrics(self):
        """Initialize metrics for component integration"""
        self.integration_metrics = {
            "operations_executed": 0,
            "components_integrated": len([c for c in [
                self.team_framework, self.ml_coordinator, self.behavioral_analytics,
                self.scanner_service, self.intelligence_service
            ] if c is not None]),
            "cross_component_events": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "average_operation_time": 0.0,
            "component_availability": {},
            "last_health_check": datetime.now().isoformat()
        }
    
    async def create_unified_operation(self, 
                                     operation_type: SecurityOperation,
                                     configuration: Dict[str, Any]) -> str:
        """Create a comprehensive unified security operation"""
        try:
            operation_id = f"unified_{operation_type.value}_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Creating unified operation {operation_id}")
            
            # Create unified operation plan
            operation_plan = await self._create_unified_plan(
                operation_id, operation_type, configuration
            )
            
            # Store operation
            self.active_operations[operation_id] = operation_plan
            
            # Update metrics
            self.integration_metrics["operations_executed"] += 1
            
            logger.info(f"Created unified operation {operation_id}")
            return operation_id
            
        except Exception as e:
            logger.error(f"Failed to create unified operation: {e}")
            raise
    
    async def _create_unified_plan(self, 
                                 operation_id: str,
                                 operation_type: SecurityOperation,
                                 configuration: Dict[str, Any]) -> UnifiedOperationPlan:
        """Create comprehensive unified operation plan"""
        
        # Determine orchestration mode
        orchestration_mode = OrchestrationMode(configuration.get("mode", "hybrid"))
        
        # Initialize plan components
        team_operation_plan_id = None
        ptaas_session_ids = []
        threat_hunting_queries = []
        tactical_strategies = []
        
        # Configure based on operation type
        if operation_type == SecurityOperation.PURPLE_TEAM_EXERCISE:
            # Create team operation plan
            if self.team_framework:
                scenario_data = await self._create_purple_team_scenario(configuration)
                scenario_id = await self._register_scenario(scenario_data)
                team_operation_plan_id = await self.team_framework.create_operation_plan(scenario_id)
            
            # Setup PTaaS scanning components
            if self.scanner_service:
                ptaas_session_ids = await self._setup_ptaas_sessions(configuration, "purple_team")
            
            # Configure threat hunting
            threat_hunting_queries = await self._configure_threat_hunting(operation_type, configuration)
            
            # Create adaptive strategies
            if self.ml_coordinator:
                tactical_strategies = await self._create_tactical_strategies(operation_type, configuration)
        
        elif operation_type == SecurityOperation.COMPREHENSIVE_ASSESSMENT:
            # Full security assessment with all components
            
            # PTaaS comprehensive scanning
            if self.scanner_service:
                ptaas_session_ids = await self._setup_comprehensive_scanning(configuration)
            
            # Threat hunting campaign
            threat_hunting_queries = await self._setup_threat_hunting_campaign(configuration)
            
            # Behavioral analytics monitoring
            behavioral_monitoring = True
            
            # ML-driven adaptive assessment
            if self.ml_coordinator:
                tactical_strategies = await self._create_adaptive_assessment_strategies(configuration)
        
        elif operation_type == SecurityOperation.THREAT_SIMULATION:
            # Red team simulation with blue team response
            
            # Team-based threat simulation
            if self.team_framework:
                scenario_data = await self._create_threat_simulation_scenario(configuration)
                scenario_id = await self._register_scenario(scenario_data)
                team_operation_plan_id = await self.team_framework.create_operation_plan(scenario_id)
            
            # Coordinated scanning for validation
            if self.scanner_service:
                ptaas_session_ids = await self._setup_validation_scanning(configuration)
            
            # Real-time threat hunting
            threat_hunting_queries = await self._configure_realtime_hunting(configuration)
            
            # Adaptive threat strategies
            if self.ml_coordinator:
                tactical_strategies = await self._create_threat_strategies(configuration)
        
        # Create execution timeline
        execution_timeline = await self._create_execution_timeline(operation_type, configuration)
        
        # Define success criteria
        success_criteria = await self._define_unified_success_criteria(operation_type, configuration)
        
        # Create unified operation plan
        operation_plan = UnifiedOperationPlan(
            operation_id=operation_id,
            operation_type=operation_type,
            orchestration_mode=orchestration_mode,
            team_operation_plan_id=team_operation_plan_id,
            ptaas_session_ids=ptaas_session_ids,
            scan_profiles=configuration.get("scan_profiles", ["comprehensive"]),
            threat_hunting_queries=threat_hunting_queries,
            behavioral_monitoring=configuration.get("behavioral_monitoring", True),
            tactical_strategies=tactical_strategies,
            adaptive_mechanisms=configuration.get("adaptive_mechanisms", ["ml_optimization", "real_time_adjustment"]),
            execution_timeline=execution_timeline,
            success_criteria=success_criteria,
            escalation_procedures=configuration.get("escalation_procedures", []),
            current_phase="planning",
            overall_progress=0.0,
            component_status={
                "team_operations": "ready" if team_operation_plan_id else "not_applicable",
                "ptaas_scanning": "ready" if ptaas_session_ids else "not_applicable",
                "threat_hunting": "ready" if threat_hunting_queries else "not_applicable",
                "behavioral_analytics": "ready" if self.behavioral_analytics else "not_available",
                "ml_coordination": "ready" if self.ml_coordinator else "not_available"
            },
            created_at=datetime.now(),
            started_at=None,
            completed_at=None
        )
        
        return operation_plan
    
    async def execute_unified_operation(self, operation_id: str) -> Dict[str, Any]:
        """Execute a unified security operation with full component coordination"""
        try:
            if operation_id not in self.active_operations:
                raise ValueError(f"Operation {operation_id} not found")
            
            operation = self.active_operations[operation_id]
            operation.started_at = datetime.now()
            operation.current_phase = "execution"
            
            logger.info(f"Starting unified operation execution: {operation_id}")
            
            # Start all components in coordination
            execution_results = {}
            
            # 1. Start team operations if configured
            if operation.team_operation_plan_id and self.team_framework:
                logger.info("Starting team operations component")
                team_execution_id = await self.team_framework.execute_operation(operation.team_operation_plan_id)
                execution_results["team_execution_id"] = team_execution_id
                operation.component_status["team_operations"] = "executing"
            
            # 2. Start PTaaS scanning if configured
            if operation.ptaas_session_ids and self.scanner_service:
                logger.info("Starting PTaaS scanning component")
                ptaas_results = await self._execute_ptaas_sessions(operation.ptaas_session_ids)
                execution_results["ptaas_sessions"] = ptaas_results
                operation.component_status["ptaas_scanning"] = "executing"
            
            # 3. Start threat hunting if configured
            if operation.threat_hunting_queries:
                logger.info("Starting threat hunting component")
                hunting_results = await self._execute_threat_hunting(operation.threat_hunting_queries)
                execution_results["threat_hunting"] = hunting_results
                operation.component_status["threat_hunting"] = "executing"
            
            # 4. Start behavioral monitoring if enabled
            if operation.behavioral_monitoring and self.behavioral_analytics:
                logger.info("Starting behavioral analytics monitoring")
                await self._start_behavioral_monitoring(operation_id)
                operation.component_status["behavioral_analytics"] = "monitoring"
            
            # 5. Initialize ML coordination if configured
            if operation.tactical_strategies and self.ml_coordinator:
                logger.info("Initializing ML tactical coordination")
                ml_results = await self._initialize_ml_coordination(operation.tactical_strategies)
                execution_results["ml_coordination"] = ml_results
                operation.component_status["ml_coordination"] = "coordinating"
            
            # Start real-time coordination monitoring
            asyncio.create_task(self._monitor_unified_execution(operation_id))
            
            # Update operation progress
            operation.overall_progress = 0.1  # Initial progress
            
            logger.info(f"Unified operation {operation_id} execution started successfully")
            
            return {
                "operation_id": operation_id,
                "status": "executing",
                "started_at": operation.started_at.isoformat(),
                "component_status": operation.component_status,
                "execution_results": execution_results,
                "estimated_duration": operation.execution_timeline.get("total_duration", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Failed to execute unified operation {operation_id}: {e}")
            raise
    
    async def _monitor_unified_execution(self, operation_id: str):
        """Monitor unified operation execution with cross-component coordination"""
        operation = self.active_operations[operation_id]
        
        while operation.current_phase == "execution":
            try:
                # Monitor each component
                component_progress = {}
                
                # Monitor team operations
                if operation.team_operation_plan_id and self.team_framework:
                    team_status = await self._get_team_operation_status(operation.team_operation_plan_id)
                    component_progress["team_operations"] = team_status.get("overall_progress", 0)
                
                # Monitor PTaaS scanning
                if operation.ptaas_session_ids:
                    ptaas_progress = await self._get_ptaas_progress(operation.ptaas_session_ids)
                    component_progress["ptaas_scanning"] = ptaas_progress
                
                # Monitor threat hunting
                if operation.threat_hunting_queries:
                    hunting_progress = await self._get_hunting_progress(operation.threat_hunting_queries)
                    component_progress["threat_hunting"] = hunting_progress
                
                # Calculate overall progress
                if component_progress:
                    operation.overall_progress = sum(component_progress.values()) / len(component_progress)
                
                # Check for cross-component events
                await self._process_cross_component_events(operation_id)
                
                # Apply ML-driven adaptations
                if operation.tactical_strategies and self.ml_coordinator:
                    await self._apply_ml_adaptations(operation_id)
                
                # Check completion criteria
                if operation.overall_progress >= 0.95:
                    await self._complete_unified_operation(operation_id)
                    break
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring unified operation {operation_id}: {e}")
                await asyncio.sleep(60)
    
    async def _process_cross_component_events(self, operation_id: str):
        """Process events from different components and coordinate responses"""
        operation = self.active_operations[operation_id]
        
        # Simulate cross-component event processing
        events_processed = 0
        
        # Check for PTaaS vulnerability findings that should trigger team actions
        if operation.ptaas_session_ids and operation.team_operation_plan_id:
            # In real implementation, would check actual PTaaS results
            events_processed += 1
        
        # Check for behavioral anomalies that should trigger threat hunting
        if operation.behavioral_monitoring and operation.threat_hunting_queries:
            # In real implementation, would check behavioral analytics
            events_processed += 1
        
        # Check for threat hunting findings that should trigger defensive actions
        if operation.threat_hunting_queries and operation.team_operation_plan_id:
            # In real implementation, would check hunting results
            events_processed += 1
        
        # Update metrics
        self.integration_metrics["cross_component_events"] += events_processed
    
    async def _apply_ml_adaptations(self, operation_id: str):
        """Apply ML-driven adaptations based on operation progress"""
        operation = self.active_operations[operation_id]
        
        if not self.ml_coordinator:
            return
        
        # Get current operation context
        context = {
            "operation_id": operation_id,
            "operation_type": operation.operation_type.value,
            "current_phase": operation.current_phase,
            "overall_progress": operation.overall_progress,
            "component_status": operation.component_status,
            "elapsed_time": (datetime.now() - operation.started_at).total_seconds() / 3600 if operation.started_at else 0
        }
        
        # Get ML recommendations for tactical adjustments
        try:
            # In real implementation, would use actual ML coordinator
            adaptation_recommendations = {
                "tactical_adjustments": [],
                "resource_reallocations": [],
                "priority_changes": [],
                "escalation_triggers": []
            }
            
            # Apply recommendations if confidence is high
            if adaptation_recommendations:
                logger.info(f"Applied ML adaptations for operation {operation_id}")
                
        except Exception as e:
            logger.error(f"Failed to apply ML adaptations: {e}")
    
    async def _complete_unified_operation(self, operation_id: str):
        """Complete unified operation and generate comprehensive results"""
        operation = self.active_operations[operation_id]
        operation.completed_at = datetime.now()
        operation.current_phase = "completed"
        operation.overall_progress = 1.0
        
        # Collect results from all components
        final_results = await self._collect_unified_results(operation_id)
        
        # Generate comprehensive report
        report = await self._generate_unified_report(operation_id, final_results)
        
        # Move to history
        self.operation_history.append(operation)
        del self.active_operations[operation_id]
        
        # Update metrics
        execution_time = (operation.completed_at - operation.started_at).total_seconds() / 3600
        self.integration_metrics["average_operation_time"] = (
            (self.integration_metrics["average_operation_time"] * (len(self.operation_history) - 1) + execution_time) 
            / len(self.operation_history)
        )
        
        logger.info(f"Unified operation {operation_id} completed successfully")
    
    async def get_unified_operation_status(self, operation_id: str) -> Dict[str, Any]:
        """Get comprehensive status of unified operation"""
        try:
            if operation_id not in self.active_operations:
                # Check history
                historical_operation = next(
                    (op for op in self.operation_history if op.operation_id == operation_id), 
                    None
                )
                if historical_operation:
                    return {
                        "operation_id": operation_id,
                        "status": "completed",
                        "operation_type": historical_operation.operation_type.value,
                        "overall_progress": 1.0,
                        "completed_at": historical_operation.completed_at.isoformat() if historical_operation.completed_at else None
                    }
                else:
                    return {"error": "Operation not found"}
            
            operation = self.active_operations[operation_id]
            
            # Get detailed status from each component
            component_details = {}
            
            if operation.team_operation_plan_id and self.team_framework:
                component_details["team_operations"] = await self._get_team_operation_status(operation.team_operation_plan_id)
            
            if operation.ptaas_session_ids:
                component_details["ptaas_scanning"] = await self._get_ptaas_status(operation.ptaas_session_ids)
            
            if operation.threat_hunting_queries:
                component_details["threat_hunting"] = await self._get_hunting_status(operation.threat_hunting_queries)
            
            return {
                "operation_id": operation_id,
                "operation_type": operation.operation_type.value,
                "orchestration_mode": operation.orchestration_mode.value,
                "current_phase": operation.current_phase,
                "overall_progress": operation.overall_progress,
                "component_status": operation.component_status,
                "component_details": component_details,
                "execution_timeline": operation.execution_timeline,
                "success_criteria": operation.success_criteria,
                "started_at": operation.started_at.isoformat() if operation.started_at else None,
                "estimated_completion": await self._estimate_completion_time(operation),
                "cross_component_events": len(self.cross_component_events)
            }
            
        except Exception as e:
            logger.error(f"Failed to get unified operation status: {e}")
            return {"error": str(e)}
    
    # Component-specific helper methods (mock implementations)
    async def _create_purple_team_scenario(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Create purple team scenario configuration"""
        return {
            "name": "Unified Purple Team Exercise",
            "description": "Integrated red/blue team exercise with PTaaS and ML coordination",
            "operation_type": "purple_team_exercise",
            "threat_level": configuration.get("threat_level", "high"),
            "target_environment": configuration.get("target_environment", "enterprise"),
            "objectives": configuration.get("objectives", [
                "Test detection capabilities",
                "Validate response procedures", 
                "Improve team coordination",
                "Demonstrate ML-driven adaptations"
            ]),
            "success_criteria": configuration.get("success_criteria", [
                "Achieve 85% detection rate",
                "Demonstrate real-time coordination",
                "Complete ML-driven adaptations"
            ])
        }
    
    async def _register_scenario(self, scenario_data: Dict[str, Any]) -> str:
        """Register scenario with team framework"""
        # In real implementation, would use actual team framework
        return f"scenario_{uuid.uuid4().hex[:8]}"
    
    async def _setup_ptaas_sessions(self, configuration: Dict[str, Any], context: str) -> List[str]:
        """Setup PTaaS scanning sessions"""
        # Mock implementation
        targets = configuration.get("targets", ["example.com"])
        session_ids = []
        
        for target in targets:
            session_id = f"ptaas_{context}_{uuid.uuid4().hex[:8]}"
            session_ids.append(session_id)
        
        return session_ids
    
    async def _configure_threat_hunting(self, operation_type: SecurityOperation, configuration: Dict[str, Any]) -> List[str]:
        """Configure threat hunting queries"""
        # Mock implementation
        base_queries = [
            "lateral_movement_detection",
            "suspicious_login_activity", 
            "data_exfiltration_hunt"
        ]
        
        if operation_type == SecurityOperation.PURPLE_TEAM_EXERCISE:
            return base_queries + ["purple_team_coordination"]
        
        return base_queries
    
    async def _create_tactical_strategies(self, operation_type: SecurityOperation, configuration: Dict[str, Any]) -> List[str]:
        """Create ML tactical strategies"""
        # Mock implementation
        strategies = []
        
        if operation_type == SecurityOperation.PURPLE_TEAM_EXERCISE:
            strategies.extend([
                "adaptive_red_team_strategy",
                "dynamic_blue_team_optimization",
                "real_time_coordination_enhancement"
            ])
        
        return strategies
    
    async def _create_execution_timeline(self, operation_type: SecurityOperation, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution timeline"""
        base_duration = configuration.get("duration_hours", 8)
        
        return {
            "total_duration": f"{base_duration} hours",
            "phases": {
                "preparation": "30 minutes",
                "execution": f"{base_duration - 1} hours",
                "wrap_up": "30 minutes"
            },
            "milestones": [
                {"time": "0h", "milestone": "Operation start"},
                {"time": f"{base_duration/2}h", "milestone": "Mid-operation sync"},
                {"time": f"{base_duration}h", "milestone": "Operation completion"}
            ]
        }
    
    async def _define_unified_success_criteria(self, operation_type: SecurityOperation, configuration: Dict[str, Any]) -> Dict[str, float]:
        """Define success criteria for unified operation"""
        base_criteria = {
            "overall_completion": 0.9,
            "component_integration": 0.85,
            "cross_component_coordination": 0.8,
            "ml_adaptation_effectiveness": 0.75
        }
        
        if operation_type == SecurityOperation.PURPLE_TEAM_EXERCISE:
            base_criteria.update({
                "detection_rate": 0.85,
                "response_time_improvement": 0.2,
                "team_collaboration_score": 0.8
            })
        elif operation_type == SecurityOperation.COMPREHENSIVE_ASSESSMENT:
            base_criteria.update({
                "vulnerability_coverage": 0.9,
                "threat_detection_accuracy": 0.85,
                "behavioral_anomaly_detection": 0.8
            })
        
        return base_criteria
    
    async def get_orchestrator_analytics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator analytics"""
        return {
            "orchestrator_status": {
                "active_operations": len(self.active_operations),
                "completed_operations": len(self.operation_history),
                "components_integrated": self.integration_metrics["components_integrated"],
                "last_health_check": self.integration_metrics["last_health_check"]
            },
            "integration_metrics": self.integration_metrics,
            "component_availability": {
                "team_framework": self.team_framework is not None,
                "ml_coordinator": self.ml_coordinator is not None,
                "behavioral_analytics": self.behavioral_analytics is not None,
                "scanner_service": self.scanner_service is not None,
                "intelligence_service": self.intelligence_service is not None
            },
            "operation_distribution": {
                op_type.value: len([op for op in self.operation_history if op.operation_type == op_type])
                for op_type in SecurityOperation
            },
            "performance_metrics": {
                "average_operation_time_hours": self.integration_metrics["average_operation_time"],
                "successful_integrations": self.integration_metrics["successful_integrations"],
                "failed_integrations": self.integration_metrics["failed_integrations"],
                "cross_component_events_processed": self.integration_metrics["cross_component_events"]
            },
            "recent_activities": await self._get_recent_activities()
        }
    
    async def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent orchestrator activities"""
        activities = []
        
        # Recent operations
        recent_ops = sorted(self.operation_history[-5:], key=lambda x: x.created_at, reverse=True)
        for op in recent_ops:
            activities.append({
                "type": "operation_completed",
                "operation_id": op.operation_id,
                "operation_type": op.operation_type.value,
                "completed_at": op.completed_at.isoformat() if op.completed_at else None
            })
        
        # Active operations
        for op_id, op in self.active_operations.items():
            activities.append({
                "type": "operation_active",
                "operation_id": op_id,
                "operation_type": op.operation_type.value,
                "current_phase": op.current_phase,
                "progress": op.overall_progress
            })
        
        return activities[:10]  # Return latest 10 activities
    
    # Mock helper methods for component status
    async def _get_team_operation_status(self, plan_id: str) -> Dict[str, Any]:
        """Get team operation status"""
        if self.team_framework:
            # In real implementation, would get actual status
            return {"overall_progress": 0.7, "current_phase": "execution"}
        return {"status": "not_available"}
    
    async def _get_ptaas_progress(self, session_ids: List[str]) -> float:
        """Get PTaaS scanning progress"""
        # Mock implementation
        return 0.6
    
    async def _get_hunting_progress(self, query_ids: List[str]) -> float:
        """Get threat hunting progress"""
        # Mock implementation
        return 0.8
    
    async def _estimate_completion_time(self, operation: UnifiedOperationPlan) -> str:
        """Estimate operation completion time"""
        if not operation.started_at:
            return "Not started"
        
        elapsed = datetime.now() - operation.started_at
        if operation.overall_progress > 0:
            estimated_total = elapsed / operation.overall_progress
            remaining = estimated_total - elapsed
            return f"{remaining.total_seconds() / 3600:.1f} hours remaining"
        
        return "Unknown"

# Global orchestrator instance
_unified_security_orchestrator: Optional[UnifiedSecurityOrchestrator] = None

async def get_unified_security_orchestrator() -> UnifiedSecurityOrchestrator:
    """Get global unified security orchestrator instance"""
    global _unified_security_orchestrator
    
    if _unified_security_orchestrator is None:
        _unified_security_orchestrator = UnifiedSecurityOrchestrator()
        await _unified_security_orchestrator.initialize()
    
    return _unified_security_orchestrator

# Utility functions for external integration
async def execute_comprehensive_security_assessment(targets: List[str], configuration: Dict[str, Any] = None) -> str:
    """Execute comprehensive security assessment with all components"""
    orchestrator = await get_unified_security_orchestrator()
    
    config = configuration or {}
    config.update({
        "targets": targets,
        "operation_type": "comprehensive_assessment",
        "behavioral_monitoring": True,
        "adaptive_mechanisms": ["ml_optimization", "real_time_adjustment"]
    })
    
    operation_id = await orchestrator.create_unified_operation(
        SecurityOperation.COMPREHENSIVE_ASSESSMENT, config
    )
    
    await orchestrator.execute_unified_operation(operation_id)
    return operation_id

async def execute_purple_team_exercise(scenario_config: Dict[str, Any]) -> str:
    """Execute purple team exercise with full integration"""
    orchestrator = await get_unified_security_orchestrator()
    
    operation_id = await orchestrator.create_unified_operation(
        SecurityOperation.PURPLE_TEAM_EXERCISE, scenario_config
    )
    
    await orchestrator.execute_unified_operation(operation_id)
    return operation_id

if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize orchestrator
        orchestrator = await get_unified_security_orchestrator()
        
        # Create and execute purple team exercise
        config = {
            "targets": ["example.com"],
            "threat_level": "high",
            "duration_hours": 4,
            "objectives": ["Test detection", "Improve coordination", "Validate ML adaptation"]
        }
        
        operation_id = await orchestrator.create_unified_operation(
            SecurityOperation.PURPLE_TEAM_EXERCISE, config
        )
        
        print(f"Created unified operation: {operation_id}")
        
        # Start execution
        execution_result = await orchestrator.execute_unified_operation(operation_id)
        print(f"Execution started: {execution_result}")
        
        # Monitor for a short time
        await asyncio.sleep(5)
        
        # Get status
        status = await orchestrator.get_unified_operation_status(operation_id)
        print(f"Operation status: {status}")
        
        # Get analytics
        analytics = await orchestrator.get_orchestrator_analytics()
        print(f"Orchestrator analytics: {analytics}")
    
    # Run if executed directly
    asyncio.run(main())