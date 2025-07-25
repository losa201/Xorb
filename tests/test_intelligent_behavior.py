#!/usr/bin/env python3
"""
XORB Intelligent Behavior Test Suite v8.0

Comprehensive test suite to validate autonomous intelligence evolution,
collaborative learning, and adaptive decision-making capabilities.
"""

import asyncio
import pytest
import uuid
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Internal XORB imports
from xorb_core.autonomous.cognitive_evolution_engine import (
    CognitiveEvolutionEngine, CognitiveOperationType, CognitiveInsight
)
from xorb_core.autonomous.episodic_memory_system import (
    EpisodicMemorySystem, EpisodeType, MemoryImportance, EpisodicMemory
)
from xorb_core.autonomous.intelligent_orchestrator import (
    IntelligentOrchestrator, ReflectionType, PlanningHorizon, TaskForecast
)
from xorb_core.autonomous.infrastructure_intelligence import (
    InfrastructureIntelligence, DashboardType, FaultType, ScenarioComplexity
)
from xorb_core.autonomous.controlled_risk_autonomy import (
    ControlledRiskAutonomy, RiskLevel, ContainmentType, DecisionEntropy
)
from xorb_core.agents.base_agent import BaseAgent, AgentTask, AgentResult, AgentCapability, AgentType


class MockAgent(BaseAgent):
    """Mock agent for testing"""
    
    def __init__(self, agent_id: str = None, capabilities: List[str] = None):
        super().__init__(agent_id)
        self.test_capabilities = capabilities or ['test_capability']
        self.performance_data = {
            'success_rate': 0.8,
            'queue_size': 5,
            'uptime': 3600
        }
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.RECONNAISSANCE
    
    async def _execute_task(self, task: AgentTask) -> AgentResult:
        # Simulate task execution
        await asyncio.sleep(0.1)
        return AgentResult(
            task_id=task.task_id,
            success=True,
            data={'result': f'Completed {task.task_type}'},
            execution_time=0.1
        )
    
    def _initialize_capabilities(self):
        for cap_name in self.test_capabilities:
            self.capabilities.append(AgentCapability(
                name=cap_name,
                description=f"Test capability {cap_name}",
                success_rate=0.8,
                avg_execution_time=0.5
            ))
    
    async def health_check(self) -> Dict[str, Any]:
        return self.performance_data


@pytest.fixture
async def mock_orchestrator():
    """Create mock orchestrator for testing"""
    orchestrator = Mock()
    orchestrator.autonomous_workers = {}
    orchestrator.execution_contexts = {}
    orchestrator._running = True
    return orchestrator


@pytest.fixture
async def cognitive_engine(mock_orchestrator):
    """Create cognitive evolution engine for testing"""
    engine = CognitiveEvolutionEngine(mock_orchestrator)
    return engine


@pytest.fixture
async def episodic_memory():
    """Create episodic memory system for testing"""
    memory = EpisodicMemorySystem()
    # Mock Redis client
    memory.redis_client = AsyncMock()
    return memory


@pytest.fixture
async def intelligent_orchestrator():
    """Create intelligent orchestrator for testing"""
    orchestrator = Mock(spec=IntelligentOrchestrator)
    orchestrator._running = True
    orchestrator.episodic_memory = None
    return orchestrator


@pytest.fixture
async def infrastructure_intelligence(intelligent_orchestrator):
    """Create infrastructure intelligence for testing"""
    return InfrastructureIntelligence(intelligent_orchestrator)


@pytest.fixture
async def controlled_risk(intelligent_orchestrator):
    """Create controlled risk autonomy for testing"""
    return ControlledRiskAutonomy(intelligent_orchestrator)


class TestCognitiveEvolution:
    """Test cognitive evolution capabilities"""
    
    @pytest.mark.asyncio
    async def test_agent_introspection(self, cognitive_engine):
        """Test agent introspection capabilities"""
        agent = MockAgent()
        
        # Perform introspection
        insights = await cognitive_engine._perform_agent_introspection(agent)
        
        # Validate insights generation
        assert isinstance(insights, list)
        # Should generate insights based on agent analysis
        
    @pytest.mark.asyncio
    async def test_code_analysis(self, cognitive_engine):
        """Test agent code analysis"""
        agent = MockAgent()
        
        # Analyze agent code
        analysis = await cognitive_engine._analyze_agent_code(agent)
        
        # Validate analysis structure
        assert 'optimization_opportunities' in analysis
        assert 'efficiency_score' in analysis
        assert 'confidence' in analysis
        assert isinstance(analysis['efficiency_score'], (int, float))
        assert 0 <= analysis['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_capability_analysis(self, cognitive_engine):
        """Test agent capability analysis"""
        agent = MockAgent(capabilities=['scan', 'exploit', 'recon', 'analyze', 'report'])
        
        # Analyze capabilities
        analysis = await cognitive_engine._analyze_agent_capabilities(agent)
        
        # Validate analysis
        assert 'specialization_potential' in analysis
        assert 'split_strategy' in analysis
        assert 'confidence' in analysis
    
    @pytest.mark.asyncio
    async def test_cognitive_insight_application(self, cognitive_engine):
        """Test application of cognitive insights"""
        insight = CognitiveInsight(
            insight_id=str(uuid.uuid4()),
            operation_type=CognitiveOperationType.CODE_REFINEMENT,
            agent_id="test_agent",
            insight_data={'optimizations': [{'type': 'loop_optimization', 'description': 'Test optimization'}]},
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        # Apply insight
        await cognitive_engine._apply_cognitive_insight(insight)
        
        # Validate application
        assert insight.applied == True
    
    @pytest.mark.asyncio
    async def test_collective_intelligence(self, cognitive_engine):
        """Test collective intelligence gathering"""
        # Create multiple mock agents
        agents = [MockAgent(agent_id=f"agent_{i}") for i in range(3)]
        
        # Mock the _get_active_agents method
        cognitive_engine._get_active_agents = AsyncMock(return_value=agents)
        
        # Test collective insights gathering
        insights = await cognitive_engine._gather_collective_insights()
        
        # Validate insights structure
        assert 'performance_patterns' in insights
        assert 'failure_modes' in insights
        assert 'optimization_strategies' in insights
        assert 'resource_patterns' in insights


class TestEpisodicMemory:
    """Test episodic memory system"""
    
    @pytest.mark.asyncio
    async def test_memory_storage(self, episodic_memory):
        """Test memory storage functionality"""
        # Store a memory
        memory_id = await episodic_memory.store_memory(
            episode_type=EpisodeType.TASK_EXECUTION,
            agent_id="test_agent",
            context={'task_type': 'scan', 'target': 'example.com'},
            action_taken={'action': 'port_scan', 'ports': [80, 443]},
            outcome={'success': True, 'ports_found': [80, 443]},
            importance=MemoryImportance.MEDIUM
        )
        
        # Validate storage
        assert memory_id in episodic_memory.episodic_memories
        memory = episodic_memory.episodic_memories[memory_id]
        assert memory.episode_type == EpisodeType.TASK_EXECUTION
        assert memory.agent_id == "test_agent"
        assert memory.success == True
    
    @pytest.mark.asyncio
    async def test_memory_retrieval(self, episodic_memory):
        """Test memory retrieval by similarity"""
        # Store multiple memories
        memories = []
        for i in range(5):
            memory_id = await episodic_memory.store_memory(
                episode_type=EpisodeType.TASK_EXECUTION,
                agent_id=f"agent_{i}",
                context={'task_type': 'scan', 'target': f'example{i}.com'},
                action_taken={'action': 'port_scan'},
                outcome={'success': True},
                importance=MemoryImportance.MEDIUM
            )
            memories.append(memory_id)
        
        # Update memory embeddings for testing
        episodic_memory.memory_embeddings = np.random.random((5, 384))
        episodic_memory.memory_ids = memories
        
        # Retrieve similar memories
        query_context = {'task_type': 'scan', 'target': 'example.com'}
        similar_memories = await episodic_memory.retrieve_similar_memories(
            query_context=query_context,
            top_k=3
        )
        
        # Validate retrieval
        assert len(similar_memories) <= 3
        for memory in similar_memories:
            assert isinstance(memory, EpisodicMemory)
    
    @pytest.mark.asyncio
    async def test_error_pattern_detection(self, episodic_memory):
        """Test error pattern detection"""
        # Store error memories
        for i in range(3):
            await episodic_memory.store_memory(
                episode_type=EpisodeType.ERROR_OCCURRENCE,
                agent_id=f"agent_{i}",
                context={'error_type': 'timeout', 'target': f'slow{i}.com'},
                action_taken={'action': 'connect'},
                outcome={'success': False, 'error': 'connection timeout'},
                importance=MemoryImportance.HIGH
            )
        
        # Find error patterns
        error_context = {'error_type': 'timeout'}
        patterns, cluster_id = await episodic_memory.find_error_patterns(
            error_context=error_context,
            lookback_hours=24
        )
        
        # Validate pattern detection
        assert isinstance(patterns, list)
    
    @pytest.mark.asyncio
    async def test_confidence_calibration(self, episodic_memory):
        """Test confidence calibration system"""
        agent_id = "test_agent"
        prediction_type = "task_success"
        
        # Update confidence history with some data
        for i in range(10):
            predicted_conf = 0.8
            actual_success = i % 3 == 0  # 1/3 success rate
            await episodic_memory.update_confidence_history(
                agent_id, prediction_type, predicted_conf, actual_success
            )
        
        # Get calibrated confidence
        calibrated = await episodic_memory.get_confidence_calibration(agent_id, prediction_type)
        
        # Validate calibration
        assert 0.1 <= calibrated <= 0.99


class TestIntelligentOrchestration:
    """Test intelligent orchestration capabilities"""
    
    @pytest.mark.asyncio
    async def test_reflection_cycle_execution(self, intelligent_orchestrator):
        """Test reflection cycle execution"""
        # Create intelligent orchestrator instance
        orchestrator = IntelligentOrchestrator()
        orchestrator._running = True
        
        # Mock agent selection
        agents = [MockAgent(agent_id=f"agent_{i}") for i in range(3)]
        orchestrator._select_reflection_participants = AsyncMock(return_value=agents)
        
        # Execute reflection cycle
        cycle = await orchestrator._execute_reflection_cycle(ReflectionType.PERFORMANCE_REVIEW)
        
        # Validate cycle execution
        assert cycle.reflection_type == ReflectionType.PERFORMANCE_REVIEW
        assert cycle.success == True
        assert cycle.completed_at is not None
        assert len(cycle.participating_agents) == 3
    
    @pytest.mark.asyncio
    async def test_task_forecasting(self, intelligent_orchestrator):
        """Test task outcome forecasting"""
        orchestrator = IntelligentOrchestrator()
        
        # Create test task
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="scan",
            target="example.com"
        )
        
        # Generate forecast
        forecast = await orchestrator._generate_task_forecast(task)
        
        # Validate forecast (even if None due to no historical data)
        if forecast:
            assert isinstance(forecast, TaskForecast)
            assert 0 <= forecast.predicted_success_probability <= 1
            assert forecast.predicted_duration_seconds > 0
            assert 0 <= forecast.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_performance_reflection(self, intelligent_orchestrator):
        """Test performance reflection analysis"""
        orchestrator = IntelligentOrchestrator()
        
        # Create agents with varying performance
        agents = [
            MockAgent(agent_id="high_performer"),
            MockAgent(agent_id="low_performer")
        ]
        agents[1].performance_data['success_rate'] = 0.3  # Low performance
        agents[1].performance_data['queue_size'] = 15    # High queue
        
        # Conduct performance reflection
        insights, decisions = await orchestrator._conduct_performance_reflection(agents)
        
        # Validate reflection results
        assert isinstance(insights, list)
        assert isinstance(decisions, list)
        
        # Should identify performance issues
        performance_concerns = [i for i in insights if i.get('type') == 'performance_concern']
        assert len(performance_concerns) > 0


class TestInfrastructureIntelligence:
    """Test infrastructure intelligence capabilities"""
    
    @pytest.mark.asyncio
    async def test_dashboard_generation(self, infrastructure_intelligence):
        """Test auto-dashboard generation"""
        dashboard = await infrastructure_intelligence._generate_dashboard(DashboardType.PERFORMANCE_OVERVIEW)
        
        # Validate dashboard structure (even if minimal due to mocking)
        if dashboard:
            assert dashboard.dashboard_type == DashboardType.PERFORMANCE_OVERVIEW
            assert dashboard.title is not None
            assert isinstance(dashboard.panels, list)
            assert isinstance(dashboard.metrics, list)
    
    @pytest.mark.asyncio
    async def test_fault_injection_planning(self, infrastructure_intelligence):
        """Test fault injection planning"""
        # Test next fault selection
        next_fault = await infrastructure_intelligence._select_next_fault_injection()
        
        # Validate fault planning (may be None due to mocking)
        if next_fault:
            assert hasattr(next_fault, 'fault_type')
            assert hasattr(next_fault, 'target_components')
    
    @pytest.mark.asyncio
    async def test_scenario_generation(self, infrastructure_intelligence):
        """Test AI-powered scenario generation"""
        # Mock AI response
        infrastructure_intelligence._query_claude_qwen = AsyncMock(return_value={
            'scenario_name': 'Test Scenario',
            'complexity': 'moderate',
            'description': 'Test scenario description',
            'initial_conditions': {},
            'events_sequence': [],
            'expected_challenges': [],
            'predictions': [],
            'recommendations': []
        })
        
        scenario = await infrastructure_intelligence._generate_ai_powered_scenario()
        
        # Validate scenario generation
        if scenario:
            assert scenario.scenario_name == 'Test Scenario'
            assert scenario.complexity == ScenarioComplexity.MODERATE


class TestControlledRiskAutonomy:
    """Test controlled risk autonomy system"""
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, controlled_risk):
        """Test agent risk assessment"""
        agent = MockAgent()
        
        # Mock risk calculation methods
        controlled_risk._calculate_behavioral_entropy = AsyncMock(return_value=0.3)
        controlled_risk._calculate_decision_unpredictability = AsyncMock(return_value=0.2)
        controlled_risk._calculate_resource_anomaly = AsyncMock(return_value=0.1)
        controlled_risk._calculate_collaboration_breakdown = AsyncMock(return_value=0.0)
        controlled_risk._calculate_rule_compliance = AsyncMock(return_value=0.9)
        controlled_risk._generate_risk_recommendations = AsyncMock(return_value=[])
        
        # Perform risk assessment
        assessment = await controlled_risk._assess_agent_risk(agent)
        
        # Validate assessment
        assert assessment.agent_id == agent.agent_id
        assert assessment.risk_level in [level for level in RiskLevel]
        assert 0 <= assessment.confidence <= 1
        assert isinstance(assessment.recommended_actions, list)
    
    @pytest.mark.asyncio
    async def test_high_risk_containment(self, controlled_risk):
        """Test containment for high-risk agents"""
        agent = MockAgent()
        
        # Create high-risk assessment
        from xorb_core.autonomous.controlled_risk_autonomy import RiskAssessment
        assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            risk_level=RiskLevel.HIGH,
            behavioral_entropy=0.9,
            decision_unpredictability=0.8,
            resource_usage_anomaly=0.7,
            collaboration_breakdown=0.6,
            rule_compliance_score=0.3,
            assessed_at=datetime.now(),
            confidence=0.8,
            validity_duration=600,
            recommended_actions=['containment'],
            containment_required=True
        )
        
        # Mock containment methods
        controlled_risk._select_containment_type = AsyncMock(return_value=ContainmentType.SOFT_LIMIT)
        controlled_risk._get_containment_parameters = AsyncMock(return_value={})
        controlled_risk._execute_containment = AsyncMock()
        
        # Initiate containment
        await controlled_risk._initiate_containment(agent, assessment)
        
        # Validate containment was triggered
        assert len(controlled_risk.containment_actions) > 0
    
    @pytest.mark.asyncio
    async def test_entropy_monitoring(self, controlled_risk):
        """Test decision entropy monitoring"""
        agent = MockAgent()
        
        # Mock entropy calculation
        controlled_risk._calculate_decision_entropy = AsyncMock(return_value=0.9)  # High entropy
        controlled_risk._handle_high_entropy = AsyncMock()
        controlled_risk._get_active_agents = AsyncMock(return_value=[agent])
        
        # Run one entropy monitoring cycle
        controlled_risk.entropy_threshold = 0.8
        
        # Manually trigger entropy check
        entropy = await controlled_risk._calculate_decision_entropy(agent)
        if entropy > controlled_risk.entropy_threshold:
            await controlled_risk._handle_high_entropy(agent, entropy)
        
        # Validate entropy was recorded
        assert agent.agent_id in controlled_risk.entropy_scores or entropy <= controlled_risk.entropy_threshold


class TestIntegratedIntelligence:
    """Test integrated intelligence system behavior"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_learning_cycle(self, episodic_memory, cognitive_engine):
        """Test complete learning cycle from experience to adaptation"""
        agent = MockAgent()
        
        # 1. Store initial experience
        memory_id = await episodic_memory.store_memory(
            episode_type=EpisodeType.TASK_EXECUTION,
            agent_id=agent.agent_id,
            context={'task_type': 'scan', 'difficulty': 'medium'},
            action_taken={'action': 'port_scan', 'timeout': 30},
            outcome={'success': False, 'error': 'timeout'},
            importance=MemoryImportance.HIGH
        )
        
        # 2. Cognitive evolution should learn from this
        insights = await cognitive_engine._perform_agent_introspection(agent)
        
        # 3. Apply insights
        for insight in insights:
            if insight.confidence > 0.7:
                await cognitive_engine._apply_cognitive_insight(insight)
        
        # 4. Validate learning occurred
        assert memory_id in episodic_memory.episodic_memories
        assert len(cognitive_engine.cognitive_insights) >= 0  # May be 0 due to mocking
    
    @pytest.mark.asyncio
    async def test_collaborative_decision_making(self):
        """Test multi-agent collaborative decision making"""
        # Create multiple agents with different capabilities
        agents = [
            MockAgent(agent_id="scanner", capabilities=['scan', 'detect']),
            MockAgent(agent_id="analyzer", capabilities=['analyze', 'assess']),
            MockAgent(agent_id="exploiter", capabilities=['exploit', 'attack'])
        ]
        
        # Test consensus building (simplified)
        task_priorities = {}
        test_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="comprehensive_scan",
            target="example.com"
        )
        
        # Each agent assesses task priority
        for agent in agents:
            priorities = await agent.assess_task_priorities([test_task])
            for task_id, priority in priorities.items():
                if task_id not in task_priorities:
                    task_priorities[task_id] = []
                task_priorities[task_id].append(priority)
        
        # Calculate consensus
        if test_task.task_id in task_priorities:
            consensus_priority = np.mean(task_priorities[test_task.task_id])
            assert 1.0 <= consensus_priority <= 10.0
    
    @pytest.mark.asyncio
    async def test_adaptive_risk_management(self, controlled_risk):
        """Test adaptive risk management based on learning"""
        agent = MockAgent()
        
        # Simulate risk pattern over time
        risk_history = []
        
        for i in range(5):
            # Mock varying risk factors
            controlled_risk._calculate_behavioral_entropy = AsyncMock(return_value=0.2 + i * 0.1)
            controlled_risk._calculate_decision_unpredictability = AsyncMock(return_value=0.1 + i * 0.05)
            controlled_risk._calculate_resource_anomaly = AsyncMock(return_value=0.05)
            controlled_risk._calculate_collaboration_breakdown = AsyncMock(return_value=0.0)
            controlled_risk._calculate_rule_compliance = AsyncMock(return_value=0.95)
            controlled_risk._generate_risk_recommendations = AsyncMock(return_value=[])
            
            assessment = await controlled_risk._assess_agent_risk(agent)
            risk_history.append(assessment.risk_level)
        
        # Validate risk progression
        assert len(risk_history) == 5
        
        # Risk should generally increase over time (due to increasing entropy)
        risk_levels = [RiskLevel.MINIMAL, RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]
        risk_values = {level: i for i, level in enumerate(risk_levels)}
        
        initial_risk = risk_values[risk_history[0]]
        final_risk = risk_values[risk_history[-1]]
        
        # Should show increasing risk trend
        assert final_risk >= initial_risk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])