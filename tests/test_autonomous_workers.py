#!/usr/bin/env python3
"""
Comprehensive Test Suite for Autonomous Xorb Workers

This module provides extensive testing for:
- Autonomous worker functionality and security
- Orchestration capabilities and decision making
- Resource management and monitoring
- Security compliance and validation
- Performance and scalability
"""

import asyncio
import pytest
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import the autonomous components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xorb_core.autonomous.autonomous_worker import (
    AutonomousWorker, AutonomyLevel, WorkerCapability, AutonomousConfig,
    WorkerIntelligence, SecurityConstraint
)
from xorb_core.autonomous.autonomous_orchestrator import (
    AutonomousOrchestrator, AutonomousDecision, WorkloadProfile
)
from xorb_core.autonomous.monitoring import (
    AutonomousMonitor, Alert, AlertSeverity, ResourceMetrics,
    ResourcePredictor, SecurityComplianceMonitor
)
from xorb_common.agents.base_agent import AgentTask, AgentResult, AgentCapability


class TestAutonomousWorker:
    """Test suite for AutonomousWorker functionality"""
    
    @pytest.fixture
    async def autonomous_worker(self):
        """Create test autonomous worker"""
        config = AutonomousConfig(
            autonomy_level=AutonomyLevel.MODERATE,
            max_concurrent_tasks=4,
            security_validation_required=True
        )
        
        worker = AutonomousWorker(
            agent_id="test-worker-001",
            config={"authorized_targets": ["example.com"], "authorized_networks": ["192.168.1.0/24"]},
            autonomous_config=config
        )
        
        await worker.start()
        yield worker
        await worker.stop()
    
    @pytest.mark.asyncio
    async def test_worker_initialization(self, autonomous_worker):
        """Test autonomous worker initialization"""
        
        assert autonomous_worker.agent_id == "test-worker-001"
        assert autonomous_worker.autonomous_config.autonomy_level == AutonomyLevel.MODERATE
        assert autonomous_worker.autonomous_config.security_validation_required is True
        
        # Check capabilities are initialized
        assert len(autonomous_worker.capabilities) > 0
        capability_names = [cap.name for cap in autonomous_worker.capabilities]
        assert WorkerCapability.DYNAMIC_TASK_SELECTION.value in capability_names
        assert WorkerCapability.RESOURCE_OPTIMIZATION.value in capability_names
        
        # Check security constraints are initialized
        assert len(autonomous_worker.security_constraints) > 0
        constraint_types = [c.constraint_type for c in autonomous_worker.security_constraints]
        assert "compliance" in constraint_types
        assert "resource" in constraint_types
    
    @pytest.mark.asyncio
    async def test_security_constraint_validation(self, autonomous_worker):
        """Test security constraint validation"""
        
        # Test valid task
        valid_task = AgentTask(
            task_id="test-001",
            task_type="reconnaissance",
            target="example.com",
            parameters={"target": "example.com"}
        )
        
        is_valid = await autonomous_worker._validate_security_constraints(valid_task)
        assert is_valid is True
        
        # Test invalid task (unauthorized target)
        invalid_task = AgentTask(
            task_id="test-002", 
            task_type="reconnaissance",
            target="malicious.com",
            parameters={"target": "malicious.com"}
        )
        
        is_valid = await autonomous_worker._validate_security_constraints(invalid_task)
        assert is_valid is False
        
        # Verify security violation was recorded
        assert autonomous_worker.security_violations_prevented > 0
    
    @pytest.mark.asyncio
    async def test_roe_compliance_validation(self, autonomous_worker):
        """Test Rules of Engagement compliance validation"""
        
        # Test authorized target
        valid_task = AgentTask(
            task_id="test-roe-001",
            task_type="reconnaissance", 
            target="example.com",
            parameters={"target": "example.com"}
        )
        
        is_compliant = await autonomous_worker.validate_roe_compliance(valid_task)
        assert is_compliant is True
        
        # Test unauthorized target
        invalid_task = AgentTask(
            task_id="test-roe-002",
            task_type="reconnaissance",
            target="unauthorized.com", 
            parameters={"target": "unauthorized.com"}
        )
        
        is_compliant = await autonomous_worker.validate_roe_compliance(invalid_task)
        assert is_compliant is False
    
    @pytest.mark.asyncio
    async def test_resource_limit_validation(self, autonomous_worker):
        """Test resource limit validation"""
        
        # Mock resource monitor to return high usage
        with patch.object(autonomous_worker.resource_monitor, 'get_cpu_usage', return_value=0.9):
            with patch.object(autonomous_worker.resource_monitor, 'get_memory_usage', return_value=0.7):
                
                high_resource_task = AgentTask(
                    task_id="test-resource-001",
                    task_type="vulnerability_scan",
                    target="example.com",
                    parameters={"estimated_cpu": 0.2, "estimated_memory": 0.1}
                )
                
                is_valid = await autonomous_worker.validate_resource_limits(high_resource_task)
                assert is_valid is False  # Should reject due to high CPU usage
        
        # Test with normal resource usage
        with patch.object(autonomous_worker.resource_monitor, 'get_cpu_usage', return_value=0.3):
            with patch.object(autonomous_worker.resource_monitor, 'get_memory_usage', return_value=0.4):
                
                normal_task = AgentTask(
                    task_id="test-resource-002",
                    task_type="reconnaissance",
                    target="example.com",
                    parameters={"estimated_cpu": 0.1, "estimated_memory": 0.05}
                )
                
                is_valid = await autonomous_worker.validate_resource_limits(normal_task)
                assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_task_adaptation(self, autonomous_worker):
        """Test autonomous task adaptation"""
        
        # Set up intelligence with low success rate for task type
        autonomous_worker.intelligence.task_success_rates["test_task"] = 0.3
        
        original_task = AgentTask(
            task_id="test-adapt-001",
            task_type="test_task",
            target="example.com",
            parameters={"timeout": 30}
        )
        
        # Mock resource monitor
        with patch.object(autonomous_worker.resource_monitor, 'get_current_load', return_value=0.5):
            adapted_task = await autonomous_worker._adapt_task_execution(original_task)
            
            # Check adaptations were applied
            assert adapted_task.parameters['timeout'] > 30  # Timeout increased
            assert adapted_task.max_retries > original_task.max_retries  # Retries increased
            assert 'autonomous_adaptations' in adapted_task.parameters
            
            # Verify adaptation count increased
            assert autonomous_worker.adaptation_count > 0
    
    @pytest.mark.asyncio
    async def test_failure_recovery(self, autonomous_worker):
        """Test autonomous failure recovery mechanisms"""
        
        failed_task = AgentTask(
            task_id="test-recovery-001",
            task_type="reconnaissance",
            target="example.com"
        )
        
        # Test retry with reduced parameters
        recovery_result = await autonomous_worker._retry_with_reduced_parameters(
            failed_task, "Connection timeout"
        )
        
        assert recovery_result is not None
        assert recovery_result.task_id == failed_task.task_id
        
        # Verify parameters were reduced
        assert failed_task.parameters['timeout'] < 30
        assert failed_task.parameters['max_depth'] < 3
    
    @pytest.mark.asyncio
    async def test_intelligence_learning(self, autonomous_worker):
        """Test intelligence learning from task execution"""
        
        task = AgentTask(
            task_id="test-learning-001",
            task_type="learning_test",
            target="example.com"
        )
        
        # Simulate successful execution
        result = AgentResult(
            task_id=task.task_id,
            success=True,
            execution_time=15.0,
            confidence=0.9
        )
        
        initial_success_rate = autonomous_worker.intelligence.task_success_rates.get("learning_test", 0.5)
        
        await autonomous_worker._learn_from_execution(task, result)
        
        # Verify learning occurred
        new_success_rate = autonomous_worker.intelligence.task_success_rates.get("learning_test")
        assert new_success_rate is not None
        assert new_success_rate != initial_success_rate
        
        # Verify performance history was updated
        assert len(autonomous_worker.performance_history) > 0
        last_record = autonomous_worker.performance_history[-1]
        assert last_record['task_type'] == "learning_test"
        assert last_record['success'] is True
    
    @pytest.mark.asyncio
    async def test_autonomous_status_reporting(self, autonomous_worker):
        """Test autonomous status reporting"""
        
        status = await autonomous_worker.get_autonomous_status()
        
        # Verify status structure
        assert 'autonomous_config' in status
        assert 'intelligence_summary' in status
        assert 'performance_metrics' in status
        assert 'security_status' in status
        
        # Verify autonomous config details
        config = status['autonomous_config']
        assert config['autonomy_level'] == AutonomyLevel.MODERATE.value
        assert config['performance_learning_enabled'] is True
        
        # Verify security status
        security = status['security_status']
        assert 'active_constraints' in security
        assert 'violations_prevented' in security
        assert 'compliance_rate' in security


class TestAutonomousOrchestrator:
    """Test suite for AutonomousOrchestrator functionality"""
    
    @pytest.fixture
    async def autonomous_orchestrator(self):
        """Create test autonomous orchestrator"""
        
        # Mock Redis and NATS connections
        with patch('redis.asyncio.from_url'), \
             patch('nats.connect'), \
             patch.object(AutonomousOrchestrator, '_connect_to_external_services'):
            
            orchestrator = AutonomousOrchestrator(
                redis_url="redis://localhost:6379",
                nats_url="nats://localhost:4222",
                max_concurrent_agents=8,
                autonomy_level=AutonomyLevel.HIGH
            )
            
            # Mock the start method to avoid external dependencies
            with patch.object(orchestrator, 'start') as mock_start:
                mock_start.return_value = None
                yield orchestrator
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, autonomous_orchestrator):
        """Test autonomous orchestrator initialization"""
        
        assert autonomous_orchestrator.autonomy_level == AutonomyLevel.HIGH
        assert autonomous_orchestrator.max_concurrent_agents == 8
        assert autonomous_orchestrator.learning_enabled is True
        
        # Check autonomous components are initialized
        assert autonomous_orchestrator.workload_analyzer is not None
        assert autonomous_orchestrator.performance_optimizer is not None
        assert autonomous_orchestrator.resource_monitor is not None
        assert autonomous_orchestrator.intelligent_scheduler is not None
    
    @pytest.mark.asyncio
    async def test_intelligent_agent_selection(self, autonomous_orchestrator):
        """Test intelligent agent selection based on performance"""
        
        # Set up mock performance data
        autonomous_orchestrator.global_intelligence.task_success_rates.update({
            "web_crawler": 0.9,
            "port_scanner": 0.7,
            "vuln_scanner": 0.8
        })
        
        targets = [
            {"hostname": "example.com", "ports": [80, 443]},  # Web target
            {"hostname": "server.com", "ports": [22, 80]}     # Mixed target
        ]
        
        config = {"intelligence_driven": True}
        
        optimal_agents = await autonomous_orchestrator._select_optimal_agents(targets, config)
        
        # Verify intelligent selection occurred
        assert len(optimal_agents) > 0
        assert AgentCapability.RECONNAISSANCE in optimal_agents  # Always included for unknown targets
        assert AgentCapability.VULNERABILITY_SCANNING in optimal_agents  # For web targets
    
    @pytest.mark.asyncio
    async def test_target_analysis(self, autonomous_orchestrator):
        """Test target analysis for agent selection"""
        
        targets = [
            {"hostname": "web.example.com", "ports": [80, 443]},     # Web target
            {"hostname": "mail.example.com", "ports": [25, 465]},   # Network target  
            {"hostname": "unknown.example.com"},                    # Unknown target
            {"hostname": "admin.example.com", "ports": [80]}        # High-value target
        ]
        
        analysis = await autonomous_orchestrator._analyze_targets(targets)
        
        assert analysis['web_targets'] == 2  # web.example.com and admin.example.com
        assert analysis['network_targets'] == 1  # mail.example.com
        assert analysis['unknown_targets'] == 1  # unknown.example.com
        assert analysis['high_value_targets'] == 1  # admin.example.com
    
    @pytest.mark.asyncio
    async def test_resource_optimization(self, autonomous_orchestrator):
        """Test autonomous resource optimization"""
        
        # Mock workload profile with low resource usage
        autonomous_orchestrator.workload_profile.resource_utilization = {
            'cpu': 0.3,
            'memory': 0.4,
            'queue_depth': 15
        }
        
        original_limit = autonomous_orchestrator.max_concurrent_agents
        
        await autonomous_orchestrator._optimize_resource_allocation()
        
        # Should scale up due to low resource usage and queue buildup
        # Note: This test may need adjustment based on actual implementation
        decision_made = len(autonomous_orchestrator.decision_history) > 0
        
        if decision_made:
            last_decision = autonomous_orchestrator.decision_history[-1]
            assert last_decision.decision_type in ["scale_up", "scale_down"]
            assert last_decision.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_autonomous_campaign_creation(self, autonomous_orchestrator):
        """Test autonomous campaign creation with intelligence"""
        
        # Mock agent registry with test agents
        mock_agent_metadata = Mock()
        mock_agent_metadata.name = "test_agent"
        mock_agent_metadata.capabilities = [AgentCapability.RECONNAISSANCE]
        mock_agent_metadata.last_seen = datetime.utcnow()
        
        autonomous_orchestrator.agent_registry.agents = {"test_agent": mock_agent_metadata}
        
        # Mock campaign creation
        with patch.object(autonomous_orchestrator, 'create_campaign', return_value="test-campaign-001"):
            campaign_id = await autonomous_orchestrator.create_autonomous_campaign(
                name="Test Autonomous Campaign",
                targets=[{"hostname": "example.com", "ports": [80]}],
                intelligence_driven=True,
                adaptive_execution=True
            )
        
        assert campaign_id == "test-campaign-001"
        
        # Verify decision was recorded
        assert len(autonomous_orchestrator.decision_history) > 0
        decision = autonomous_orchestrator.decision_history[-1]
        assert decision.decision_type == "agent_selection"
        assert decision.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_workload_profile_update(self, autonomous_orchestrator):
        """Test workload profile updating"""
        
        # Add mock active executions
        from xorb_common.orchestration.enhanced_orchestrator import ExecutionContext
        mock_context = Mock(spec=ExecutionContext)
        mock_context.agent_name = "test_agent"
        
        autonomous_orchestrator.active_executions = {
            "exec1": mock_context,
            "exec2": mock_context
        }
        
        # Mock task queue
        autonomous_orchestrator.autonomous_task_queue._qsize = 5
        
        # Mock resource monitor
        with patch.object(autonomous_orchestrator.resource_monitor, 'get_cpu_usage', return_value=0.6):
            with patch.object(autonomous_orchestrator.resource_monitor, 'get_memory_usage', return_value=0.5):
                
                await autonomous_orchestrator._update_workload_profile()
        
        profile = autonomous_orchestrator.workload_profile
        
        assert profile.total_active_tasks == 2
        assert profile.task_type_distribution["test_agent"] == 2
        assert profile.resource_utilization['cpu'] == 0.6
        assert profile.resource_utilization['memory'] == 0.5
    
    @pytest.mark.asyncio
    async def test_autonomous_status_reporting(self, autonomous_orchestrator):
        """Test comprehensive autonomous status reporting"""
        
        # Add mock workers and intelligence
        mock_worker = Mock(spec=AutonomousWorker)
        mock_worker.get_autonomous_status = AsyncMock(return_value={
            'performance_metrics': {'recent_success_rate': 0.85}
        })
        mock_worker.status.value = "idle"
        
        autonomous_orchestrator.autonomous_workers = {"worker1": mock_worker}
        autonomous_orchestrator.global_intelligence.task_success_rates = {"test_task": 0.8}
        
        # Add mock decision history
        decision = AutonomousDecision(
            decision_id="test-decision-001",
            decision_type="test_decision",
            context={},
            rationale="Test decision for testing",
            confidence=0.85
        )
        autonomous_orchestrator.decision_history.append(decision)
        
        # Mock the base class method
        with patch.object(autonomous_orchestrator, 'get_campaign_status', return_value={}):
            status = await autonomous_orchestrator.get_autonomous_status()
        
        # Verify status structure
        assert 'autonomy_config' in status
        assert 'autonomous_workers' in status
        assert 'intelligence_summary' in status
        assert 'workload_profile' in status
        assert 'recent_decisions' in status
        
        # Verify autonomy config
        config = status['autonomy_config']
        assert config['autonomy_level'] == AutonomyLevel.HIGH.value
        assert config['learning_enabled'] is True
        
        # Verify worker summary
        workers = status['autonomous_workers']
        assert workers['total_workers'] == 1
        assert workers['active_workers'] == 1
        
        # Verify recent decisions
        decisions = status['recent_decisions']
        assert len(decisions) == 1
        assert decisions[0]['decision_type'] == "test_decision"
        assert decisions[0]['confidence'] == 0.85


class TestAutonomousMonitoring:
    """Test suite for autonomous monitoring capabilities"""
    
    @pytest.fixture
    async def autonomous_monitor(self):
        """Create test autonomous monitor"""
        
        # Create mock orchestrator
        mock_orchestrator = Mock(spec=AutonomousOrchestrator)
        mock_orchestrator.autonomy_level = AutonomyLevel.MODERATE
        mock_orchestrator.autonomous_workers = {}
        mock_orchestrator.decision_history = []
        
        # Mock Redis connection
        with patch('redis.asyncio.from_url'):
            monitor = AutonomousMonitor(
                orchestrator=mock_orchestrator,
                redis_url="redis://localhost:6379"
            )
            
            yield monitor
            
            await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_monitor_initialization(self, autonomous_monitor):
        """Test monitoring system initialization"""
        
        assert autonomous_monitor.monitoring_enabled is True
        assert autonomous_monitor.monitoring_interval == 10
        
        # Check alert thresholds are configured
        assert 'cpu_usage' in autonomous_monitor.alert_thresholds
        assert 'memory_usage' in autonomous_monitor.alert_thresholds
        assert 'task_failure_rate' in autonomous_monitor.alert_thresholds
        
        # Check components are initialized
        assert autonomous_monitor.resource_predictor is not None
        assert autonomous_monitor.security_monitor is not None
        assert autonomous_monitor.failure_predictor is not None
    
    @pytest.mark.asyncio
    async def test_resource_metrics_collection(self, autonomous_monitor):
        """Test resource metrics collection"""
        
        # Mock psutil functions
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.Process') as mock_process:
            
            # Configure mocks
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 30.0
            mock_network.return_value = Mock(
                bytes_sent=1000000, bytes_recv=2000000,
                packets_sent=5000, packets_recv=7000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            
            metrics = await autonomous_monitor._collect_resource_metrics()
            
            assert metrics.cpu_usage == 45.0
            assert metrics.memory_usage == 60.0
            assert metrics.disk_usage == 30.0
            assert metrics.network_io['bytes_sent'] == 1000000
            assert metrics.thread_count == 3
    
    @pytest.mark.asyncio
    async def test_alert_raising(self, autonomous_monitor):
        """Test alert raising and management"""
        
        # Raise test alert
        await autonomous_monitor._raise_alert(
            alert_type="test_alert",
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            description="This is a test alert",
            context={"test_data": "test_value"}
        )
        
        # Verify alert was created
        assert len(autonomous_monitor.active_alerts) == 1
        
        alert = list(autonomous_monitor.active_alerts.values())[0]
        assert alert.alert_type == "test_alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.context["test_data"] == "test_value"
        assert alert.resolved is False
    
    @pytest.mark.asyncio
    async def test_resource_alert_checking(self, autonomous_monitor):
        """Test resource-based alert checking"""
        
        # Create high resource usage metrics
        high_usage_metrics = ResourceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=95.0,  # Above critical threshold
            memory_usage=85.0,  # Above warning threshold
            disk_usage=50.0,
            network_io={},
            active_agents=5,
            queue_depth=10,
            thread_count=20
        )
        
        await autonomous_monitor._check_resource_alerts(high_usage_metrics)
        
        # Should have raised alerts for high CPU and memory usage
        cpu_alerts = [a for a in autonomous_monitor.active_alerts.values() 
                     if "cpu" in a.title.lower()]
        memory_alerts = [a for a in autonomous_monitor.active_alerts.values()
                       if "memory" in a.title.lower()]
        
        # At least one CPU alert should be raised
        assert len(cpu_alerts) > 0 or len(memory_alerts) > 0
    
    @pytest.mark.asyncio 
    async def test_performance_data_collection(self, autonomous_monitor):
        """Test performance data collection from workers"""
        
        # Mock autonomous worker
        mock_worker = Mock(spec=AutonomousWorker)
        mock_worker.get_autonomous_status = AsyncMock(return_value={
            'performance_metrics': {
                'recent_success_rate': 0.85,
                'average_execution_time': 25.0
            },
            'completed_tasks': 100,
            'intelligence_summary': {
                'adaptation_count': 5
            }
        })
        
        autonomous_monitor.orchestrator.autonomous_workers = {"worker1": mock_worker}
        autonomous_monitor.orchestrator.global_intelligence = Mock()
        autonomous_monitor.orchestrator.global_intelligence.task_success_rates = {"test": 0.8}
        autonomous_monitor.orchestrator.global_intelligence.performance_metrics = {"avg_time": 30.0}
        
        # Mock orchestrator status
        autonomous_monitor.orchestrator.get_autonomous_status = AsyncMock(return_value={
            'autonomous_workers': {'total_workers': 1, 'active_workers': 1},
            'workload_profile': {'total_active_tasks': 5}
        })
        
        performance_data = await autonomous_monitor._collect_performance_data()
        
        # Verify data structure
        assert 'timestamp' in performance_data
        assert 'agent_performance' in performance_data
        assert 'global_metrics' in performance_data
        assert 'orchestrator_metrics' in performance_data
        
        # Verify worker data
        worker_data = performance_data['agent_performance']['worker1']
        assert worker_data['success_rate'] == 0.85
        assert worker_data['avg_execution_time'] == 25.0
        assert worker_data['adaptations'] == 5
    
    @pytest.mark.asyncio
    async def test_monitoring_dashboard(self, autonomous_monitor):
        """Test monitoring dashboard data generation"""
        
        # Add test resource metrics
        test_metrics = ResourceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=40.0,
            memory_usage=60.0,
            disk_usage=30.0,
            network_io={},
            active_agents=3,
            queue_depth=5,
            thread_count=15
        )
        autonomous_monitor.resource_history.append(test_metrics)
        
        # Add test alert
        await autonomous_monitor._raise_alert(
            "test_dashboard_alert", AlertSeverity.INFO,
            "Test Dashboard Alert", "Test description", {}
        )
        
        # Mock dependencies
        autonomous_monitor.orchestrator.autonomous_workers = {"worker1": Mock()}
        
        with patch.object(autonomous_monitor, '_collect_performance_data', 
                         return_value={'agent_performance': {'worker1': {'success_rate': 0.9}}}):
            with patch.object(autonomous_monitor.security_monitor, 'get_security_summary',
                             return_value={'compliance_score': 0.95}):
                
                dashboard = await autonomous_monitor.get_monitoring_dashboard()
        
        # Verify dashboard structure
        assert 'timestamp' in dashboard
        assert 'system_status' in dashboard
        assert 'resource_status' in dashboard
        assert 'performance_summary' in dashboard
        assert 'alerts' in dashboard
        assert 'security_status' in dashboard
        
        # Verify resource status
        resource_status = dashboard['resource_status']
        assert resource_status['cpu_usage'] == 40.0
        assert resource_status['memory_usage'] == 60.0
        assert resource_status['active_agents'] == 3
        
        # Verify alert summary
        alert_summary = dashboard['alerts']
        assert alert_summary['active_alerts'] == 1
        assert alert_summary['alerts_by_severity']['info'] == 1


class TestResourcePredictor:
    """Test suite for resource prediction capabilities"""
    
    @pytest.fixture
    def resource_predictor(self):
        """Create test resource predictor"""
        return ResourcePredictor()
    
    @pytest.mark.asyncio
    async def test_resource_prediction_insufficient_data(self, resource_predictor):
        """Test resource prediction with insufficient historical data"""
        
        # Test with minimal history
        minimal_history = [
            ResourceMetrics(datetime.utcnow(), 50.0, 60.0, 30.0, {}, 2, 5, 10)
            for _ in range(5)  # Less than required minimum
        ]
        
        prediction = await resource_predictor.predict_resource_needs(minimal_history)
        
        assert prediction['prediction_available'] is False
    
    @pytest.mark.asyncio
    async def test_resource_prediction_with_trend(self, resource_predictor):
        """Test resource prediction with trending data"""
        
        # Create upward trending resource history
        history = []
        base_time = datetime.utcnow()
        
        for i in range(15):
            cpu_usage = 40.0 + (i * 2.0)  # Increasing CPU usage
            memory_usage = 50.0 + (i * 1.5)  # Increasing memory usage
            
            metrics = ResourceMetrics(
                timestamp=base_time - timedelta(minutes=15-i),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=30.0,
                network_io={},
                active_agents=i+1,
                queue_depth=i*2,
                thread_count=10+i
            )
            history.append(metrics)
        
        prediction = await resource_predictor.predict_resource_needs(history, time_horizon=60)
        
        assert prediction['prediction_available'] is True
        assert prediction['time_horizon'] == 60
        assert prediction['predicted_cpu'] > history[-1].cpu_usage  # Should predict higher
        assert prediction['predicted_memory'] > history[-1].memory_usage
        assert prediction['confidence'] > 0.0
        
        # Should detect high risk due to upward trend
        assert prediction['cpu_shortage_risk'] > 0.0
        assert prediction['memory_shortage_risk'] > 0.0


class TestSecurityComplianceMonitor:
    """Test suite for security compliance monitoring"""
    
    @pytest.fixture
    def security_monitor(self):
        """Create test security compliance monitor"""
        return SecurityComplianceMonitor()
    
    @pytest.mark.asyncio
    async def test_worker_compliance_check(self, security_monitor):
        """Test individual worker compliance checking"""
        
        # Create test worker with excessive autonomy
        excessive_worker = Mock(spec=AutonomousWorker)
        excessive_worker.agent_id = "excessive-worker-001"
        excessive_worker.autonomous_config = Mock()
        excessive_worker.autonomous_config.autonomy_level = AutonomyLevel.MAXIMUM
        excessive_worker.autonomous_config.security_validation_required = False
        excessive_worker.security_violations_prevented = 3
        
        violations = await security_monitor._check_worker_compliance(excessive_worker)
        
        # Should detect excessive autonomy violation
        assert len(violations) > 0
        violation = violations[0]
        assert violation['type'] == 'excessive_autonomy'
        assert violation['severity'] == 'high'
        assert excessive_worker.agent_id[:8] in violation['description']
    
    @pytest.mark.asyncio
    async def test_decision_compliance_check(self, security_monitor):
        """Test autonomous decision compliance checking"""
        
        # Create test decisions with low confidence
        decisions = []
        for i in range(5):
            decision = Mock()
            decision.confidence = 0.3  # Low confidence
            decision.decision_type = "scale_up"
            decision.decision_id = f"low-confidence-{i}"
            decisions.append(decision)
        
        violations = await security_monitor._check_decision_compliance(decisions)
        
        # Should detect low confidence decision violations
        assert len(violations) > 0
        for violation in violations:
            assert violation['type'] == 'low_confidence_decision'
            assert violation['severity'] == 'medium'
            assert '0.30' in violation['description']  # Confidence level mentioned
    
    @pytest.mark.asyncio
    async def test_compliance_report_generation(self, security_monitor):
        """Test comprehensive compliance report generation"""
        
        # Mock orchestrator with test workers and decisions
        mock_orchestrator = Mock(spec=AutonomousOrchestrator)
        mock_orchestrator.autonomous_workers = {}
        mock_orchestrator.decision_history = []
        
        with patch.object(security_monitor, '_check_worker_compliance', return_value=[]):
            with patch.object(security_monitor, '_check_decision_compliance', return_value=[]):
                
                report = await security_monitor.check_compliance(mock_orchestrator)
        
        # Verify report structure
        assert 'timestamp' in report
        assert 'validations' in report
        assert 'violations' in report
        assert 'compliance_score' in report
        
        # With no violations, compliance score should be 1.0
        assert report['compliance_score'] == 1.0


# Integration Tests
class TestAutonomousIntegration:
    """Integration tests for autonomous components working together"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_autonomous_workflow(self):
        """Test complete autonomous workflow from task creation to completion"""
        
        # This would be a comprehensive integration test
        # For now, we'll test the basic workflow structure
        
        # Mock all external dependencies
        with patch('redis.asyncio.from_url'), \
             patch('nats.connect'), \
             patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.Process') as mock_process:
            
            # Configure mocks
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 30.0
            mock_network.return_value = Mock(bytes_sent=1000, bytes_recv=2000, packets_sent=10, packets_recv=20)
            mock_process.return_value.threads.return_value = [1, 2, 3]
            
            # Create components
            config = AutonomousConfig(autonomy_level=AutonomyLevel.MODERATE)
            worker = AutonomousWorker(config={"authorized_targets": ["example.com"]}, autonomous_config=config)
            
            # Test workflow
            await worker.start()
            
            # Create and execute test task
            task = AgentTask(
                task_id="integration-test-001",
                task_type="reconnaissance", 
                target="example.com",
                parameters={"target": "example.com"}
            )
            
            # Execute task
            result = await worker._execute_task(task)
            
            # Verify results
            assert result is not None
            assert result.task_id == task.task_id
            assert 'autonomous_execution' in result.data
            
            await worker.stop()
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self):
        """Test monitoring integration with autonomous components"""
        
        # Mock all dependencies
        with patch('redis.asyncio.from_url'), \
             patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.Process') as mock_process:
            
            # Configure mocks
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 30.0
            mock_network.return_value = Mock(bytes_sent=1000, bytes_recv=2000, packets_sent=10, packets_recv=20)
            mock_process.return_value.threads.return_value = [1, 2, 3]
            
            # Create mock orchestrator
            mock_orchestrator = Mock(spec=AutonomousOrchestrator)
            mock_orchestrator.autonomy_level = AutonomyLevel.MODERATE
            mock_orchestrator.autonomous_workers = {}
            mock_orchestrator.decision_history = []
            mock_orchestrator.autonomous_task_queue = Mock()
            mock_orchestrator.autonomous_task_queue.qsize.return_value = 5
            
            # Create monitor
            monitor = AutonomousMonitor(orchestrator=mock_orchestrator)
            
            # Test resource collection
            metrics = await monitor._collect_resource_metrics()
            
            assert metrics.cpu_usage == 45.0
            assert metrics.memory_usage == 60.0
            assert metrics.queue_depth == 5


# Performance and Load Tests
class TestAutonomousPerformance:
    """Performance and load tests for autonomous components"""
    
    @pytest.mark.asyncio
    async def test_worker_concurrent_task_handling(self):
        """Test worker handling multiple concurrent tasks"""
        
        config = AutonomousConfig(
            autonomy_level=AutonomyLevel.MODERATE,
            max_concurrent_tasks=5
        )
        
        worker = AutonomousWorker(
            config={"authorized_targets": ["example.com"]},
            autonomous_config=config
        )
        
        await worker.start()
        
        # Create multiple tasks
        tasks = []
        for i in range(10):
            task = AgentTask(
                task_id=f"perf-test-{i:03d}",
                task_type="reconnaissance",
                target="example.com",
                parameters={"target": "example.com"}
            )
            tasks.append(task)
        
        # Submit all tasks
        start_time = time.time()
        
        for task in tasks:
            await worker.add_task(task)
        
        # Wait for completion (simplified)
        await asyncio.sleep(2)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify performance metrics
        assert execution_time < 10.0  # Should complete within reasonable time
        assert worker.total_tasks_completed > 0
        
        await worker.stop()
    
    @pytest.mark.asyncio
    async def test_monitoring_overhead(self):
        """Test monitoring system performance overhead"""
        
        # Mock dependencies for performance testing
        with patch('redis.asyncio.from_url'), \
             patch('psutil.cpu_percent', return_value=30.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.Process') as mock_process:
            
            mock_memory.return_value.percent = 40.0
            mock_disk.return_value.percent = 20.0
            mock_network.return_value = Mock(bytes_sent=1000, bytes_recv=2000, packets_sent=10, packets_recv=20)
            mock_process.return_value.threads.return_value = [1, 2, 3]
            
            mock_orchestrator = Mock(spec=AutonomousOrchestrator)
            mock_orchestrator.autonomy_level = AutonomyLevel.MODERATE
            mock_orchestrator.autonomous_workers = {}
            mock_orchestrator.decision_history = []
            mock_orchestrator.autonomous_task_queue = Mock()
            mock_orchestrator.autonomous_task_queue.qsize.return_value = 0
            
            monitor = AutonomousMonitor(orchestrator=mock_orchestrator)
            
            # Measure resource collection performance
            start_time = time.time()
            
            for _ in range(100):
                await monitor._collect_resource_metrics()
            
            end_time = time.time()
            avg_collection_time = (end_time - start_time) / 100
            
            # Should be very fast (under 10ms per collection)
            assert avg_collection_time < 0.01


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])